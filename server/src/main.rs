mod db;
#[cfg(test)]
mod tests;

use actix_web::{
    body::{BoxBody, MessageBody},
    dev::{ServiceRequest, ServiceResponse},
    http::StatusCode,
    middleware::{self, Next},
    web, App, HttpMessage, HttpRequest, HttpResponse, HttpServer,
};
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

struct AppState {
    db: db::Executor,
    admission: Arc<Admission>,
    submission_quota: u64,
}
struct Admission {
    limiter: Mutex<RateLimiter>,
    in_flight: Arc<tokio::sync::Semaphore>,
}
struct Window {
    started: Instant,
    count: u32,
}
struct RateLimiter {
    peers: HashMap<IpAddr, Window>,
    global: Window,
    window: Duration,
    peer_max: u32,
    global_max: u32,
}
struct RateRejection {
    message: String,
    retry_after: u64,
}
impl RateLimiter {
    fn new(peer_max: u32, global_max: u32, window_secs: u64) -> Result<Self, &'static str> {
        if !(1..=100_000).contains(&peer_max)
            || !(1..=1_000_000).contains(&global_max)
            || !(1..=86_400).contains(&window_secs)
        {
            return Err("invalid rate configuration");
        }
        Ok(Self {
            peers: HashMap::new(),
            global: Window {
                started: Instant::now(),
                count: 0,
            },
            window: Duration::from_secs(window_secs),
            peer_max,
            global_max,
        })
    }
    fn check(&mut self, peer: IpAddr) -> Result<(), RateRejection> {
        let now = Instant::now();
        self.peers
            .retain(|_, w| now.duration_since(w.started) < self.window);
        if now.duration_since(self.global.started) >= self.window {
            self.global = Window {
                started: now,
                count: 0,
            };
        }
        if self.global.count >= self.global_max {
            return Err(self.rejection("global", self.global_max, &self.global, now));
        }
        if !self.peers.contains_key(&peer) && self.peers.len() >= 1024 {
            return Err(RateRejection {
                message: "Peer admission capacity reached".into(),
                retry_after: self.window.as_secs(),
            });
        }
        if let Some(w) = self.peers.get(&peer) {
            if w.count >= self.peer_max {
                return Err(self.rejection("peer", self.peer_max, w, now));
            }
        }
        self.global.count += 1;
        let entry = self.peers.entry(peer).or_insert(Window {
            started: now,
            count: 0,
        });
        entry.count += 1;
        Ok(())
    }
    fn rejection(&self, scope: &str, max: u32, window: &Window, now: Instant) -> RateRejection {
        let remaining = self
            .window
            .saturating_sub(now.duration_since(window.started));
        RateRejection {
            message: format!(
                "Rate limit exceeded: {max} requests per {} seconds per {scope}",
                self.window.as_secs()
            ),
            retry_after: remaining
                .as_secs()
                .saturating_add(u64::from(remaining.subsec_nanos() > 0))
                .max(1),
        }
    }
}
impl Admission {
    fn new(peer_max: u32, global_max: u32, window: u64) -> Result<Self, &'static str> {
        Ok(Self {
            limiter: Mutex::new(RateLimiter::new(peer_max, global_max, window)?),
            in_flight: Arc::new(tokio::sync::Semaphore::new(16)),
        })
    }
}
fn failure(status: StatusCode, message: &str) -> HttpResponse {
    HttpResponse::build(status).json(serde_json::json!({"error":message}))
}
fn unavailable() -> HttpResponse {
    HttpResponse::ServiceUnavailable()
        .insert_header(("Retry-After", "1"))
        .json(serde_json::json!({"error":"Service temporarily unavailable"}))
}
fn db_failure(error: db::Error) -> HttpResponse {
    match error {
        db::Error::Unavailable => unavailable(),
        db::Error::OutcomeUnknown => failure(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Submission outcome unknown; do not automatically retry",
        ),
        db::Error::Quota => failure(StatusCode::INSUFFICIENT_STORAGE, "Storage quota reached"),
        db::Error::ResponseTooLarge => failure(
            StatusCode::BAD_REQUEST,
            "Response exceeds 2 MiB; request a smaller page",
        ),
        db::Error::Sql(rusqlite::Error::SqliteFailure(code, _))
            if matches!(
                code.code,
                rusqlite::ErrorCode::DatabaseBusy
                    | rusqlite::ErrorCode::DatabaseLocked
                    | rusqlite::ErrorCode::OperationInterrupted
            ) =>
        {
            unavailable()
        }
        db::Error::Sql(rusqlite::Error::SqliteFailure(code, _))
            if code.code == rusqlite::ErrorCode::DiskFull =>
        {
            failure(StatusCode::INSUFFICIENT_STORAGE, "Storage quota reached")
        }
        _ => failure(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Database operation failed",
        ),
    }
}
fn json_response(result: db::Result<Vec<u8>>) -> HttpResponse {
    match result {
        Ok(body) => HttpResponse::Ok()
            .content_type("application/json")
            .body(body),
        Err(error) => db_failure(error),
    }
}
/// The header is an anonymous installation identifier, not account authentication.
fn optional_key(req: &HttpRequest) -> Result<Option<String>, &'static str> {
    let Some(header) = req.headers().get("X-API-Key") else {
        return Ok(None);
    };
    let Ok(key) = header.to_str() else {
        return Err("Invalid X-API-Key header");
    };
    if key.is_empty() {
        return Ok(None);
    }
    if key.len() > 128 || !key.bytes().all(|b| b.is_ascii_graphic()) {
        return Err("Invalid X-API-Key header");
    }
    Ok(Some(key.to_owned()))
}
async fn admission<B: MessageBody + 'static>(
    req: ServiceRequest,
    next: Next<B>,
) -> Result<ServiceResponse<BoxBody>, actix_web::Error> {
    // Liveness is independent of admission and database availability.
    if req.path() == "/health" {
        return Ok(next.call(req).await?.map_into_boxed_body());
    }
    let Some(state) = req.app_data::<web::Data<AppState>>() else {
        return Ok(req.into_response(unavailable()));
    };
    let state = state.clone();
    let permit = match state.admission.in_flight.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => return Ok(req.into_response(unavailable())),
    };
    let peer = req
        .peer_addr()
        .map(|a| a.ip())
        .unwrap_or(IpAddr::V4(Ipv4Addr::UNSPECIFIED));
    let rejection = match state.admission.limiter.try_lock() {
        Ok(mut limiter) => limiter.check(peer).err(),
        Err(_) => return Ok(req.into_response(unavailable())),
    };
    if let Some(rejection) = rejection {
        let response = HttpResponse::TooManyRequests()
            .insert_header(("Retry-After", rejection.retry_after.to_string()))
            .json(serde_json::json!({"error":rejection.message}));
        return Ok(req.into_response(response));
    }
    if req.uri().to_string().len() > 8192 {
        return Ok(req.into_response(failure(StatusCode::URI_TOO_LONG, "Request URI too long")));
    }
    if req.path() == "/api/v1/benchmark" && req.method() == actix_web::http::Method::POST {
        match optional_key(req.request()) {
            Ok(Some(_)) => {}
            Ok(None) => {
                return Ok(req.into_response(failure(
                    StatusCode::UNAUTHORIZED,
                    "Missing X-API-Key header",
                )))
            }
            Err(message) => return Ok(req.into_response(failure(StatusCode::BAD_REQUEST, message))),
        }
    }
    let deadline = Instant::now() + Duration::from_secs(5);
    let operation = db::Operation::new(deadline);
    req.extensions_mut().insert(operation.clone());
    let response = request_outcome(next.call(req), operation, deadline)
        .await?
        .map_into_boxed_body();
    drop(permit);
    Ok(response)
}
async fn request_outcome<T>(
    future: impl std::future::Future<Output = Result<T, actix_web::Error>>,
    operation: db::Operation,
    deadline: Instant,
) -> Result<T, actix_web::Error> {
    tokio::pin!(future);
    match tokio::time::timeout_at(deadline.into(), &mut future).await {
        Ok(result) => result,
        Err(_) if operation.cancel() => Err(actix_web::error::ErrorRequestTimeout(
            "Request deadline exceeded",
        )),
        // A durable commit may be in progress. Do not replace its result with
        // a timeout that tells a client it is safe to retry.
        Err(_) => future.await,
    }
}
async fn health() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({"status":"ok","service":"autarky-server","version":env!("CARGO_PKG_VERSION")}))
}
async fn ready(state: web::Data<AppState>) -> HttpResponse {
    match state
        .db
        .call(|conn, _| {
            conn.query_row("SELECT 1 FROM api_keys LIMIT 1", [], |r| r.get::<_, i64>(0))
                .optional()?;
            Ok(())
        })
        .await
    {
        Ok(()) => HttpResponse::Ok().json(serde_json::json!({"status":"ready"})),
        Err(error) => db_failure(error),
    }
}
use rusqlite::OptionalExtension;
async fn submit(
    state: web::Data<AppState>,
    req: HttpRequest,
    payload: web::Json<db::SubmitPayload>,
) -> HttpResponse {
    let key = match optional_key(&req) {
        Ok(Some(key)) => key,
        Ok(None) => return failure(StatusCode::UNAUTHORIZED, "Missing X-API-Key header"),
        Err(message) => return failure(StatusCode::BAD_REQUEST, message),
    };
    let mut payload = payload.into_inner();
    if let Err(error) = payload.validate() {
        let message = match error {
            db::Error::Invalid(message) => message,
            _ => "Invalid benchmark payload",
        };
        return failure(StatusCode::BAD_REQUEST, message);
    }
    payload.remove_private_errors();
    let quota = state.submission_quota;
    let operation = req
        .extensions()
        .get::<db::Operation>()
        .cloned()
        .unwrap_or_else(|| db::Operation::new(Instant::now() + db::DB_DEADLINE));
    let result = state
        .db
        .call_with(operation, move |conn, deadline| {
            let id = db::store_submission(conn, &key, &payload, quota, deadline)?;
            db::encode(&serde_json::json!({"status":"submitted","id":id}))
        })
        .await;
    json_response(result)
}
async fn query(
    state: web::Data<AppState>,
    req: HttpRequest,
    query: web::Query<db::QueryParams>,
) -> HttpResponse {
    let key = match optional_key(&req) {
        Ok(key) => key,
        Err(message) => return failure(StatusCode::BAD_REQUEST, message),
    };
    let query = query.into_inner();
    if query.validate().is_err() {
        return failure(
            StatusCode::BAD_REQUEST,
            "Invalid query filters or pagination",
        );
    }
    json_response(
        state
            .db
            .call(move |conn, _| {
                let exclude = key
                    .as_deref()
                    .map(|key| db::lookup_key(conn, key))
                    .transpose()?
                    .flatten();
                let results = db::query_submissions(conn, &query, exclude, None, None)?;
                db::encode(&serde_json::json!({"count":results.len(),"results":results}))
            })
            .await,
    )
}
async fn model_benchmarks(
    state: web::Data<AppState>,
    path: web::Path<String>,
    query: web::Query<db::QueryParams>,
) -> HttpResponse {
    let model = path.into_inner();
    let query = query.into_inner();
    if model.is_empty() || model.len() > 256 || query.validate().is_err() {
        return failure(StatusCode::BAD_REQUEST, "Invalid model or pagination");
    }
    json_response(
        state
            .db
            .call(move |conn, _| {
                let results = db::query_submissions(conn, &query, None, Some(&model), None)?;
                db::encode(
                    &serde_json::json!({"model":model,"count":results.len(),"results":results}),
                )
            })
            .await,
    )
}
async fn hardware_benchmarks(
    state: web::Data<AppState>,
    path: web::Path<i64>,
    query: web::Query<db::QueryParams>,
) -> HttpResponse {
    let hardware_id = path.into_inner();
    let query = query.into_inner();
    if hardware_id <= 0 || query.validate().is_err() {
        return failure(StatusCode::BAD_REQUEST, "Invalid hardware ID or pagination");
    }
    json_response(state.db.call(move |conn,_| {
        let results=db::query_submissions(conn,&query,None,None,Some(hardware_id))?;
        db::encode(&serde_json::json!({"hardware_id":hardware_id,"count":results.len(),"results":results}))
    }).await)
}
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CompareRequest {
    id1: i64,
    id2: i64,
}
async fn compare(state: web::Data<AppState>, payload: web::Json<CompareRequest>) -> HttpResponse {
    let ids = payload.into_inner();
    if ids.id1 <= 0 || ids.id2 <= 0 || ids.id1 == ids.id2 {
        return failure(
            StatusCode::BAD_REQUEST,
            "Two distinct positive benchmark IDs are required",
        );
    }
    match state
        .db
        .call(move |conn, _| {
            let Some(comparison) = db::compare_pair(conn, ids.id1, ids.id2)? else {
                return Ok(None);
            };
            let mut delta = serde_json::Map::new();
            if let Some(value) = comparison.tps.percent {
                delta.insert("tps_percent".into(), format!("{value:+.1}%").into());
            }
            if let Some(value) = comparison.prompt.percent {
                delta.insert("prompt_percent".into(), format!("{value:+.1}%").into());
            }
            let mut value = serde_json::to_value(comparison)?;
            value["delta"] = delta.into();
            Ok(Some(db::encode(&value)?))
        })
        .await
    {
        Ok(Some(body)) => HttpResponse::Ok()
            .content_type("application/json")
            .body(body),
        Ok(None) => failure(StatusCode::NOT_FOUND, "One or both benchmark IDs not found"),
        Err(error) => db_failure(error),
    }
}
fn routes(config: &mut web::ServiceConfig) {
    config
        .app_data(
            web::JsonConfig::default()
                .limit(db::MAX_BODY)
                .error_handler(|error, _| {
                    let status = match error {
                        actix_web::error::JsonPayloadError::Overflow { .. }
                        | actix_web::error::JsonPayloadError::OverflowKnownLength { .. } => {
                            StatusCode::PAYLOAD_TOO_LARGE
                        }
                        _ => StatusCode::BAD_REQUEST,
                    };
                    actix_web::error::InternalError::from_response(
                        "invalid JSON",
                        failure(status, "Invalid JSON payload"),
                    )
                    .into()
                }),
        )
        .app_data(web::QueryConfig::default().error_handler(|_, _| {
            actix_web::error::InternalError::from_response(
                "invalid query",
                failure(StatusCode::BAD_REQUEST, "Invalid query parameters"),
            )
            .into()
        }))
        .app_data(web::PathConfig::default().error_handler(|_, _| {
            actix_web::error::InternalError::from_response(
                "invalid path",
                failure(StatusCode::BAD_REQUEST, "Invalid path parameters"),
            )
            .into()
        }))
        .route("/health", web::get().to(health))
        .route("/ready", web::get().to(ready))
        .route("/api/v1/benchmark", web::post().to(submit))
        .route("/api/v1/benchmarks", web::get().to(query))
        .route(
            "/api/v1/models/{model_name}",
            web::get().to(model_benchmarks),
        )
        .route(
            "/api/v1/hardware/{hardware_id}/benchmarks",
            web::get().to(hardware_benchmarks),
        )
        .route("/api/v1/compare", web::post().to(compare));
}
fn default_db_path() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("autarky-server/submissions.db")
}
fn bounded(
    name: &str,
    value: Option<&str>,
    default: u64,
    min: u64,
    max: u64,
) -> std::io::Result<u64> {
    let number = match value {
        Some(value) => value.parse::<u64>().map_err(|_| {
            std::io::Error::other(format!("{name} must be an integer in {min}..={max}"))
        })?,
        None => default,
    };
    if !(min..=max).contains(&number) {
        return Err(std::io::Error::other(format!(
            "{name} must be in {min}..={max}"
        )));
    }
    Ok(number)
}
fn environment(name: &str, default: u64, min: u64, max: u64) -> std::io::Result<u64> {
    let value = match std::env::var(name) {
        Ok(v) => Some(v),
        Err(std::env::VarError::NotPresent) => None,
        Err(_) => return Err(std::io::Error::other(format!("{name} is not UTF-8"))),
    };
    bounded(name, value.as_deref(), default, min, max)
}
fn bind_address(value: Option<&str>) -> std::io::Result<IpAddr> {
    value
        .unwrap_or("127.0.0.1")
        .parse()
        .map_err(|_| std::io::Error::other("BIND_ADDR must be an IP address"))
}
#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let path = std::env::var_os("AUTARKY_SERVER_DB")
        .map(PathBuf::from)
        .unwrap_or_else(default_db_path);
    let port = environment("PORT", 8080, 1, 65535)? as u16;
    let bind_value = match std::env::var("BIND_ADDR") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(_) => return Err(std::io::Error::other("BIND_ADDR is not UTF-8")),
    };
    let bind = bind_address(bind_value.as_deref())?;
    let peer = environment("AUTARKY_RATE_MAX", 10, 1, 100_000)? as u32;
    let window = environment("AUTARKY_RATE_WINDOW", 60, 1, 86_400)?;
    let global = environment("AUTARKY_GLOBAL_RATE_MAX", 100, 1, 1_000_000)? as u32;
    let quota = environment("AUTARKY_MAX_SUBMISSIONS", 100_000, 1, 10_000_000)?;
    let max_mb = environment("AUTARKY_DB_MAX_MB", 256, 1, 16_384)?;
    let database =
        db::Executor::open(&path, max_mb * 1024 * 1024).map_err(std::io::Error::other)?;
    let admission_state = Admission::new(peer, global, window).map_err(std::io::Error::other)?;
    let state = web::Data::new(AppState {
        db: database,
        admission: Arc::new(admission_state),
        submission_quota: quota,
    });
    log::info!("Admission: {peer} requests per peer and {global} globally per {window} seconds; peer addresses come from the direct connection, not forwarding headers");
    log::info!("Listening on {bind}:{port}; use an HTTPS terminator for external traffic");
    HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .wrap(middleware::from_fn(admission))
            .configure(routes)
    })
    .workers(2)
    .max_connections(256)
    .client_request_timeout(Duration::from_secs(5))
    .client_disconnect_timeout(Duration::from_secs(2))
    .bind((bind, port))?
    .run()
    .await
}
