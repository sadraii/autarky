use super::*;
use actix_web::{http::StatusCode, test};
use rusqlite::{params, Connection};
use serde_json::{json, Value};
use std::path::Path;

struct Fixture {
    _dir: tempfile::TempDir,
    path: PathBuf,
    state: web::Data<AppState>,
}
impl Fixture {
    fn new() -> Self {
        Self::configured(10_000, 100_000, 10_000)
    }
    fn configured(peer: u32, global: u32, quota: u64) -> Self {
        let scratch =
            std::env::var_os("TMPDIR").expect("TMPDIR must name the test scratch directory");
        let dir = tempfile::Builder::new()
            .prefix("autarky-server-test-")
            .tempdir_in(scratch)
            .unwrap();
        let path = dir.path().join("test.db");
        let db = db::Executor::open(&path, 16 * 1024 * 1024).unwrap();
        let state = web::Data::new(AppState {
            db,
            admission: Arc::new(Admission::new(peer, global, 60).unwrap()),
            submission_quota: quota,
        });
        Self {
            _dir: dir,
            path,
            state,
        }
    }
    async fn counts(&self) -> Vec<i64> {
        self.state
            .db
            .call(|conn, _| {
                [
                    "api_keys",
                    "hardware_fingerprints",
                    "model_fingerprints",
                    "benchmark_submissions",
                ]
                .into_iter()
                .map(|table| {
                    Ok(conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))?)
                })
                .collect()
            })
            .await
            .unwrap()
    }
}
macro_rules! app {
    ($fixture:expr) => {
        test::init_service(
            App::new()
                .app_data($fixture.state.clone())
                .wrap(middleware::from_fn(admission))
                .configure(routes),
        )
        .await
    };
}
macro_rules! post {
    ($app:expr,$key:expr,$payload:expr) => {{
        let response = test::call_service(
            &$app,
            test::TestRequest::post()
                .uri("/api/v1/benchmark")
                .insert_header(("X-API-Key", $key))
                .set_json($payload)
                .to_request(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body: Value = test::read_body_json(response).await;
        body["id"].as_i64().unwrap()
    }};
}
macro_rules! get {
    ($app:expr,$uri:expr) => {{
        let response =
            test::call_service(&$app, test::TestRequest::get().uri($uri).to_request()).await;
        assert_eq!(response.status(), StatusCode::OK);
        test::read_body_json::<Value, _>(response).await
    }};
}
macro_rules! comparison {
    ($app:expr,$a:expr,$b:expr)=>{{
        let response=test::call_service(&$app,test::TestRequest::post().uri("/api/v1/compare").set_json(json!({"id1":$a,"id2":$b})).to_request()).await;
        assert_eq!(response.status(),StatusCode::OK);
        test::read_body_json::<Value,_>(response).await
    }};
}
fn payload() -> Value {
    json!({"hardware":{"gpu_family":"CPU","vram_gb":0.0,"system_ram_gb":32,"os_platform":"linux"},
        "model":{"model_name":"model","quantization":"Q4","architecture_type":"dense","expert_count":0,"parameters_billions":7.0},
        "flags_used":["--model","model.gguf","--threads","8"],"context_lengths_tested":[1024,2048],
        "raw_metrics":[{"context_length":1024,"tokens_per_second":10.0,"prompt_per_second":20.0,"status":"success"},
            {"context_length":2048,"tokens_per_second":5.0,"prompt_per_second":10.0,"status":"success"}],
        "oom_boundary":null,"llama_server_version":"test"})
}

#[actix_web::test]
async fn matched_positive_subnormal_rates_do_not_become_zero_baselines() {
    let fixture = Fixture::new();
    let app = app!(fixture);
    let smallest = f64::from_bits(1);
    let mut body = payload();
    for point in body["raw_metrics"].as_array_mut().unwrap() {
        point["tokens_per_second"] = json!(smallest);
        point["prompt_per_second"] = json!(smallest);
    }
    let baseline = post!(app, "subnormal-baseline", &body);
    let candidate = post!(app, "subnormal-candidate", &body);
    let comparison = comparison!(app, baseline, candidate);
    for metric in ["tps", "prompt"] {
        assert_eq!(comparison[metric]["state"], "available");
        assert_eq!(comparison[metric]["percent"].as_f64(), Some(0.0));
        assert_eq!(
            comparison[metric]["baseline_average"].as_f64(),
            Some(smallest)
        );
        assert_eq!(
            comparison[metric]["candidate_average"].as_f64(),
            Some(smallest)
        );
    }
}

#[actix_web::test]
async fn comparisons_reject_contradictory_known_model_metadata() {
    let fixture = Fixture::new();
    let app = app!(fixture);
    let mut size_candidate = payload();
    size_candidate["model"]["parameters_billions"] = json!(13.0);
    let mut architecture_base = payload();
    architecture_base["model"]["expert_count"] = Value::Null;
    let mut architecture_candidate = architecture_base.clone();
    architecture_candidate["model"]["architecture_type"] = json!("moe");
    let mut expert_base = payload();
    expert_base["model"]["architecture_type"] = json!("moe");
    expert_base["model"]["expert_count"] = json!(4);
    let mut expert_candidate = expert_base.clone();
    expert_candidate["model"]["expert_count"] = json!(8);
    for (baseline, candidate) in [
        (payload(), size_candidate),
        (architecture_base, architecture_candidate),
        (expert_base, expert_candidate),
    ] {
        let baseline_id = post!(app, "metadata-baseline", &baseline);
        let candidate_id = post!(app, "metadata-candidate", &candidate);
        let comparison = comparison!(app, baseline_id, candidate_id);
        for metric in ["tps", "prompt"] {
            assert_eq!(comparison[metric]["state"], "incompatible_model");
            assert!(comparison[metric]["percent"].is_null());
        }
        assert_eq!(
            comparison["benchmark_1"]["raw_metrics"],
            baseline["raw_metrics"]
        );
        assert_eq!(
            comparison["benchmark_2"]["raw_metrics"],
            candidate["raw_metrics"]
        );
    }
}

#[actix_web::test]
async fn missing_optional_model_metadata_is_not_an_invented_contradiction() {
    let fixture = Fixture::new();
    let app = app!(fixture);
    let baseline = post!(app, "known-metadata", payload());
    for field in ["architecture_type", "expert_count", "parameters_billions"] {
        let mut candidate = payload();
        candidate["model"].as_object_mut().unwrap().remove(field);
        let candidate = post!(app, "partial-metadata", &candidate);
        let comparison = comparison!(app, baseline, candidate);
        assert_eq!(comparison["tps"]["state"], "available");
        assert!(comparison["benchmark_2"][field].is_null());
    }
}

#[actix_web::test]
async fn all_metric_points_and_database_ranges_are_validated_before_writes() {
    let fixture = Fixture::new();
    let app = app!(fixture);
    let mut cases = vec![];
    macro_rules! invalid {
        ($pointer:expr,$value:expr) => {{
            let mut p = payload();
            *p.pointer_mut($pointer).unwrap() = $value;
            cases.push(p);
        }};
    }
    invalid!("/hardware/system_ram_gb", json!(u64::MAX));
    invalid!("/hardware/system_ram_gb", json!(0));
    invalid!("/hardware/vram_gb", json!(-1));
    invalid!("/hardware/vram_gb", json!(1e300));
    invalid!("/hardware/gpu_family", json!(" "));
    invalid!("/hardware/os_platform", json!("x".repeat(65)));
    invalid!("/model/expert_count", json!(-1));
    invalid!("/model/parameters_billions", json!(1e300));
    invalid!("/flags_used", json!([]));
    invalid!("/flags_used", json!(vec!["x"; 129]));
    invalid!("/flags_used/0", json!("x".repeat(513)));
    invalid!("/context_lengths_tested", json!([1024, 1024]));
    invalid!("/context_lengths_tested/1", json!(u64::MAX));
    invalid!("/context_lengths_tested/1", json!(0));
    invalid!("/raw_metrics/1", json!({"status":"success"}));
    invalid!("/raw_metrics/1/status", json!("bogus"));
    invalid!("/raw_metrics/1/context_length", json!(1024));
    invalid!("/raw_metrics/1/tokens_per_second", json!(1e300));
    invalid!("/raw_metrics/1/prompt_per_second", json!(-0.1));
    invalid!("/raw_metrics/1/status", json!("oom"));
    invalid!("/oom_boundary", json!(2048));
    invalid!("/oom_boundary", json!(i64::MAX));
    let mut private = payload();
    private["hardware"]["devices"] = json!([{"name":"GPU","memory_gb":8,"device_id":"private"}]);
    cases.push(private);
    for (i, body) in cases.iter().enumerate() {
        let response = test::call_service(
            &app,
            test::TestRequest::post()
                .uri("/api/v1/benchmark")
                .insert_header(("X-API-Key", format!("key-{i}")))
                .set_json(body)
                .to_request(),
        )
        .await;
        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "invalid case {i}"
        );
    }
    assert_eq!(fixture.counts().await, vec![0, 0, 0, 0]);
}

#[actix_web::test]
async fn cpu_zero_vram_public_devices_and_all_statuses_round_trip_without_private_errors() {
    let fixture = Fixture::new();
    let app = app!(fixture);
    let mut body = payload();
    body["hardware"]["devices"] = json!([{"name":"Unified GPU","memory_gb":0.0}]);
    body["context_lengths_tested"] = json!([1024, 2048, 4096, 8192]);
    body["raw_metrics"] = json!([
        {"context_length":1024,"tokens_per_second":10,"prompt_per_second":20,"status":"success"},
        {"context_length":2048,"tokens_per_second":0,"prompt_per_second":0,"status":"oom","error":"secret /home/user token=123"},
        {"context_length":4096,"tokens_per_second":0,"prompt_per_second":0,"status":"error","error":"private process stderr"},
        {"context_length":8192,"tokens_per_second":0,"prompt_per_second":0,"status":"cancelled"}]);
    body["oom_boundary"] = json!(2048);
    let id = post!(app, "anonymous-key", &body);
    assert!(id > 0);
    let result = get!(app, "/api/v1/benchmarks");
    assert_eq!(result["count"], 1);
    assert_eq!(
        result["results"][0]["raw_metrics"]
            .as_array()
            .unwrap()
            .len(),
        4
    );
    assert_eq!(result["results"][0]["devices"], body["hardware"]["devices"]);
    assert!(!result.to_string().contains("secret"));
    assert!(!result.to_string().contains("stderr"));
    fixture
        .state
        .db
        .call(|conn, _| {
            let metrics: String =
                conn.query_row("SELECT raw_metrics FROM benchmark_submissions", [], |r| {
                    r.get(0)
                })?;
            assert!(!metrics.contains("secret"));
            assert!(!metrics.contains("stderr"));
            Ok(())
        })
        .await
        .unwrap();
}

#[actix_web::test]
async fn pagination_rejects_negative_zero_and_excessive_values_on_all_list_routes() {
    let fixture = Fixture::new();
    let app = app!(fixture);
    post!(app, "key", &payload());
    for route in [
        "/api/v1/benchmarks",
        "/api/v1/models/model",
        "/api/v1/hardware/1/benchmarks",
    ] {
        for query in [
            "limit=-1",
            "limit=0",
            "limit=51",
            "limit=9223372036854775807",
            "offset=-1",
            "offset=10001",
            "min_vram=NaN",
            "min_vram=inf",
        ] {
            let response = test::call_service(
                &app,
                test::TestRequest::get()
                    .uri(&format!("{route}?{query}"))
                    .to_request(),
            )
            .await;
            assert_eq!(
                response.status(),
                StatusCode::BAD_REQUEST,
                "{route}?{query}"
            );
        }
        let result = get!(app, &format!("{route}?limit=1&offset=0"));
        assert_eq!(result["count"], 1);
    }
    for _ in 0..22 {
        post!(app, "key", &payload());
    }
    assert_eq!(get!(app, "/api/v1/benchmarks")["count"], 20);
    assert_eq!(
        get!(app, "/api/v1/benchmarks?limit=1&offset=22")["count"],
        1
    );
}

#[actix_web::test]
async fn exclusion_reads_work_on_a_query_only_connection_and_never_register_keys() {
    let fixture = Fixture::new();
    let app = app!(fixture);
    post!(app, "known", &payload());
    fixture
        .state
        .db
        .call(|conn, _| {
            conn.execute("UPDATE api_keys SET last_used_at='unchanged'", [])?;
            conn.pragma_update(None, "query_only", true)?;
            Ok(())
        })
        .await
        .unwrap();
    for (key, count) in [("unknown", 1), ("", 1), ("known", 0)] {
        let response = test::call_service(
            &app,
            test::TestRequest::get()
                .uri("/api/v1/benchmarks")
                .insert_header(("X-API-Key", key))
                .to_request(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body: Value = test::read_body_json(response).await;
        assert_eq!(body["count"], count);
    }
    assert_eq!(fixture.counts().await, vec![1, 1, 1, 1]);
    fixture
        .state
        .db
        .call(|conn, _| {
            let timestamp: String =
                conn.query_row("SELECT last_used_at FROM api_keys", [], |r| r.get(0))?;
            assert_eq!(timestamp, "unchanged");
            Ok(())
        })
        .await
        .unwrap();
}

#[actix_web::test]
async fn injected_failures_roll_back_keys_hardware_models_and_key_timestamps() {
    for table in [
        "hardware_fingerprints",
        "model_fingerprints",
        "benchmark_submissions",
    ] {
        let fixture = Fixture::new();
        let app = app!(fixture);
        fixture.state.db.call(move |conn,_| {
            conn.execute_batch(&format!("CREATE TRIGGER fail_write BEFORE INSERT ON {table} BEGIN SELECT RAISE(ABORT,'private-secret SQL detail'); END;"))?;Ok(())
        }).await.unwrap();
        let response = test::call_service(
            &app,
            test::TestRequest::post()
                .uri("/api/v1/benchmark")
                .insert_header(("X-API-Key", "new-key"))
                .set_json(payload())
                .to_request(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = test::read_body(response).await;
        assert!(!String::from_utf8_lossy(&body).contains("private-secret"));
        assert_eq!(fixture.counts().await, vec![0, 0, 0, 0]);
    }
    let fixture = Fixture::new();
    let app = app!(fixture);
    post!(app, "existing", &payload());
    fixture.state.db.call(|conn,_| {
        conn.execute_batch("UPDATE api_keys SET last_used_at='unchanged'; CREATE TRIGGER fail_write BEFORE INSERT ON benchmark_submissions BEGIN SELECT RAISE(ABORT,'failure'); END;")?;Ok(())
    }).await.unwrap();
    let mut body = payload();
    body["hardware"]["gpu_family"] = json!("different");
    body["model"]["model_name"] = json!("other");
    let response = test::call_service(
        &app,
        test::TestRequest::post()
            .uri("/api/v1/benchmark")
            .insert_header(("X-API-Key", "existing"))
            .set_json(&body)
            .to_request(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(fixture.counts().await, vec![1, 1, 1, 1]);
    fixture
        .state
        .db
        .call(|conn, _| {
            assert_eq!(
                conn.query_row("SELECT last_used_at FROM api_keys", [], |r| r
                    .get::<_, String>(0))?,
                "unchanged"
            );
            Ok(())
        })
        .await
        .unwrap();
}

#[actix_web::test]
async fn canonical_hardware_and_models_have_database_enforced_unique_identity() {
    let fixture = Fixture::new();
    let app = app!(fixture);
    let mut first = payload();
    first["hardware"]["devices"] =
        json!([{"name":"GPU B","memory_gb":8},{"name":"GPU A","memory_gb":16}]);
    post!(app, "one", &first);
    let mut second = first.clone();
    second["hardware"]["gpu_family"] = json!(" cpu ");
    second["hardware"]["os_platform"] = json!("LINUX");
    second["hardware"]["devices"] =
        json!([{"name":"gpu a","memory_gb":16},{"name":"gpu b","memory_gb":8}]);
    second["model"]["model_name"] = json!(" model ");
    post!(app, "two", &second);
    assert_eq!(fixture.counts().await, vec![2, 1, 1, 2]);
    assert_eq!(get!(app, "/api/v1/hardware/1/benchmarks")["count"], 2);
    fixture.state.db.call(|conn,_| {
        assert!(conn.execute("INSERT INTO hardware_fingerprints(gpu_family,vram_gb,system_ram_gb,os_platform,identity) SELECT gpu_family,vram_gb,system_ram_gb,os_platform,identity FROM hardware_fingerprints",[]).is_err());
        assert!(conn.execute("INSERT INTO model_fingerprints(model_name,identity) SELECT model_name,identity FROM model_fingerprints",[]).is_err());Ok(())
    }).await.unwrap();
}

#[actix_web::test]
async fn exact_model_and_literal_substring_and_flags_filters_are_distinct() {
    let fixture = Fixture::new();
    let app = app!(fixture);
    for (name, flag, gpu, os) in [
        ("model", "--flash-attn", "GPU A", "linux"),
        ("model-extra", "--threads", "GPU B", "windows"),
        ("model_%", "--literal_%", "GPU A", "linux"),
    ] {
        let mut body = payload();
        body["model"]["model_name"] = json!(name);
        body["flags_used"] = json!([flag, "on"]);
        body["hardware"]["gpu_family"] = json!(gpu);
        body["hardware"]["vram_gb"] = json!(8);
        body["hardware"]["os_platform"] = json!(os);
        post!(app, "key", &body);
    }
    assert_eq!(get!(app, "/api/v1/models/model")["count"], 1);
    assert_eq!(get!(app, "/api/v1/benchmarks?model=model")["count"], 3);
    assert_eq!(get!(app, "/api/v1/benchmarks?model=%25")["count"], 1);
    assert_eq!(get!(app, "/api/v1/benchmarks?model=_")["count"], 1);
    assert_eq!(
        get!(app, "/api/v1/benchmarks?flags=--flash-attn")["count"],
        1
    );
    assert_eq!(get!(app, "/api/v1/benchmarks?flags=%25")["count"], 1);
    assert_eq!(
        get!(app, "/api/v1/benchmarks?flags=not-present")["count"],
        0
    );
    assert_eq!(
        get!(app, "/api/v1/benchmarks?flags=--flash-attn%20on")["count"],
        1
    );
    assert_eq!(
        get!(app, "/api/v1/benchmarks?gpu=GPU%20A&min_vram=8&os=linux")["count"],
        2
    );
}

#[actix_web::test]
async fn comparisons_match_successful_contexts_and_explicitly_report_unavailable_deltas() {
    let fixture = Fixture::new();
    let app = app!(fixture);
    let a = post!(app, "key", &payload());
    let mut body = payload();
    body["raw_metrics"][0]["tokens_per_second"] = json!(5);
    body["raw_metrics"][1]["tokens_per_second"] = json!(0);
    body["raw_metrics"][1]["prompt_per_second"] = json!(0);
    body["raw_metrics"][1]["status"] = json!("oom");
    let b = post!(app, "key", &body);
    let result = comparison!(app, a, b);
    assert_eq!(result["matched_contexts"], json!([1024]));
    assert_eq!(result["tps"]["percent"], 100.0);
    assert_eq!(result["delta"]["tps_percent"], "+100.0%");
    body["raw_metrics"][0]["tokens_per_second"] = json!(0);
    let zero = post!(app, "key", &body);
    let result = comparison!(app, a, zero);
    assert_eq!(result["tps"]["state"], "zero_baseline");
    assert!(result["tps"]["percent"].is_null());
    assert!(result["delta"].get("tps_percent").is_none());
    body["context_lengths_tested"] = json!([4096]);
    body["raw_metrics"] = json!([{"context_length":4096,"tokens_per_second":10,"prompt_per_second":20,"status":"success"}]);
    let disjoint = post!(app, "key", &body);
    let result = comparison!(app, a, disjoint);
    assert_eq!(result["tps"]["state"], "no_matched_successful_contexts");
    assert!(result["tps"]["percent"].is_null());
    body["context_lengths_tested"] = json!([1024]);
    body["raw_metrics"][0]["context_length"] = json!(1024);
    body["raw_metrics"][0]["tokens_per_second"] = json!(5e-324);
    let tiny = post!(app, "key", &body);
    let result = comparison!(app, a, tiny);
    assert_eq!(result["tps"]["state"], "non_finite_delta");
    assert!(result["tps"]["percent"].is_null());
    let response = test::call_service(
        &app,
        test::TestRequest::post()
            .uri("/api/v1/compare")
            .set_json(json!({"id1":a,"id2":99999}))
            .to_request(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[actix_web::test]
async fn corrupt_rows_and_lookup_database_errors_are_opaque_failures_not_partial_success() {
    let fixture = Fixture::new();
    let app = app!(fixture);
    let id = post!(app, "key", &payload());
    let other = post!(app, "key", &payload());
    fixture
        .state
        .db
        .call(move |conn, _| {
            conn.execute(
                "UPDATE benchmark_submissions SET raw_metrics='private malformed data' WHERE id=?",
                [id],
            )?;
            Ok(())
        })
        .await
        .unwrap();
    for uri in [
        "/api/v1/benchmarks",
        "/api/v1/models/model",
        "/api/v1/hardware/1/benchmarks",
    ] {
        let response =
            test::call_service(&app, test::TestRequest::get().uri(uri).to_request()).await;
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert!(!String::from_utf8_lossy(&test::read_body(response).await).contains("private"));
    }
    let response = test::call_service(
        &app,
        test::TestRequest::post()
            .uri("/api/v1/compare")
            .set_json(json!({"id1":id,"id2":other}))
            .to_request(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    fixture
        .state
        .db
        .call(|conn, _| {
            conn.execute_batch("ALTER TABLE api_keys RENAME TO broken_keys;")?;
            Ok(())
        })
        .await
        .unwrap();
    let response = test::call_service(
        &app,
        test::TestRequest::get()
            .uri("/api/v1/benchmarks")
            .insert_header(("X-API-Key", "unknown"))
            .to_request(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
}

#[actix_web::test]
async fn rotated_keys_and_forwarded_headers_do_not_bypass_peer_or_global_admission_before_json() {
    let fixture = Fixture::configured(2, 100, 100);
    let app = app!(fixture);
    for i in 0..2 {
        let response = test::call_service(
            &app,
            test::TestRequest::post()
                .uri("/api/v1/benchmark")
                .peer_addr("127.0.0.1:1234".parse().unwrap())
                .insert_header(("X-API-Key", format!("key-{i}")))
                .insert_header(("X-Forwarded-For", format!("192.0.2.{i}")))
                .set_json(payload())
                .to_request(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
    }
    let response = test::call_service(
        &app,
        test::TestRequest::post()
            .uri("/api/v1/benchmark")
            .peer_addr("127.0.0.1:1234".parse().unwrap())
            .insert_header(("X-API-Key", "rotated"))
            .insert_header(("Content-Type", "application/json"))
            .set_payload("not JSON private-secret")
            .to_request(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(response.headers().contains_key("Retry-After"));
    let body: Value = test::read_body_json(response).await;
    assert!(body["error"]
        .as_str()
        .unwrap()
        .contains("2 requests per 60 seconds per peer"));
    assert_eq!(fixture.counts().await, vec![2, 1, 1, 2]);
    assert_eq!(get!(app, "/health")["status"], "ok");
    let fixture = Fixture::configured(10, 1, 100);
    let app = app!(fixture);
    post!(app, "one", &payload());
    let response = test::call_service(
        &app,
        test::TestRequest::get()
            .uri("/api/v1/benchmarks")
            .peer_addr("192.0.2.2:1234".parse().unwrap())
            .to_request(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
}

#[actix_web::test]
async fn storage_quota_and_concurrent_admission_are_bounded_without_blocking_health() {
    let fixture = Fixture::configured(100, 100, 1);
    let app = app!(fixture);
    post!(app, "one", &payload());
    let response = test::call_service(
        &app,
        test::TestRequest::post()
            .uri("/api/v1/benchmark")
            .insert_header(("X-API-Key", "two"))
            .set_json(payload())
            .to_request(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::INSUFFICIENT_STORAGE);
    assert_eq!(fixture.counts().await, vec![1, 1, 1, 1]);
    let permits = fixture
        .state
        .admission
        .in_flight
        .clone()
        .acquire_many_owned(16)
        .await
        .unwrap();
    let response = test::call_service(
        &app,
        test::TestRequest::get()
            .uri("/api/v1/benchmarks")
            .to_request(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(get!(app, "/health")["status"], "ok");
    drop(permits);
}

#[actix_web::test]
async fn database_contention_does_not_block_health_and_busy_is_retryable() {
    let fixture = Fixture::new();
    let app = app!(fixture);
    let lock = Connection::open(&fixture.path).unwrap();
    lock.execute_batch("BEGIN EXCLUSIVE;").unwrap();
    let write = test::call_service(
        &app,
        test::TestRequest::post()
            .uri("/api/v1/benchmark")
            .insert_header(("X-API-Key", "key"))
            .set_json(payload())
            .to_request(),
    );
    let liveness = async {
        tokio::time::sleep(Duration::from_millis(20)).await;
        let start = Instant::now();
        let response =
            test::call_service(&app, test::TestRequest::get().uri("/health").to_request()).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(start.elapsed() < Duration::from_millis(100));
    };
    let (response, ()) = tokio::join!(write, liveness);
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(response.headers().contains_key("Retry-After"));
    lock.execute_batch("ROLLBACK;").unwrap();
    assert_eq!(fixture.counts().await, vec![0, 0, 0, 0]);
    assert_eq!(get!(app, "/ready")["status"], "ready");
}

#[actix_web::test]
async fn limiter_configuration_bounds_and_state_capacity_are_safe() {
    assert_eq!(bind_address(None).unwrap(), IpAddr::V4(Ipv4Addr::LOCALHOST));
    assert!(bind_address(Some("not-a-host")).is_err());
    assert_eq!(
        bind_address(Some("::1")).unwrap(),
        IpAddr::V6(std::net::Ipv6Addr::LOCALHOST)
    );
    for (max, global, window) in [
        (0, 1, 60),
        (1, 0, 60),
        (1, 1, 0),
        (u32::MAX, 1, 60),
        (1, u32::MAX, 60),
        (1, 1, u64::MAX),
    ] {
        assert!(RateLimiter::new(max, global, window).is_err());
    }
    for value in ["0", "18446744073709551615", "-1", "bogus"] {
        assert!(bounded("AUTARKY_RATE_WINDOW", Some(value), 60, 1, 86400).is_err());
    }
    let mut limiter = RateLimiter::new(1, 100_000, 60).unwrap();
    for n in 0..1024u32 {
        limiter
            .check(IpAddr::V4(Ipv4Addr::from(n)))
            .unwrap_or_else(|_| panic!("peer {n} should fit"));
    }
    assert!(limiter.check(IpAddr::V4(Ipv4Addr::from(1024u32))).is_err());
    assert_eq!(limiter.peers.len(), 1024);
    for entry in limiter.peers.values_mut() {
        entry.started = Instant::now() - Duration::from_secs(61);
    }
    assert!(limiter.check(IpAddr::V4(Ipv4Addr::from(2048u32))).is_ok());
    assert_eq!(limiter.peers.len(), 1);
    let fixture = Fixture::new();
    let app = app!(fixture);
    let admission = fixture.state.admission.clone();
    let _ = std::thread::spawn(move || {
        let _guard = admission.limiter.lock().unwrap();
        panic!("test poison");
    })
    .join();
    let response = test::call_service(
        &app,
        test::TestRequest::get()
            .uri("/api/v1/benchmarks")
            .to_request(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(get!(app, "/health")["status"], "ok");
}

#[actix_web::test]
async fn malformed_and_oversized_requests_and_keys_do_not_echo_secrets_or_write() {
    let fixture = Fixture::new();
    let app = app!(fixture);
    for (key, body, status) in [
        (
            "key",
            "private-secret malformed".to_string(),
            StatusCode::BAD_REQUEST,
        ),
        (
            "key",
            "x".repeat(db::MAX_BODY + 1),
            StatusCode::PAYLOAD_TOO_LARGE,
        ),
        ("", payload().to_string(), StatusCode::UNAUTHORIZED),
        (
            &"x".repeat(129),
            payload().to_string(),
            StatusCode::BAD_REQUEST,
        ),
    ] {
        let response = test::call_service(
            &app,
            test::TestRequest::post()
                .uri("/api/v1/benchmark")
                .insert_header(("X-API-Key", key))
                .insert_header(("Content-Type", "application/json"))
                .set_payload(body)
                .to_request(),
        )
        .await;
        assert_eq!(response.status(), status);
        assert!(
            !String::from_utf8_lossy(&test::read_body(response).await).contains("private-secret")
        );
    }
    assert_eq!(fixture.counts().await, vec![0, 0, 0, 0]);
}

mod cancellation_tests;
mod comparison_tests;
mod load_tests;

#[actix_web::test]
async fn unknown_model_numbers_remain_null_and_distinct_from_known_zero() {
    let fixture = Fixture::new();
    let app = app!(fixture);
    let mut unknown = payload();
    unknown["model"]["expert_count"] = Value::Null;
    unknown["model"]["parameters_billions"] = Value::Null;
    let unknown_id = post!(app, "unknown", &unknown);
    let mut zero = unknown.clone();
    zero["model"]["expert_count"] = json!(0);
    zero["model"]["parameters_billions"] = json!(0.0);
    let zero_id = post!(app, "zero", &zero);
    assert_eq!(fixture.counts().await, vec![2, 1, 2, 2]);
    let rows = get!(app, "/api/v1/benchmarks");
    assert_eq!(rows["results"][0]["id"], zero_id);
    assert_eq!(rows["results"][0]["expert_count"], 0);
    assert_eq!(rows["results"][0]["parameters_billions"], 0.0);
    assert_eq!(rows["results"][1]["id"], unknown_id);
    assert!(rows["results"][1]["expert_count"].is_null());
    assert!(rows["results"][1]["parameters_billions"].is_null());
    fixture.state.db.call(|conn, _| {
        let stored: (Option<i64>, Option<f64>) = conn.query_row(
            "SELECT expert_count,parameters_billions FROM model_fingerprints WHERE expert_count IS NULL", [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        assert_eq!(stored, (None, None));
        let known: (Option<i64>, Option<f64>) = conn.query_row(
            "SELECT expert_count,parameters_billions FROM model_fingerprints WHERE expert_count=0", [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        assert_eq!(known, (Some(0), Some(0.0)));
        Ok(())
    }).await.unwrap();
    let mut expert_only = unknown.clone();
    expert_only["model"]["expert_count"] = json!(0);
    post!(app, "expert-zero", &expert_only);
    let mut parameters_only = unknown;
    parameters_only["model"]["parameters_billions"] = json!(0.0);
    post!(app, "parameters-zero", &parameters_only);
    // Each optional number is part of identity independently. Neither NULL
    // may collapse into a known zero, even when the other number is unchanged.
    assert_eq!(fixture.counts().await, vec![4, 1, 4, 4]);
    let rows = get!(app, "/api/v1/benchmarks");
    assert!(rows["results"][0]["expert_count"].is_null());
    assert_eq!(rows["results"][0]["parameters_billions"], 0.0);
    assert_eq!(rows["results"][1]["expert_count"], 0);
    assert!(rows["results"][1]["parameters_billions"].is_null());
}
mod migration_tests;

fn create_legacy(path: &Path) {
    let conn = Connection::open(path).unwrap();
    conn.execute_batch(include_str!("../tests/fixtures/initial-schema.sql"))
        .unwrap();
    conn.execute_batch("INSERT INTO api_keys(id,key) VALUES(1,'legacy');
        INSERT INTO hardware_fingerprints VALUES(1,'CPU',0,32,'linux'),(2,' cpu ',0,32,'LINUX');
        INSERT INTO model_fingerprints VALUES(1,'model','Q4','dense',0,7),(2,'model','Q4','dense',0,7);") .unwrap();
    let p = payload();
    for id in [1, 2] {
        conn.execute("INSERT INTO benchmark_submissions(id,api_key_id,hardware_id,model_id,flags_used,context_lengths_tested,raw_metrics) VALUES(?,1,?,?,?,?,?)",
            params![id,id,id,p["flags_used"].to_string(),p["context_lengths_tested"].to_string(),p["raw_metrics"].to_string()]).unwrap();
    }
}
#[actix_web::test]
async fn legacy_migration_merges_duplicates_preserves_submissions_and_is_idempotent() {
    let fixture = Fixture::new();
    let legacy = fixture._dir.path().join("legacy.db");
    create_legacy(&legacy);
    let db = db::Executor::open(&legacy, 16 * 1024 * 1024).unwrap();
    let fixture = Fixture {
        _dir: fixture._dir,
        path: legacy.clone(),
        state: web::Data::new(AppState {
            db,
            admission: Arc::new(Admission::new(100, 1000, 60).unwrap()),
            submission_quota: 100,
        }),
    };
    let app = app!(fixture);
    assert_eq!(fixture.counts().await, vec![1, 1, 1, 2]);
    assert_eq!(get!(app, "/api/v1/hardware/1/benchmarks")["count"], 2);
    assert_eq!(get!(app, "/api/v1/models/model")["count"], 2);
    post!(app, "new", &payload());
    assert_eq!(fixture.counts().await, vec![2, 1, 1, 3]);
    let reopened = db::Executor::open(&legacy, 16 * 1024 * 1024).unwrap();
    reopened
        .call(|conn, _| {
            assert_eq!(
                conn.pragma_query_value(None, "user_version", |r| r.get::<_, i64>(0))?,
                3
            );
            assert_eq!(
                conn.query_row("SELECT COUNT(*) FROM benchmark_submissions", [], |r| r
                    .get::<_, i64>(0))?,
                3
            );
            Ok(())
        })
        .await
        .unwrap();
}
