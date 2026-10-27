use rusqlite::{params, types::Type, Connection, OptionalExtension, TransactionBehavior};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::mpsc::{self, SyncSender};
use std::time::{Duration, Instant};

pub const MAX_BODY: usize = 64 * 1024;
pub const MAX_RESPONSE: usize = 2 * 1024 * 1024;
pub const MAX_CONTEXT: u64 = 16_777_216;
pub const MAX_POINTS: usize = 128;
pub const MAX_RATE: f64 = 1_000_000_000.0;
pub const MAX_MEMORY: f64 = 1_048_576.0;
pub const MAX_LIMIT: i64 = 50;
pub const MAX_OFFSET: i64 = 10_000;
pub const DB_DEADLINE: Duration = Duration::from_secs(2);
const SCHEMA_VERSION: i64 = 3;

#[derive(Debug)]
pub enum Error {
    Sql(rusqlite::Error),
    Json(serde_json::Error),
    Io(std::io::Error),
    Invalid(&'static str),
    Quota,
    ResponseTooLarge,
    Unavailable,
    OutcomeUnknown,
}
impl From<rusqlite::Error> for Error {
    fn from(value: rusqlite::Error) -> Self {
        Self::Sql(value)
    }
}
impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}
impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sql(e) => write!(f, "database failure: {e}"),
            Self::Json(e) => write!(f, "stored JSON failure: {e}"),
            Self::Io(e) => write!(f, "I/O failure: {e}"),
            Self::Invalid(e) => f.write_str(e),
            Self::Quota => f.write_str("storage quota reached"),
            Self::ResponseTooLarge => f.write_str("response size limit reached"),
            Self::Unavailable => f.write_str("database unavailable"),
            Self::OutcomeUnknown => f.write_str("transaction outcome unknown"),
        }
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

/// Shared cancellation gate for the request, caller, and SQLite worker.
/// SQLite's commit hook runs after DELETE-journal lock acquisition, before
/// durable commit. After that gate, deadlines must await the real outcome:
/// an OS write/fsync cannot safely have a hard wall-clock abort deadline.
#[derive(Clone)]
pub struct Operation {
    control: std::sync::Arc<std::sync::Mutex<Control>>,
    #[cfg(test)]
    commit_pause: std::sync::Arc<std::sync::Mutex<Option<CommitPause>>>,
}
#[cfg(test)]
struct CommitPause {
    after_gate: bool,
    entered: tokio::sync::oneshot::Sender<()>,
    resume: mpsc::Receiver<()>,
}
#[cfg(test)]
impl CommitPause {
    fn wait(self) {
        let _ = self.entered.send(());
        self.resume.recv_timeout(Duration::from_secs(10)).unwrap();
    }
}
struct Control {
    deadline: Instant,
    phase: Phase,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Cancellable,
    Cancelled,
    Committing,
}
impl Operation {
    pub fn new(deadline: Instant) -> Self {
        Self {
            control: std::sync::Arc::new(std::sync::Mutex::new(Control {
                deadline,
                phase: Phase::Cancellable,
            })),
            #[cfg(test)]
            commit_pause: Default::default(),
        }
    }

    fn limit(&self, deadline: Instant) -> Instant {
        let mut control = self.control.lock().unwrap();
        control.deadline = control.deadline.min(deadline);
        control.deadline
    }

    /// True: cancellation won and no commit can pass the hook.
    /// False: commit won and only its terminal result is authoritative.
    pub fn cancel(&self) -> bool {
        let mut control = self.control.lock().unwrap();
        if control.phase == Phase::Committing {
            return false;
        }
        control.phase = Phase::Cancelled;
        true
    }

    fn lost_result(&self) -> Error {
        if self.cancel() {
            Error::Unavailable
        } else {
            Error::OutcomeUnknown
        }
    }

    fn interrupted(&self) -> bool {
        let mut control = self.control.lock().unwrap();
        if control.phase == Phase::Cancellable && Instant::now() >= control.deadline {
            control.phase = Phase::Cancelled;
        }
        control.phase == Phase::Cancelled
    }

    #[cfg(test)]
    pub fn pause_commit(
        &self,
        after_gate: bool,
    ) -> (tokio::sync::oneshot::Receiver<()>, mpsc::Sender<()>) {
        let (entered, receive) = tokio::sync::oneshot::channel();
        let (resume, wait) = mpsc::channel();
        *self.commit_pause.lock().unwrap() = Some(CommitPause {
            after_gate,
            entered,
            resume: wait,
        });
        (receive, resume)
    }

    fn begin_commit(&self) -> bool {
        #[cfg(test)]
        let pause = self.commit_pause.lock().unwrap().take();
        #[cfg(test)]
        let pause = if pause.as_ref().is_some_and(|p| !p.after_gate) {
            pause.unwrap().wait();
            None
        } else {
            pause
        };
        let mut control = self.control.lock().unwrap();
        if control.phase == Phase::Cancellable && Instant::now() >= control.deadline {
            control.phase = Phase::Cancelled;
        }
        if control.phase == Phase::Cancelled {
            return false;
        }
        control.phase = Phase::Committing;
        drop(control);
        #[cfg(test)]
        if let Some(pause) = pause {
            pause.wait();
        }
        true
    }
}

struct CancelOnDrop(Operation);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

type Work = Box<dyn FnOnce(&mut Connection, Operation, Instant) -> bool + Send>;
struct Job {
    deadline: Instant,
    operation: Operation,
    work: Work,
}

/// One database owner; HTTP workers never wait on a SQLite connection mutex.
/// Production write jobs contain one atomic transaction. A fully received POST
/// may commit after a transport disconnect; dropping the executor future is
/// explicit cancellation, not transport-level exactly-once delivery.
#[derive(Clone)]
pub struct Executor {
    sender: SyncSender<Job>,
}
impl Executor {
    pub fn open(path: &Path, max_bytes: u64) -> Result<Self> {
        let mut conn = open_database(path, max_bytes)?;
        migrate(&mut conn)?;
        let (sender, receiver) = mpsc::sync_channel::<Job>(16);
        std::thread::Builder::new()
            .name("autarky-db".into())
            .spawn(move || {
                while let Ok(job) = receiver.recv() {
                    if job.operation.interrupted() {
                        continue;
                    }
                    if !(job.work)(&mut conn, job.operation, job.deadline) {
                        break; // A failed rollback must retire the connection.
                    }
                }
            })?;
        Ok(Self { sender })
    }

    pub async fn call<T, F>(&self, operation: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut Connection, Instant) -> Result<T> + Send + 'static,
    {
        self.call_with(Operation::new(Instant::now() + DB_DEADLINE), operation)
            .await
    }

    pub async fn call_with<T, F>(&self, control: Operation, operation: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut Connection, Instant) -> Result<T> + Send + 'static,
    {
        let deadline = control.limit(Instant::now() + DB_DEADLINE);
        let _cancel = CancelOnDrop(control.clone());
        let (send, mut receive) = tokio::sync::oneshot::channel();
        let work = Box::new(
            move |conn: &mut Connection, control: Operation, deadline: Instant| {
                let progress = control.clone();
                conn.progress_handler(1000, Some(move || progress.interrupted()));
                let commit = control.clone();
                conn.commit_hook(Some(move || !commit.begin_commit()));
                let mut result = conn
                    .busy_timeout(
                        Duration::from_millis(200)
                            .min(deadline.saturating_duration_since(Instant::now())),
                    )
                    .map_err(Error::from)
                    .and_then(|()| {
                        if control.interrupted() {
                            Err(Error::Unavailable)
                        } else {
                            operation(conn, deadline)
                        }
                    });
                conn.progress_handler(0, None::<fn() -> bool>);
                conn.commit_hook(None::<fn() -> bool>);
                // Rollback must not be interrupted, or leak into the next job.
                if !conn.is_autocommit() {
                    result = conn
                        .execute_batch("ROLLBACK")
                        .map_err(Error::from)
                        .and(Err(Error::Unavailable));
                }
                if control.interrupted() {
                    result = Err(Error::Unavailable);
                }
                let reusable = conn.is_autocommit();
                let _ = send.send(result);
                reusable
            },
        );
        self.sender
            .try_send(Job {
                deadline,
                operation: control.clone(),
                work,
            })
            .map_err(|_| Error::Unavailable)?;
        match tokio::time::timeout_at(deadline.into(), &mut receive).await {
            Ok(result) => result.map_err(|_| control.lost_result())?,
            Err(_) if control.cancel() => Err(Error::Unavailable),
            // Commit won the gate. Never send a retryable timeout while the
            // write outcome is still unknown.
            Err(_) => receive.await.map_err(|_| control.lost_result())?,
        }
    }
}

fn open_database(path: &Path, max_bytes: u64) -> Result<Connection> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let conn = Connection::open(path)?;
    conn.busy_timeout(Duration::from_millis(200))?;
    conn.execute_batch("PRAGMA foreign_keys=ON; PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL; PRAGMA temp_store=MEMORY;")?;
    let page_size: u64 = conn.pragma_query_value(None, "page_size", |r| r.get(0))?;
    let pages: u64 = conn.pragma_query_value(None, "page_count", |r| r.get(0))?;
    if pages > max_bytes / page_size {
        return Err(Error::Quota);
    }
    conn.pragma_update(None, "max_page_count", max_bytes / page_size)?;
    Ok(conn)
}

fn has_column(conn: &Connection, table: &str, column: &str) -> Result<bool> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let columns = stmt
        .query_map([], |r| r.get::<_, String>(1))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(columns.iter().any(|c| c == column))
}

/// Upgrade legacy databases in one transaction, retaining IDs and all submissions.
fn migrate(conn: &mut Connection) -> Result<()> {
    let version: i64 = conn.pragma_query_value(None, "user_version", |r| r.get(0))?;
    if version > SCHEMA_VERSION {
        return Err(Error::Invalid("unsupported database schema"));
    }
    if version == SCHEMA_VERSION {
        return Ok(());
    }
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    tx.execute_batch("CREATE TABLE IF NOT EXISTS api_keys (
        id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE NOT NULL,
        label TEXT DEFAULT '', created_at TEXT DEFAULT (datetime('now')), last_used_at TEXT);
        CREATE TABLE IF NOT EXISTS hardware_fingerprints (
        id INTEGER PRIMARY KEY AUTOINCREMENT, gpu_family TEXT NOT NULL, vram_gb REAL NOT NULL,
        system_ram_gb INTEGER, os_platform TEXT);
        CREATE TABLE IF NOT EXISTS model_fingerprints (
        id INTEGER PRIMARY KEY AUTOINCREMENT, model_name TEXT NOT NULL, quantization TEXT,
        architecture_type TEXT, expert_count INTEGER DEFAULT 0, parameters_billions REAL);
        CREATE TABLE IF NOT EXISTS benchmark_submissions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, api_key_id INTEGER REFERENCES api_keys(id),
        hardware_id INTEGER REFERENCES hardware_fingerprints(id), model_id INTEGER REFERENCES model_fingerprints(id),
        flags_used TEXT, context_lengths_tested TEXT, raw_metrics TEXT, oom_boundary INTEGER,
        oom INTEGER DEFAULT 0, llama_server_version TEXT, submitted_at TEXT DEFAULT (datetime('now')));
        CREATE INDEX IF NOT EXISTS idx_submissions_hardware ON benchmark_submissions(hardware_id);
        CREATE INDEX IF NOT EXISTS idx_submissions_model ON benchmark_submissions(model_id);
        CREATE INDEX IF NOT EXISTS idx_submissions_submitted ON benchmark_submissions(submitted_at);")?;
    if !has_column(&tx, "benchmark_submissions", "llama_server_version")? {
        tx.execute_batch(
            "ALTER TABLE benchmark_submissions ADD COLUMN llama_server_version TEXT;",
        )?;
    }
    if !has_column(&tx, "hardware_fingerprints", "devices_json")? {
        tx.execute_batch("ALTER TABLE hardware_fingerprints ADD COLUMN devices_json TEXT;")?;
    }
    for table in ["hardware_fingerprints", "model_fingerprints"] {
        if !has_column(&tx, table, "identity")? {
            tx.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN identity TEXT;"))?;
        }
    }
    tx.execute_batch(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_hardware_identity ON hardware_fingerprints(identity);
        CREATE UNIQUE INDEX IF NOT EXISTS uq_model_identity ON model_fingerprints(identity);",
    )?;
    let mut after: Option<i64> = None;
    loop {
        let hardware = {
            let predicate = if after.is_some() { "id>?" } else { "? IS NULL" };
            let mut stmt = tx.prepare(&format!("SELECT id,gpu_family,vram_gb,system_ram_gb,os_platform,devices_json FROM hardware_fingerprints WHERE {predicate} ORDER BY id LIMIT 256"))?;
            let rows = stmt
                .query_map([after], |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        HardwarePayload {
                            gpu_family: r.get(1)?,
                            vram_gb: r.get(2)?,
                            system_ram_gb: r.get::<_, Option<u64>>(3)?.unwrap_or(0),
                            os_platform: r.get::<_, Option<String>>(4)?.unwrap_or_default(),
                            devices: json_optional(r.get::<_, Option<String>>(5)?, 5)?,
                        },
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            rows
        };
        if hardware.is_empty() {
            break;
        }
        for (id, hw) in hardware {
            after = Some(id);
            tx.execute("UPDATE hardware_fingerprints SET gpu_family=?,os_platform=CASE WHEN os_platform IS NULL THEN NULL ELSE ? END WHERE id=?",
            params![hw.gpu_family.trim(),normalized(&hw.os_platform),id])?;
            merge_identity(
                &tx,
                "hardware_fingerprints",
                "hardware_id",
                id,
                &hardware_identity(&hw)?,
            )?;
        }
    }
    let mut after: Option<i64> = None;
    loop {
        let models = {
            let predicate = if after.is_some() { "id>?" } else { "? IS NULL" };
            let mut stmt = tx.prepare(&format!("SELECT id,model_name,quantization,architecture_type,expert_count,parameters_billions FROM model_fingerprints WHERE {predicate} ORDER BY id LIMIT 256"))?;
            let rows = stmt
                .query_map([after], |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        ModelPayload {
                            model_name: r.get(1)?,
                            quantization: r.get(2)?,
                            architecture_type: r.get(3)?,
                            expert_count: r.get(4)?,
                            parameters_billions: r.get(5)?,
                        },
                    ))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            rows
        };
        if models.is_empty() {
            break;
        }
        for (id, model) in models {
            after = Some(id);
            tx.execute("UPDATE model_fingerprints SET model_name=?,quantization=?,architecture_type=? WHERE id=?",
            params![model.model_name.trim(),model.quantization.as_deref().map(str::trim),model.architecture_type.as_deref().map(str::trim),id])?;
            merge_identity(
                &tx,
                "model_fingerprints",
                "model_id",
                id,
                &model_identity(&model)?,
            )?;
        }
    }
    tx.execute_batch("CREATE UNIQUE INDEX IF NOT EXISTS uq_hardware_identity ON hardware_fingerprints(identity);
        CREATE UNIQUE INDEX IF NOT EXISTS uq_model_identity ON model_fingerprints(identity);
        CREATE TRIGGER IF NOT EXISTS hardware_identity_required BEFORE INSERT ON hardware_fingerprints
        WHEN NEW.identity IS NULL BEGIN SELECT RAISE(ABORT,'hardware identity required'); END;
        CREATE TRIGGER IF NOT EXISTS model_identity_required BEFORE INSERT ON model_fingerprints
        WHEN NEW.identity IS NULL BEGIN SELECT RAISE(ABORT,'model identity required'); END;")?;
    if tx
        .query_row("PRAGMA foreign_key_check", [], |_| Ok(()))
        .optional()?
        .is_some()
    {
        return Err(Error::Invalid("database contains invalid foreign keys"));
    }
    tx.pragma_update(None, "user_version", SCHEMA_VERSION)?;
    tx.commit()?;
    Ok(())
}

fn merge_identity(
    conn: &Connection,
    table: &str,
    foreign_key: &str,
    id: i64,
    identity: &str,
) -> Result<()> {
    let existing: Option<i64> = conn
        .query_row(
            &format!("SELECT id FROM {table} WHERE identity=? AND id!=? ORDER BY id LIMIT 1"),
            params![identity, id],
            |r| r.get(0),
        )
        .optional()?;
    if let Some(keep) = existing {
        conn.execute(
            &format!("UPDATE benchmark_submissions SET {foreign_key}=? WHERE {foreign_key}=?"),
            params![keep, id],
        )?;
        conn.execute(&format!("DELETE FROM {table} WHERE id=?"), [id])?;
    } else {
        conn.execute(
            &format!("UPDATE {table} SET identity=? WHERE id=?"),
            params![identity, id],
        )?;
    }
    Ok(())
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DevicePayload {
    pub name: String,
    pub memory_gb: f64,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HardwarePayload {
    pub gpu_family: String,
    pub vram_gb: f64,
    pub system_ram_gb: u64,
    pub os_platform: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub devices: Option<Vec<DevicePayload>>,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelPayload {
    pub model_name: String,
    pub quantization: Option<String>,
    pub architecture_type: Option<String>,
    pub expert_count: Option<i64>,
    pub parameters_billions: Option<f64>,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Success,
    Oom,
    Error,
    Cancelled,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MetricPoint {
    pub context_length: u64,
    pub tokens_per_second: f64,
    pub prompt_per_second: f64,
    pub status: Status,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SubmitPayload {
    pub hardware: HardwarePayload,
    pub model: Option<ModelPayload>,
    pub flags_used: Vec<String>,
    pub context_lengths_tested: Vec<u64>,
    pub raw_metrics: Vec<MetricPoint>,
    pub oom_boundary: Option<i64>,
    pub llama_server_version: Option<String>,
}
fn text(value: &str, max: usize) -> bool {
    !value.trim().is_empty() && value.len() <= max && !value.chars().any(char::is_control)
}
fn finite(value: f64, max: f64) -> bool {
    value.is_finite() && (0.0..=max).contains(&value)
}
impl SubmitPayload {
    /// Validate the entire typed payload before a key or fingerprint can be written.
    pub fn validate(&self) -> Result<()> {
        let h = &self.hardware;
        if !text(&h.gpu_family, 256)
            || !text(&h.os_platform, 64)
            || !finite(h.vram_gb, MAX_MEMORY)
            || h.system_ram_gb == 0
            || h.system_ram_gb > i64::MAX as u64
        {
            return Err(Error::Invalid("invalid hardware fields"));
        }
        if let Some(devices) = &h.devices {
            if devices.len() > 32
                || devices
                    .iter()
                    .any(|d| !text(&d.name, 256) || !finite(d.memory_gb, MAX_MEMORY))
            {
                return Err(Error::Invalid("invalid public devices"));
            }
        }
        if let Some(m) = &self.model {
            if !text(&m.model_name, 256)
                || m.quantization.as_ref().is_some_and(|s| !text(s, 64))
                || m.architecture_type.as_ref().is_some_and(|s| !text(s, 64))
                || m.expert_count
                    .is_some_and(|n| !(0..=1_000_000).contains(&n))
                || m.parameters_billions
                    .is_some_and(|v| !finite(v, 1_000_000.0))
            {
                return Err(Error::Invalid("invalid model fields"));
            }
        }
        if self.flags_used.is_empty()
            || self.flags_used.len() > 128
            || self.flags_used.iter().any(|s| !text(s, 512))
            || self
                .llama_server_version
                .as_ref()
                .is_some_and(|s| !text(s, 256))
        {
            return Err(Error::Invalid("invalid flags or server version"));
        }
        let contexts: BTreeSet<u64> = self.context_lengths_tested.iter().copied().collect();
        if contexts.is_empty()
            || contexts.len() > MAX_POINTS
            || contexts.len() != self.context_lengths_tested.len()
            || contexts.iter().any(|c| !(1..=MAX_CONTEXT).contains(c))
            || self.raw_metrics.len() != contexts.len()
        {
            return Err(Error::Invalid("invalid tested contexts"));
        }
        let mut seen = BTreeSet::new();
        for m in &self.raw_metrics {
            if !contexts.contains(&m.context_length)
                || !seen.insert(m.context_length)
                || !finite(m.tokens_per_second, MAX_RATE)
                || !finite(m.prompt_per_second, MAX_RATE)
                || m.error.as_ref().is_some_and(|e| e.len() > 1024)
                || (matches!(m.status, Status::Success) && m.error.is_some())
                || (!matches!(m.status, Status::Success)
                    && (m.tokens_per_second != 0.0 || m.prompt_per_second != 0.0))
            {
                return Err(Error::Invalid("invalid benchmark metrics"));
            }
        }
        if let Some(boundary) = self.oom_boundary {
            if boundary <= 0
                || boundary as u64 > MAX_CONTEXT
                || !self
                    .raw_metrics
                    .iter()
                    .any(|m| m.context_length == boundary as u64 && matches!(m.status, Status::Oom))
            {
                return Err(Error::Invalid("invalid OOM boundary"));
            }
        }
        Ok(())
    }
    pub fn remove_private_errors(&mut self) {
        for point in &mut self.raw_metrics {
            point.error = None;
        }
    }
}
fn normalized(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}
fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}
fn hardware_identity(h: &HardwarePayload) -> Result<String> {
    let mut devices: Vec<(String, f64)> = h
        .devices
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .map(|d| (normalized(&d.name), canonical_zero(d.memory_gb)))
        .collect();
    devices.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    Ok(serde_json::to_string(&(
        normalized(&h.gpu_family),
        canonical_zero(h.vram_gb),
        h.system_ram_gb,
        normalized(&h.os_platform),
        devices,
    ))?)
}
fn model_identity(m: &ModelPayload) -> Result<String> {
    Ok(serde_json::to_string(&(
        m.model_name.trim(),
        m.quantization.as_deref().map(str::trim),
        m.architecture_type.as_deref().map(str::trim),
        m.expert_count,
        m.parameters_billions.map(canonical_zero),
    ))?)
}

pub fn lookup_key(conn: &Connection, key: &str) -> Result<Option<i64>> {
    if key.is_empty() {
        return Ok(None);
    }
    Ok(conn
        .query_row("SELECT id FROM api_keys WHERE key=?", [key], |r| r.get(0))
        .optional()?)
}

pub fn store_submission(
    conn: &mut Connection,
    key: &str,
    payload: &SubmitPayload,
    quota: u64,
    deadline: Instant,
) -> Result<i64> {
    payload.validate()?;
    let mut public = payload.clone();
    public.remove_private_errors();
    let flags = serde_json::to_string(&public.flags_used)?;
    let contexts = serde_json::to_string(&public.context_lengths_tested)?;
    let metrics = serde_json::to_string(&public.raw_metrics)?;
    let hardware_identity = hardware_identity(&public.hardware)?;
    let model_identity = public.model.as_ref().map(model_identity).transpose()?;
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let count: u64 = tx.query_row("SELECT COUNT(*) FROM benchmark_submissions", [], |r| {
        r.get(0)
    })?;
    let keys: u64 = tx.query_row("SELECT COUNT(*) FROM api_keys", [], |r| r.get(0))?;
    if count >= quota || (keys >= quota && lookup_key(&tx, key)?.is_none()) {
        return Err(Error::Quota);
    }
    tx.execute("INSERT INTO api_keys(key,label,last_used_at) VALUES (?,'anonymous',datetime('now')) ON CONFLICT(key) DO UPDATE SET last_used_at=excluded.last_used_at", [key])?;
    let key_id = lookup_key(&tx, key)?.ok_or(Error::Invalid("missing inserted key"))?;
    let h = &public.hardware;
    tx.execute("INSERT INTO hardware_fingerprints(gpu_family,vram_gb,system_ram_gb,os_platform,devices_json,identity) VALUES (?,?,?,?,?,?) ON CONFLICT(identity) DO NOTHING",
        params![h.gpu_family.trim(), h.vram_gb, h.system_ram_gb, normalized(&h.os_platform), h.devices.as_ref().map(serde_json::to_string).transpose()?, hardware_identity])?;
    let hardware_id: i64 = tx.query_row(
        "SELECT id FROM hardware_fingerprints WHERE identity=?",
        [&hardware_identity],
        |r| r.get(0),
    )?;
    let model_id = if let (Some(m), Some(identity)) = (&public.model, model_identity) {
        tx.execute("INSERT INTO model_fingerprints(model_name,quantization,architecture_type,expert_count,parameters_billions,identity) VALUES (?,?,?,?,?,?) ON CONFLICT(identity) DO NOTHING",
            params![m.model_name.trim(),m.quantization.as_deref().map(str::trim),m.architecture_type.as_deref().map(str::trim),m.expert_count,m.parameters_billions,identity])?;
        Some(tx.query_row(
            "SELECT id FROM model_fingerprints WHERE identity=?",
            [identity],
            |r| r.get::<_, i64>(0),
        )?)
    } else {
        None
    };
    tx.execute("INSERT INTO benchmark_submissions(api_key_id,hardware_id,model_id,flags_used,context_lengths_tested,raw_metrics,oom_boundary,oom,llama_server_version) VALUES (?,?,?,?,?,?,?,?,?)",
        params![key_id,hardware_id,model_id,flags,contexts,metrics,public.oom_boundary,public.raw_metrics.iter().any(|m| matches!(m.status,Status::Oom)),public.llama_server_version])?;
    let id = tx.last_insert_rowid();
    if Instant::now() >= deadline {
        return Err(Error::Unavailable);
    }
    tx.commit()?;
    Ok(id)
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryParams {
    pub model: Option<String>,
    pub gpu: Option<String>,
    pub min_vram: Option<f64>,
    pub os: Option<String>,
    pub flags: Option<String>,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}
impl QueryParams {
    pub fn validate(&self) -> Result<()> {
        if !(1..=MAX_LIMIT).contains(&self.limit.unwrap_or(20))
            || !(0..=MAX_OFFSET).contains(&self.offset.unwrap_or(0))
            || [&self.model, &self.gpu, &self.os, &self.flags]
                .iter()
                .any(|s| s.as_ref().is_some_and(|v| !text(v, 512)))
            || self.min_vram.is_some_and(|v| !finite(v, MAX_MEMORY))
        {
            return Err(Error::Invalid("invalid query filters or pagination"));
        }
        Ok(())
    }
}
#[derive(Debug, Serialize)]
pub struct SubmissionRow {
    pub id: i64,
    pub gpu_family: String,
    pub vram_gb: f64,
    pub system_ram_gb: u64,
    pub os_platform: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub devices: Option<Vec<DevicePayload>>,
    pub model_name: Option<String>,
    pub quantization: Option<String>,
    pub architecture_type: Option<String>,
    pub parameters_billions: Option<f64>,
    pub expert_count: Option<i64>,
    pub flags_used: Vec<String>,
    pub context_lengths_tested: Vec<u64>,
    pub oom_boundary: Option<i64>,
    pub raw_metrics: Vec<MetricPoint>,
    pub submitted_at: String,
}
const SELECT: &str = "SELECT bs.id,hf.gpu_family,hf.vram_gb,hf.system_ram_gb,hf.os_platform,
    mf.model_name,mf.quantization,mf.architecture_type,mf.parameters_billions,
    bs.flags_used,bs.context_lengths_tested,bs.oom_boundary,bs.raw_metrics,bs.submitted_at,hf.devices_json,mf.expert_count
    FROM benchmark_submissions bs JOIN hardware_fingerprints hf ON bs.hardware_id=hf.id
    LEFT JOIN model_fingerprints mf ON bs.model_id=mf.id WHERE 1=1";
fn literal_like(value: &str) -> String {
    format!(
        "%{}%",
        value
            .replace('\\', "\\\\")
            .replace('%', "\\%")
            .replace('_', "\\_")
    )
}
pub fn query_submissions(
    conn: &Connection,
    q: &QueryParams,
    exclude: Option<i64>,
    exact_model: Option<&str>,
    hardware: Option<i64>,
) -> Result<Vec<SubmissionRow>> {
    q.validate()?;
    let mut sql = SELECT.to_string();
    let mut values: Vec<rusqlite::types::Value> = vec![];
    for (column, value) in [("mf.model_name", &q.model), ("hf.gpu_family", &q.gpu)] {
        if let Some(value) = value {
            sql.push_str(&format!(" AND {column} LIKE ? ESCAPE '\\'"));
            values.push(literal_like(value).into());
        }
    }
    if let Some(flags) = &q.flags {
        sql.push_str(" AND (SELECT group_concat(value,' ') FROM (SELECT value FROM json_each(bs.flags_used) ORDER BY CAST(key AS INTEGER))) LIKE ? ESCAPE '\\'");
        values.push(literal_like(flags).into());
    }
    if let Some(vram) = q.min_vram {
        sql.push_str(" AND hf.vram_gb>=?");
        values.push(vram.into());
    }
    if let Some(os) = &q.os {
        sql.push_str(" AND hf.os_platform=?");
        values.push(normalized(os).into());
    }
    if let Some(id) = exclude {
        sql.push_str(" AND (bs.api_key_id IS NULL OR bs.api_key_id!=?)");
        values.push(id.into());
    }
    if let Some(model) = exact_model {
        sql.push_str(" AND mf.model_name=? COLLATE BINARY");
        values.push(model.to_string().into());
    }
    if let Some(id) = hardware {
        sql.push_str(" AND bs.hardware_id=?");
        values.push(id.into());
    }
    sql.push_str(" ORDER BY bs.id DESC LIMIT ? OFFSET ?");
    values.push(q.limit.unwrap_or(20).into());
    values.push(q.offset.unwrap_or(0).into());
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt
        .query_map(rusqlite::params_from_iter(values), map_row)?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows)
}
fn json<T: DeserializeOwned>(source: &str, index: usize) -> rusqlite::Result<T> {
    serde_json::from_str(source)
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(index, Type::Text, Box::new(e)))
}
fn json_optional<T: DeserializeOwned>(
    source: Option<String>,
    index: usize,
) -> rusqlite::Result<Option<T>> {
    source.map(|s| json(&s, index)).transpose()
}
fn map_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<SubmissionRow> {
    for index in [1, 4, 5, 6, 7, 9, 10, 12, 13, 14] {
        if let rusqlite::types::ValueRef::Text(value) = row.get_ref(index)? {
            if value.len() > MAX_BODY {
                return Err(rusqlite::Error::InvalidQuery);
            }
        }
    }
    let mut metrics: Vec<MetricPoint> = json(&row.get::<_, String>(12)?, 12)?;
    // Historic rows may contain process stderr. It is never part of the public API.
    for m in &mut metrics {
        m.error = None;
    }
    Ok(SubmissionRow {
        id: row.get(0)?,
        gpu_family: row.get(1)?,
        vram_gb: row.get(2)?,
        system_ram_gb: row.get(3)?,
        os_platform: row.get(4)?,
        model_name: row.get(5)?,
        quantization: row.get(6)?,
        architecture_type: row.get(7)?,
        parameters_billions: row.get(8)?,
        expert_count: row.get(15)?,
        flags_used: json(&row.get::<_, String>(9)?, 9)?,
        context_lengths_tested: json(&row.get::<_, String>(10)?, 10)?,
        oom_boundary: row.get(11)?,
        raw_metrics: metrics,
        submitted_at: row.get(13)?,
        devices: json_optional(row.get(14)?, 14)?,
    })
}
fn fetch(conn: &Connection, id: i64) -> Result<Option<SubmissionRow>> {
    Ok(conn
        .query_row(&format!("{SELECT} AND bs.id=?"), [id], map_row)
        .optional()?)
}
#[derive(Debug, Serialize)]
pub struct MetricComparison {
    pub state: &'static str,
    pub percent: Option<f64>,
    pub baseline_average: Option<f64>,
    pub candidate_average: Option<f64>,
}
impl MetricComparison {
    fn unavailable(state: &'static str) -> Self {
        Self {
            state,
            percent: None,
            baseline_average: None,
            candidate_average: None,
        }
    }
}
#[derive(Debug, Serialize)]
pub struct Comparison {
    pub benchmark_1: SubmissionRow,
    pub benchmark_2: SubmissionRow,
    pub matched_contexts: Vec<u64>,
    pub tps: MetricComparison,
    pub prompt: MetricComparison,
}
fn compare_metric(
    a: &[&MetricPoint],
    b: &[&MetricPoint],
    metric: fn(&MetricPoint) -> f64,
) -> MetricComparison {
    if a.is_empty() {
        return MetricComparison::unavailable("no_matched_successful_contexts");
    }
    if a.iter()
        .chain(b.iter())
        .any(|m| !finite(metric(m), MAX_RATE))
    {
        return MetricComparison::unavailable("invalid_metrics");
    }
    // Validated rate and point bounds keep the sum finite. Sum first so positive
    // subnormal rates do not underflow to a fabricated zero baseline.
    let a_avg = a.iter().map(|m| metric(m)).sum::<f64>() / a.len() as f64;
    let b_avg = b.iter().map(|m| metric(m)).sum::<f64>() / b.len() as f64;
    let mut result = MetricComparison {
        state: "zero_baseline",
        percent: None,
        baseline_average: Some(b_avg),
        candidate_average: Some(a_avg),
    };
    if b_avg == 0.0 {
        return result;
    }
    let delta = ((a_avg - b_avg) / b_avg) * 100.0;
    if delta.is_finite() {
        result.state = "available";
        result.percent = Some(delta);
    } else {
        result.state = "non_finite_delta";
    }
    result
}
pub fn compare_pair(conn: &Connection, id1: i64, id2: i64) -> Result<Option<Comparison>> {
    let (Some(a), Some(b)) = (fetch(conn, id1)?, fetch(conn, id2)?) else {
        return Ok(None);
    };
    let successful = |metrics: &[MetricPoint]| -> BTreeMap<u64, usize> {
        metrics
            .iter()
            .enumerate()
            .filter(|(_, m)| matches!(m.status, Status::Success))
            .map(|(i, m)| (m.context_length, i))
            .collect()
    };
    let a_success = successful(&a.raw_metrics);
    let b_success = successful(&b.raw_metrics);
    let matched: Vec<u64> = a_success
        .keys()
        .filter(|c| b_success.contains_key(c))
        .copied()
        .collect();
    let av: Vec<&MetricPoint> = matched
        .iter()
        .map(|c| &a.raw_metrics[a_success[c]])
        .collect();
    let bv: Vec<&MetricPoint> = matched
        .iter()
        .map(|c| &b.raw_metrics[b_success[c]])
        .collect();
    // Presence establishes knowledge; no model name or third-party quantization
    // label is a sentinel. Hardware may differ for the same known workload.
    let architecture_agrees = a
        .architecture_type
        .as_deref()
        .zip(b.architecture_type.as_deref())
        .is_none_or(|(left, right)| left.trim() == right.trim());
    let experts_agree = a
        .expert_count
        .zip(b.expert_count)
        .is_none_or(|(left, right)| left == right);
    let parameters_agree = a
        .parameters_billions
        .zip(b.parameters_billions)
        .is_none_or(|(left, right)| left == right);
    let identity_state = match (
        a.model_name.as_deref(),
        a.quantization.as_deref(),
        b.model_name.as_deref(),
        b.quantization.as_deref(),
    ) {
        (Some(an), Some(aq), Some(bn), Some(bq))
            if [an, aq, bn, bq].iter().all(|s| !s.trim().is_empty()) =>
        {
            if an.trim() == bn.trim()
                && aq.trim() == bq.trim()
                && architecture_agrees
                && experts_agree
                && parameters_agree
            {
                None
            } else {
                Some("incompatible_model")
            }
        }
        _ => Some("unknown_model_identity"),
    };
    let (tps, prompt) = if let Some(state) = identity_state {
        (
            MetricComparison::unavailable(state),
            MetricComparison::unavailable(state),
        )
    } else {
        (
            compare_metric(&av, &bv, |m| m.tokens_per_second),
            compare_metric(&av, &bv, |m| m.prompt_per_second),
        )
    };
    Ok(Some(Comparison {
        benchmark_1: a,
        benchmark_2: b,
        matched_contexts: matched,
        tps,
        prompt,
    }))
}

pub fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    // Serialize through a capped writer so legacy oversized rows cannot allocate an unbounded response.
    struct Capped(Vec<u8>);
    impl std::io::Write for Capped {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if bytes.len() > MAX_RESPONSE.saturating_sub(self.0.len()) {
                return Err(std::io::Error::other("response limit"));
            }
            self.0.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut writer = Capped(Vec::new());
    serde_json::to_writer(&mut writer, value).map_err(|error| {
        if error.is_io() {
            Error::ResponseTooLarge
        } else {
            Error::Json(error)
        }
    })?;
    Ok(writer.0)
}
