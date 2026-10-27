//! Database module
//! SQLite operations for storing benchmark results

use anyhow::{ensure, Context, Result};
use rusqlite::{params, Connection, Transaction, TransactionBehavior};
use std::path::Path;

use super::gguf::GgufInfo;
use super::hardware::HardwareInfo;
use super::monitor::{summarize_results, BenchmarkResult, MetricSummary};

/// Parse stored flags using the native option/value boundaries. Last occurrence wins.
/// Two-value options are represented as JSON arrays, not a lossy joined string.
pub fn parse_flags(flags: &[String]) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for arg in crate::arguments::walk(flags) {
        if !arg.known {
            continue;
        }
        let value = match arg.values.as_slice() {
            [] => serde_json::Value::String(String::new()),
            [value] => serde_json::Value::String((*value).to_owned()),
            values => serde_json::json!(values),
        };
        map.insert(arg.key, value);
    }
    map.into()
}

/// Flags whose entire entry (key + value) must be stripped before storage.
const REDACTED_FLAGS: &[&str] = &[
    "--api-key",
    "--api-key-file",
    "--ssl-key-file",
    "--ssl-cert-file",
    "--hf-token",
    "-hft",
    // Backend-local device IDs belong only in the private hardware snapshot.
    "--device",
    "-dev",
    "--spec-draft-device",
    "--device-draft",
    "-devd",
];

/// Flags that should be completely omitted (autarky-injected or noisy infrastructure).
const OMIT_FLAGS: &[&str] = &["--host", "--port", "--reuse-port"];

/// Flags whose values are file paths or URLs — only the last path segment is kept.
const PATH_VALUE_FLAGS: &[&str] = &[
    // Server-specific path/URL flags
    "--lookup-cache-static",
    "-lcs",
    "--lookup-cache-dynamic",
    "-lcd",
    "--mmproj",
    "-mm",
    "--mmproj-url",
    "-mmu",
    "--path",
    "--webui-config-file",
    "--slot-save-path",
    "--media-path",
    "--models-dir",
    "--models-preset",
    "--chat-template-file",
    "--model-vocoder",
    "-mv",
    "--model-draft",
    "--spec-draft-model",
    "-md",
    "--spec-draft-hf",
    "-hfd",
    "-hfrd",
    "--hf-repo-draft",
    // Common path flags that may appear in passthrough
    "--model",
    "-m",
    "--model-url",
    "-mu",
    "--lora",
    "--lora-scaled",
    "--control-vector",
    "--control-vector-scaled",
    "--hf-repo",
    "-hf",
    "-hfr",
    "--hf-file",
    "-hff",
    "--hf-repo-v",
    "-hfv",
    "-hfrv",
    "--hf-file-v",
    "-hffv",
    "--log-file",
    "--file",
    "-f",
    "--system-prompt-file",
    "-sysf",
    "--in-file",
    "--binary-file",
    "-bf",
    "--prompt-cache",
    "--grammar-file",
    "--json-schema-file",
    "-jf",
    "--save-all-logits",
    "--kl-divergence-base",
    "--image",
    "--audio",
    "--context-file",
    "--output",
    "--output-file",
    "-o",
    "--positive-file",
    "--negative-file",
    "--tts-speaker-file",
    "--logits-output-dir",
];

/// Match llama-server's long-option grammar without changing option values.
pub fn canonical_flag_name(key: &str) -> String {
    crate::arguments::canonical_name(key)
}

fn sanitize_path_value(key: &str, value: &str) -> String {
    let is_url = ["--model-url", "-mu", "--mmproj-url", "-mmu"].contains(&key);
    // Older records can contain a URL path with its query still attached.
    let value = if is_url {
        value.split(['?', '#']).next().unwrap_or(value)
    } else {
        value
    };
    // A URL query can contain commas. Remove it before splitting path lists.
    if let Ok(url) = reqwest::Url::parse(value) {
        if url.has_host() || url.scheme() == "file" {
            return format!("/{}", url.path().rsplit('/').next().unwrap_or_default());
        }
    }
    if is_url {
        return format!("/{}", value.rsplit(['/', '\\']).next().unwrap_or(value));
    }
    value
        .split(',')
        .map(|value| {
            let last = value.rsplit(['/', '\\']).next().unwrap_or(value);
            format!("/{}", last)
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// Sanitize a raw flat flags array for storage.
///
/// - **Redacted** keys + their values are stripped entirely.
/// - **Omitted** keys (--host/--port/--reuse-port) are stripped entirely.
/// - **Path** values are truncated to their last segment: `/foo/bar.gguf` → `/bar.gguf`
/// - Everything else passes through unchanged.
pub fn sanitize_flags(flags: &[String]) -> Vec<String> {
    let strict = crate::arguments::parse(flags);
    // Old records may predate an arity change (e.g. boolean flash-attn).
    // On invalid/unknown input, mark *every* explicit secret and its required
    // value before walking. A preceding legacy option cannot hide a secret.
    // This conservative fallback is for storage only, never native execution.
    let mut masked = vec![false; flags.len()];
    // Also interpret the one documented historical arity change, even when
    // a secret happens to look like a valid option and current syntax parses.
    // Both walks still respect values of unrelated options such as --alias.
    for arg in crate::arguments::legacy_storage_walk(flags) {
        if REDACTED_FLAGS.contains(&arg.key.as_str()) {
            for index in arg.range {
                masked[index] = true;
            }
        }
    }
    if strict.is_err() {
        for (index, token) in flags.iter().enumerate() {
            let (key, inline) = token
                .split_once('=')
                .map_or((token.as_str(), false), |(k, _)| (k, true));
            if REDACTED_FLAGS.contains(&canonical_flag_name(key).as_str()) {
                masked[index] = true;
                if !inline && index + 1 < flags.len() {
                    masked[index + 1] = true;
                }
            }
        }
    }
    let parsed = strict.unwrap_or_else(|_| crate::arguments::walk(flags));
    let mut result = Vec::new();
    for arg in parsed {
        if REDACTED_FLAGS.contains(&arg.key.as_str())
            || OMIT_FLAGS.contains(&arg.key.as_str())
            || arg.range.clone().any(|index| masked[index])
        {
            continue;
        }
        // Unknown values have no safe public interpretation. Do not upload them.
        if !arg.known {
            continue;
        }
        if !PATH_VALUE_FLAGS.contains(&arg.key.as_str()) {
            result.extend_from_slice(arg.tokens);
            continue;
        }
        let mut values = arg.values.iter().enumerate();
        if arg.inline {
            let key = arg.tokens[0].split_once('=').unwrap().0;
            if let Some((_, value)) = values.next() {
                result.push(format!("{}={}", key, sanitize_path_value(&arg.key, value)));
            }
        } else {
            result.push(arg.tokens[0].clone());
        }
        for (index, value) in values {
            result.push(if index == 0 {
                sanitize_path_value(&arg.key, value)
            } else {
                (*value).to_owned()
            });
        }
    }
    result
}

const SCHEMA_VERSION: i64 = 4;
const MAX_RECOMMENDATIONS: i64 = 1_000;

/// A local database with an explicit data directory. No global environment changes
/// are needed to use a separate database (including in tests).
pub struct Database {
    connection: Connection,
}

/// Run completion is separate from valid, already completed measurements.
pub enum RunOutcome<'a> {
    Completed,
    Failed(&'a str),
    Cancelled,
}

pub struct RunMeasurements<'a> {
    pub results: &'a [BenchmarkResult],
    pub outcome: RunOutcome<'a>,
}

impl Database {
    pub fn open(data_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(data_dir)?;
        let mut connection = Connection::open(data_dir.join("benchmarks.db"))?;
        connection.busy_timeout(std::time::Duration::from_secs(5))?;
        migrate(&mut connection)?;
        connection.pragma_update(None, "foreign_keys", true)?;
        Ok(Self { connection })
    }

    fn open_default() -> Result<Self> {
        let data_dir = dirs::data_dir()
            .context("Cannot find the local data directory")?
            .join("autarky");
        Self::open(&data_dir)
    }

    /// Store the captured model metadata, never reopen its former pathname.
    /// Every run owns an immutable model snapshot, even when paths are reused.
    pub fn store_benchmark(
        &mut self,
        hardware: &HardwareInfo,
        model_path: &Option<String>,
        model_info: Option<&GgufInfo>,
        results: &[BenchmarkResult],
        flags: &[String],
        llama_server_version: &str,
    ) -> Result<i64> {
        let outcome = if results.iter().any(|metric| metric.status == "cancelled") {
            RunOutcome::Cancelled
        } else if results.iter().any(|metric| metric.status != "success") {
            RunOutcome::Failed("One or more contexts failed")
        } else {
            RunOutcome::Completed
        };
        self.store_benchmark_run(
            hardware,
            model_path,
            model_info,
            &RunMeasurements { results, outcome },
            flags,
            llama_server_version,
        )
    }

    pub fn store_benchmark_run(
        &mut self,
        hardware: &HardwareInfo,
        model_path: &Option<String>,
        model_info: Option<&GgufInfo>,
        measurements: &RunMeasurements<'_>,
        flags: &[String],
        llama_server_version: &str,
    ) -> Result<i64> {
        let results = measurements.results;
        let (status, failure_reason) = match measurements.outcome {
            RunOutcome::Completed => ("completed", None),
            RunOutcome::Failed(reason) => ("failed", Some(reason)),
            RunOutcome::Cancelled => ("cancelled", Some("Benchmark cancelled")),
        };
        let devices_json = local_devices_json(hardware)?;
        let system_ram = i64::try_from(hardware.system_ram_gb)
            .context("System RAM exceeds SQLite integer range")?;
        validate_results(results)?;
        ensure!(
            !results.is_empty(),
            "Cannot store a run with no measurements"
        );
        ensure!(
            status != "completed" || results.iter().all(|metric| metric.status == "success"),
            "Completed runs require successful measurements"
        );
        let snapshot = model_info
            .map(|info| {
                if let Some(parameters) = info.parameters_billions {
                    ensure!(
                        parameters.is_finite() && parameters >= 0.0,
                        "Invalid model parameter count"
                    );
                }
                serde_json::to_string(info).context("Cannot serialize model snapshot")
            })
            .transpose()?;
        let sanitized = sanitize_flags(flags);
        let flags_json = serde_json::to_string(&sanitized)?;
        let parsed_json = serde_json::to_string(&parse_flags(&sanitized))?;
        let context_lengths: Vec<usize> =
            results.iter().map(|result| result.context_length).collect();
        let contexts_json = serde_json::to_string(&context_lengths)?;
        let metrics_json = serde_json::to_string(results)?;
        let oom_boundary = results
            .iter()
            .filter(|result| result.status == "oom")
            .map(|result| i64::try_from(result.context_length))
            .collect::<std::result::Result<Vec<_>, _>>()?
            .into_iter()
            .min();

        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let hardware_id: i64 = tx.query_row(
            "INSERT INTO hardware_fingerprints (gpu_family, vram_gb, system_ram_gb, os_platform, devices_json)
             VALUES (?1, ?2, ?3, ?4, ?5) RETURNING id",
            params![hardware.gpu_family, hardware.vram_gb, system_ram, hardware.os_platform, devices_json],
            |row| row.get(0),
        )?;
        let model_id: Option<i64> = model_info.map(|info| {
            tx.query_row(
                "INSERT INTO model_fingerprints (model_name, file_path, quantization, architecture_type,
                 expert_count, parameters_billions, architecture, block_count, context_length, snapshot_json, file_type)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11) RETURNING id",
                params![info.model_name, model_path, info.quantization,
                    info.expert_count.map(|count| if count > 0 { "moe" } else { "dense" }), info.expert_count,
                    info.parameters_billions, info.architecture, info.block_count, info.context_length, snapshot, info.file_type],
                |row| row.get(0),
            )
        }).transpose()?;
        let id: i64 = tx.query_row(
            "INSERT INTO benchmark_runs (hardware_id, model_id, flags_used, parsed_flags, context_lengths_tested,
             raw_metrics, oom_boundary, oom, benchmark_suite_version, llama_server_version, status, failure_reason)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12) RETURNING id",
            params![hardware_id, model_id, flags_json, parsed_json, contexts_json, metrics_json,
                oom_boundary, i64::from(oom_boundary.is_some()), env!("CARGO_PKG_VERSION"), llama_server_version, status, failure_reason],
            |row| row.get(0),
        )?;
        tx.commit()?;
        Ok(id)
    }

    /// Rank eligible successful measurements before applying the user limit.
    pub fn get_recommendations(
        &self,
        gpu_family: &str,
        min_tps: f64,
        limit: i64,
    ) -> Result<Vec<BenchmarkSummary>> {
        ensure!(
            min_tps.is_finite() && min_tps >= 0.0,
            "Minimum TPS must be finite and non-negative"
        );
        ensure!(
            (1..=MAX_RECOMMENDATIONS).contains(&limit),
            "Recommendation limit must be between 1 and {MAX_RECOMMENDATIONS}"
        );
        let mut statement = self.connection.prepare(
            "SELECT br.id, mf.model_name, hf.gpu_family, hf.vram_gb, br.flags_used, br.raw_metrics, br.oom_boundary
             FROM benchmark_runs br JOIN hardware_fingerprints hf ON br.hardware_id = hf.id
             LEFT JOIN model_fingerprints mf ON br.model_id = mf.id WHERE hf.gpu_family = ?1",
        )?;
        let mut rows = statement.query(params![gpu_family])?;
        let mut eligible = Vec::new();
        while let Some(row) = rows.next()? {
            let id: i64 = row.get(0)?;
            let raw_metrics = row
                .get::<_, Option<String>>(5)?
                .unwrap_or_else(|| "[]".to_string());
            let metrics: Vec<BenchmarkResult> = serde_json::from_str(&raw_metrics)
                .with_context(|| format!("Invalid metrics in benchmark {id}"))?;
            let summary = validate_results(&metrics)
                .with_context(|| format!("Invalid metrics in benchmark {id}"))?;
            if summary.success_count == 0 {
                continue;
            }
            if summary.max_tps < min_tps {
                continue;
            }
            let raw_flags = row
                .get::<_, Option<String>>(4)?
                .unwrap_or_else(|| "[]".to_string());
            let flags: Vec<String> = serde_json::from_str(&raw_flags)
                .with_context(|| format!("Invalid flags in benchmark {id}"))?;
            let flags = sanitize_flags(&flags);
            eligible.push(BenchmarkSummary {
                id,
                model_name: row.get(1)?,
                gpu_family: row.get(2)?,
                vram_gb: row.get(3)?,
                flags_used: serde_json::to_string(&flags)?,
                parsed_flags: serde_json::to_string(&parse_flags(&flags))?,
                raw_metrics,
                max_tps: summary.max_tps,
                avg_tps: summary.avg_tps,
                oom_boundary: row.get(6)?,
            });
        }
        eligible.sort_by(|left, right| {
            right
                .max_tps
                .total_cmp(&left.max_tps)
                .then_with(|| right.avg_tps.total_cmp(&left.avg_tps))
                .then_with(|| right.id.cmp(&left.id))
        });
        eligible.truncate(usize::try_from(limit)?);
        Ok(eligible)
    }

    /// Match the complete local identity, including device IDs. Legacy records
    /// without IDs match only an equally unknown (empty) device inventory.
    pub fn delete_all_for_hardware(&self, hardware: &HardwareInfo) -> Result<usize> {
        let devices_json = local_devices_json(hardware)?;
        let system_ram = i64::try_from(hardware.system_ram_gb)?;
        Ok(self.connection.execute(
            "DELETE FROM benchmark_runs WHERE hardware_id IN (
             SELECT id FROM hardware_fingerprints WHERE gpu_family = ?1 AND vram_gb = ?2
             AND system_ram_gb = ?3 AND os_platform = ?4 AND devices_json = ?5)",
            params![
                hardware.gpu_family,
                hardware.vram_gb,
                system_ram,
                hardware.os_platform,
                devices_json
            ],
        )?)
    }

    pub fn delete_last_entry(&self) -> Result<usize> {
        Ok(self.connection.execute(
            "DELETE FROM benchmark_runs WHERE id = (
             SELECT id FROM benchmark_runs ORDER BY timestamp DESC, id DESC LIMIT 1)",
            [],
        )?)
    }
}

fn local_devices_json(hardware: &HardwareInfo) -> Result<String> {
    ensure!(
        hardware.vram_gb.is_finite() && hardware.vram_gb >= 0.0,
        "Invalid hardware VRAM"
    );
    let mut devices: Vec<_> = hardware.devices.iter().collect();
    for device in &devices {
        ensure!(
            device.memory_gb.is_finite() && device.memory_gb >= 0.0,
            "Invalid device memory"
        );
    }
    devices.sort_by(|left, right| {
        left.id
            .cmp(&right.id)
            .then_with(|| left.name.cmp(&right.name))
            .then_with(|| left.memory_gb.total_cmp(&right.memory_gb))
    });
    Ok(serde_json::to_string(&devices)?)
}

fn validate_results(results: &[BenchmarkResult]) -> Result<MetricSummary> {
    let summary = summarize_results(results)?;
    for result in results {
        ensure!(
            i64::try_from(result.context_length).is_ok(),
            "Benchmark context exceeds SQLite integer range"
        );
    }
    Ok(summary)
}

fn has_column(tx: &Transaction<'_>, table: &str, column: &str) -> Result<bool> {
    // Identifiers below are internal constants, never user input.
    let mut statement = tx.prepare(&format!("PRAGMA table_info({table})"))?;
    let columns = statement.query_map([], |row| row.get::<_, String>(1))?;
    for name in columns {
        if name? == column {
            return Ok(true);
        }
    }
    Ok(false)
}

fn add_column(tx: &Transaction<'_>, table: &str, name: &str, definition: &str) -> Result<()> {
    if !has_column(tx, table, name)? {
        tx.execute_batch(&format!(
            "ALTER TABLE {table} ADD COLUMN {name} {definition}"
        ))?;
    }
    Ok(())
}

fn migrate(connection: &mut Connection) -> Result<()> {
    let version: i64 = connection.pragma_query_value(None, "user_version", |row| row.get(0))?;
    ensure!(
        (0..=SCHEMA_VERSION).contains(&version),
        "Unsupported local database schema version {version}"
    );
    if version == SCHEMA_VERSION {
        return Ok(());
    }
    // SQLite requires FK enforcement disabled before a transaction which rebuilds
    // a referenced table. Check every FK before commit, then enable enforcement.
    connection.pragma_update(None, "foreign_keys", false)?;
    let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    // Re-read after taking the lock: another process may have migrated meanwhile.
    let version: i64 = tx.pragma_query_value(None, "user_version", |row| row.get(0))?;
    ensure!(
        (0..=SCHEMA_VERSION).contains(&version),
        "Unsupported local database schema version {version}"
    );
    if version < 1 {
        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS hardware_fingerprints (
             id INTEGER PRIMARY KEY AUTOINCREMENT, gpu_family TEXT NOT NULL, vram_gb REAL NOT NULL,
             system_ram_gb INTEGER, os_platform TEXT);
             CREATE TABLE IF NOT EXISTS model_fingerprints (
             id INTEGER PRIMARY KEY AUTOINCREMENT, model_name TEXT NOT NULL, file_path TEXT,
             quantization TEXT, architecture_type TEXT, expert_count INTEGER DEFAULT 0, parameters_billions REAL);
             CREATE TABLE IF NOT EXISTS benchmark_runs (
             id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
             hardware_id INTEGER REFERENCES hardware_fingerprints(id), model_id INTEGER REFERENCES model_fingerprints(id),
             flags_used TEXT, context_lengths_tested TEXT, raw_metrics TEXT, oom_boundary INTEGER,
             oom INTEGER DEFAULT 0, benchmark_suite_version TEXT);
             CREATE INDEX IF NOT EXISTS idx_benchmarks_hardware ON benchmark_runs(hardware_id);
             CREATE INDEX IF NOT EXISTS idx_benchmarks_model ON benchmark_runs(model_id);
             CREATE INDEX IF NOT EXISTS idx_benchmarks_timestamp ON benchmark_runs(timestamp);"
        )?;
        add_column(&tx, "model_fingerprints", "file_path", "TEXT")?;
        add_column(&tx, "model_fingerprints", "gguf_hash", "TEXT")?;
        add_column(&tx, "benchmark_runs", "parsed_flags", "TEXT")?;
        add_column(&tx, "benchmark_runs", "llama_server_version", "TEXT")?;
        let legacy_flags = {
            let mut statement = tx.prepare("SELECT id, flags_used FROM benchmark_runs")?;
            let rows = statement.query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, Option<String>>(1)?))
            })?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };
        for (id, flags) in legacy_flags {
            let flags: Vec<String> = serde_json::from_str(flags.as_deref().unwrap_or("[]"))
                .with_context(|| format!("Cannot migrate flags for benchmark {id}"))?;
            let sanitized = sanitize_flags(&flags);
            tx.execute(
                "UPDATE benchmark_runs SET flags_used = ?1, parsed_flags = ?2 WHERE id = ?3",
                params![
                    serde_json::to_string(&sanitized)?,
                    serde_json::to_string(&parse_flags(&sanitized))?,
                    id
                ],
            )?;
        }
        tx.execute(
            "UPDATE benchmark_runs SET model_id = NULL WHERE model_id = 0
            AND NOT EXISTS (SELECT 1 FROM model_fingerprints WHERE id = benchmark_runs.model_id)",
            [],
        )?;
        tx.pragma_update(None, "user_version", 1)?;
    }
    // Capture before the first rebuild: copying retained rows alone loses the
    // AUTOINCREMENT high-water mark when higher model IDs have been deleted.
    // Restore after all rebuilds, in this same transaction, never lowering it.
    let model_sequence: Option<i64> = tx.query_row(
        "SELECT max(seq) FROM sqlite_sequence WHERE name = 'model_fingerprints'",
        [],
        |row| row.get(0),
    )?;
    if version < 2 {
        add_column(
            &tx,
            "hardware_fingerprints",
            "devices_json",
            "TEXT NOT NULL DEFAULT '[]'",
        )?;
        // Rebuild to remove both historical UNIQUE(path) and UNIQUE(hash).
        // A legacy hash is retained as a hash; it is never relabelled as a path.
        tx.execute_batch(
            "CREATE TABLE model_fingerprints_v2 (
             id INTEGER PRIMARY KEY AUTOINCREMENT, model_name TEXT NOT NULL, file_path TEXT, gguf_hash TEXT,
             quantization TEXT, architecture_type TEXT, expert_count INTEGER DEFAULT 0, parameters_billions REAL,
             architecture TEXT, block_count INTEGER, context_length INTEGER, snapshot_json TEXT);
             INSERT INTO model_fingerprints_v2 (id, model_name, file_path, gguf_hash, quantization,
             architecture_type, expert_count, parameters_billions)
             SELECT id, model_name, file_path, gguf_hash, quantization, architecture_type,
             expert_count, parameters_billions FROM model_fingerprints;
             DROP TABLE model_fingerprints;
             ALTER TABLE model_fingerprints_v2 RENAME TO model_fingerprints;
             CREATE INDEX idx_models_path ON model_fingerprints(file_path);"
        )?;
        tx.pragma_update(None, "user_version", 2)?;
    }
    if version < 3 {
        // The first two historical schemas had no `oom` column. Version 2
        // incorrectly marked those databases as migrated, so repair them in a
        // new transactionally versioned step rather than trusting that marker.
        if !has_column(&tx, "benchmark_runs", "oom")? {
            add_column(&tx, "benchmark_runs", "oom", "INTEGER DEFAULT 0")?;
            tx.execute(
                "UPDATE benchmark_runs SET oom = CASE WHEN oom_boundary IS NULL THEN 0 ELSE 1 END",
                [],
            )?;
        }
        add_column(&tx, "benchmark_runs", "parsed_flags", "TEXT")?;
        add_column(&tx, "benchmark_runs", "llama_server_version", "TEXT")?;
        add_column(
            &tx,
            "hardware_fingerprints",
            "devices_json",
            "TEXT NOT NULL DEFAULT '[]'",
        )?;
        for (name, definition) in [
            ("file_path", "TEXT"),
            ("gguf_hash", "TEXT"),
            ("architecture", "TEXT"),
            ("block_count", "INTEGER"),
            ("context_length", "INTEGER"),
            ("snapshot_json", "TEXT"),
        ] {
            add_column(&tx, "model_fingerprints", name, definition)?;
        }
        tx.pragma_update(None, "user_version", 3)?;
    }
    if version < 4 {
        let had_status = has_column(&tx, "benchmark_runs", "status")?;
        add_column(
            &tx,
            "benchmark_runs",
            "status",
            "TEXT NOT NULL DEFAULT 'unknown'",
        )?;
        add_column(&tx, "benchmark_runs", "failure_reason", "TEXT")?;
        if !had_status {
            // Legacy empty/corrupt measurements do not prove run completion.
            // CASE guards each JSON operation; a WHERE predicate does not
            // guarantee that SQLite evaluates a shape check before extraction.
            tx.execute_batch("UPDATE benchmark_runs SET status = CASE
                WHEN raw_metrics IS NULL OR NOT json_valid(raw_metrics) THEN 'unknown'
                WHEN json_type(raw_metrics) != 'array' OR json_array_length(raw_metrics) = 0 THEN 'unknown'
                WHEN EXISTS (SELECT 1 FROM json_each(raw_metrics) WHERE type != 'object') THEN 'unknown'
                WHEN EXISTS (SELECT 1 FROM json_each(raw_metrics) WHERE json_extract(value, '$.status') = 'cancelled') THEN 'cancelled'
                WHEN EXISTS (SELECT 1 FROM json_each(raw_metrics) WHERE json_extract(value, '$.status') IS NOT 'success') THEN 'failed'
                ELSE 'completed' END;")?;
        }
        // Presence is part of identity. Rebuild the NOT NULL name column rather
        // than inventing a second presence flag or a new placeholder string.
        // Classification of ambiguous old labels happens once, here only.
        add_column(&tx, "model_fingerprints", "file_type", "INTEGER")?;
        tx.execute_batch(
            "CREATE TABLE model_fingerprints_v4 (
             id INTEGER PRIMARY KEY AUTOINCREMENT, model_name TEXT, file_path TEXT, gguf_hash TEXT,
             quantization TEXT, architecture_type TEXT, expert_count INTEGER, parameters_billions REAL,
             architecture TEXT, block_count INTEGER, context_length INTEGER, snapshot_json TEXT, file_type INTEGER);
             INSERT INTO model_fingerprints_v4
             SELECT id,
             CASE WHEN lower(trim(model_name)) IN ('', 'unknown') THEN NULL ELSE model_name END,
             file_path, gguf_hash,
             CASE WHEN lower(trim(quantization)) IN ('', 'unknown', 'guessed') OR quantization GLOB 'FTYPE_*'
                  THEN NULL ELSE quantization END,
             architecture_type, expert_count, parameters_billions,
             CASE WHEN lower(trim(architecture)) IN ('', 'unknown') THEN NULL ELSE architecture END,
             block_count, context_length,
             coalesce(snapshot_json, json_object('model_name', model_name, 'quantization', quantization,
                 'architecture', architecture, 'architecture_type', architecture_type,
                 'expert_count', expert_count, 'parameters_billions', parameters_billions,
                 'block_count', block_count, 'context_length', context_length, 'file_type', file_type)),
             file_type FROM model_fingerprints;
             DROP TABLE model_fingerprints;
             ALTER TABLE model_fingerprints_v4 RENAME TO model_fingerprints;
             CREATE INDEX idx_models_path ON model_fingerprints(file_path);"
        )?;
        if let Some(sequence) = model_sequence {
            tx.execute(
                "INSERT INTO sqlite_sequence(name, seq) SELECT 'model_fingerprints', ?1
                WHERE NOT EXISTS (SELECT 1 FROM sqlite_sequence WHERE name = 'model_fingerprints')",
                [sequence],
            )?;
            tx.execute(
                "UPDATE sqlite_sequence SET seq = max(seq, ?1) WHERE name = 'model_fingerprints'",
                [sequence],
            )?;
        }
        tx.pragma_update(None, "user_version", 4)?;
    }
    {
        let mut check = tx.prepare("PRAGMA foreign_key_check")?;
        ensure!(
            check.query([])?.next()?.is_none(),
            "Local database has invalid foreign keys"
        );
    }
    tx.commit()?;
    Ok(())
}

pub fn init_db(data_dir: &Path) -> Result<()> {
    Database::open(data_dir).map(|_| ())
}

pub fn get_db_path() -> Result<String> {
    let data_dir = dirs::data_dir()
        .context("Cannot find the local data directory")?
        .join("autarky");
    Ok(data_dir
        .join("benchmarks.db")
        .to_string_lossy()
        .into_owned())
}

pub fn store_benchmark(
    hardware: &HardwareInfo,
    model_path: &Option<String>,
    model_info: Option<&GgufInfo>,
    results: &[BenchmarkResult],
    flags: &[String],
    llama_server_version: &str,
) -> Result<i64> {
    Database::open_default()?.store_benchmark(
        hardware,
        model_path,
        model_info,
        results,
        flags,
        llama_server_version,
    )
}

pub fn store_benchmark_run(
    hardware: &HardwareInfo,
    model_path: &Option<String>,
    model_info: Option<&GgufInfo>,
    measurements: &RunMeasurements<'_>,
    flags: &[String],
    llama_server_version: &str,
) -> Result<i64> {
    Database::open_default()?.store_benchmark_run(
        hardware,
        model_path,
        model_info,
        measurements,
        flags,
        llama_server_version,
    )
}

/// Summary of a benchmark run for recommendations
#[derive(Debug, Clone, serde::Serialize)]
pub struct BenchmarkSummary {
    pub id: i64,
    pub model_name: Option<String>,
    pub gpu_family: String,
    pub vram_gb: f64,
    pub flags_used: String,
    pub parsed_flags: String,
    pub raw_metrics: String,
    pub max_tps: f64,
    pub avg_tps: f64,
    pub oom_boundary: Option<i64>,
}

/// Get benchmark recommendations: setups with same GPU that achieved higher TPS
pub fn get_recommendations(
    gpu_family: &str,
    min_tps: f64,
    limit: i64,
) -> Result<Vec<BenchmarkSummary>> {
    Database::open_default()?.get_recommendations(gpu_family, min_tps, limit)
}

/// Delete all benchmark runs matching the given hardware fingerprint.
/// Returns the number of deleted rows.
pub fn delete_all_for_hardware(hardware: &HardwareInfo) -> Result<usize> {
    Database::open_default()?.delete_all_for_hardware(hardware)
}

/// Delete the most recent benchmark run. Returns the number of deleted rows (0 or 1).
pub fn delete_last_entry() -> Result<usize> {
    Database::open_default()?.delete_last_entry()
}
