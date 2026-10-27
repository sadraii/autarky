//! Persistence tests use explicit temporary databases, never the user's data.

use autarky::db::{init_db, parse_flags, sanitize_flags, Database};
use autarky::gguf::{parse_gguf, GgufInfo};
use autarky::hardware::{HardwareDevice, HardwareInfo};
use autarky::monitor::BenchmarkResult;
use rusqlite::{params, Connection};
use tempfile::tempdir;

#[path = "cli/storage_identity.rs"]
mod storage_identity;

fn hardware() -> HardwareInfo {
    HardwareInfo {
        gpu_family: "Test GPU".to_string(),
        vram_gb: 16.0,
        system_ram_gb: 64,
        os_platform: "linux".to_string(),
        devices: vec![HardwareDevice {
            id: "local-device-A".to_string(),
            name: "Test GPU".to_string(),
            memory_gb: 16.0,
        }],
    }
}

fn model(name: &str) -> GgufInfo {
    GgufInfo {
        model_name: Some(name.to_string()),
        architecture: Some("llama".to_string()),
        quantization: Some("Q4_K_M".to_string()),
        file_type: Some(15),
        is_moe: false,
        expert_count: Some(0),
        block_count: Some(32),
        parameters_billions: Some(7.0),
        context_length: Some(8192),
    }
}

fn save(database: &mut Database, hw: &HardwareInfo, speed: f64) -> i64 {
    database
        .store_benchmark(
            hw,
            &None,
            Some(&model("test_model")),
            &[BenchmarkResult::new(1024, speed, 100.0)],
            &[],
            "test-server",
        )
        .unwrap()
}

fn count(connection: &Connection, table: &str) -> i64 {
    connection
        .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
            row.get(0)
        })
        .unwrap()
}

fn legacy_schema(connection: &Connection, identity: &str, with_parsed_flags: bool) {
    connection.execute_batch(&format!(
        "PRAGMA foreign_keys = OFF;
         CREATE TABLE hardware_fingerprints (id INTEGER PRIMARY KEY AUTOINCREMENT, gpu_family TEXT NOT NULL,
         vram_gb REAL NOT NULL, cpu_arch TEXT, cpu_cores INTEGER, system_ram_gb INTEGER, os_platform TEXT);
         CREATE TABLE model_fingerprints (id INTEGER PRIMARY KEY AUTOINCREMENT, model_name TEXT NOT NULL,
         {identity} TEXT UNIQUE, quantization TEXT, architecture_type TEXT, expert_count INTEGER DEFAULT 0,
         parameters_billions REAL);
         CREATE TABLE benchmark_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
         hardware_id INTEGER REFERENCES hardware_fingerprints(id), model_id INTEGER REFERENCES model_fingerprints(id),
         flags_used TEXT, context_lengths_tested TEXT, raw_metrics TEXT, oom_boundary INTEGER,
         oom INTEGER DEFAULT 0, benchmark_suite_version TEXT);
         INSERT INTO hardware_fingerprints (id, gpu_family, vram_gb, system_ram_gb, os_platform)
         VALUES (9, 'Test GPU', 16, 64, 'linux');
         INSERT INTO model_fingerprints (id, model_name, {identity}, quantization, architecture_type)
         VALUES (7, 'legacy_model', 'legacy-identity', 'Q4_0', 'dense');"
    )).unwrap();
    if with_parsed_flags {
        connection
            .execute_batch(
                "ALTER TABLE benchmark_runs ADD COLUMN parsed_flags TEXT;
            ALTER TABLE benchmark_runs ADD COLUMN llama_server_version TEXT;",
            )
            .unwrap();
    }
    let flags = serde_json::json!([
        "--api-key=-private-secret",
        "--model=/private/model.gguf",
        "--threads",
        "8"
    ])
    .to_string();
    let raw = serde_json::to_string(&[BenchmarkResult::new(1024, 30.0, 100.0)]).unwrap();
    connection
        .execute(
            "INSERT INTO benchmark_runs (id, hardware_id, model_id, flags_used, raw_metrics)
        VALUES (11, 9, 7, ?1, ?2)",
            params![flags, raw],
        )
        .unwrap();
    connection
        .execute(
            "INSERT INTO benchmark_runs (id, hardware_id, model_id) VALUES (12, 9, 0)",
            [],
        )
        .unwrap();
}

#[derive(serde::Deserialize)]
struct HistoricalSchema {
    commit: String,
    ddl: Vec<String>,
}

fn historical_schemas() -> Vec<HistoricalSchema> {
    // Exact CREATE/INDEX/ALTER statements extracted with git show <commit>:src/db.rs.
    // Keep every known source revision, even when two revisions share their DDL.
    let schemas: Vec<HistoricalSchema> =
        serde_json::from_str(include_str!("fixtures/historical-schemas.json")).unwrap();
    assert_eq!(schemas.len(), 10);
    let commits: std::collections::BTreeSet<_> =
        schemas.iter().map(|s| s.commit.as_str()).collect();
    assert_eq!(commits.len(), 10);
    schemas
}

fn column_exists(connection: &Connection, table: &str, column: &str) -> bool {
    connection
        .prepare(&format!("PRAGMA table_info({table})"))
        .unwrap()
        .query_map([], |row| row.get::<_, String>(1))
        .unwrap()
        .map(Result::unwrap)
        .any(|name| name == column)
}

fn historical_database(connection: &Connection, schema: &HistoricalSchema) {
    connection
        .pragma_update(None, "foreign_keys", false)
        .unwrap();
    for ddl in &schema.ddl {
        if ddl.starts_with("ALTER TABLE ") {
            let words: Vec<_> = ddl.split_whitespace().collect();
            // Old init_db ignored duplicate-column errors. Reproduce its schema
            // without ignoring any actual SQL error in the fixture loader.
            if column_exists(connection, words[2], words[5]) {
                continue;
            }
        }
        connection.execute_batch(ddl).unwrap();
    }
    let identity = if column_exists(connection, "model_fingerprints", "file_path") {
        "file_path"
    } else {
        "gguf_hash"
    };
    connection.execute_batch(&format!(
        "INSERT INTO hardware_fingerprints (id, gpu_family, vram_gb, system_ram_gb, os_platform)
         VALUES (9, 'Test GPU', 16, 64, 'linux');
         INSERT INTO model_fingerprints (id, model_name, {identity}, quantization, architecture_type,
         expert_count, parameters_billions) VALUES (7, 'historical_model', 'original-identity', 'Q4_0', 'dense', 0, 7);
         INSERT INTO benchmark_runs (id, timestamp, hardware_id, model_id, flags_used,
         context_lengths_tested, raw_metrics, oom_boundary, benchmark_suite_version)
         VALUES (11, '2020-01-02 03:04:05', 9, 7, '[\"--threads\",\"8\"]', '[1024,8192]',
         '[]', 8192, 'historical-suite');
         INSERT INTO benchmark_runs (id, hardware_id, model_id, flags_used, raw_metrics)
         VALUES (12, 9, 0, '[]', '[]');"
    )).unwrap();
    let metrics = serde_json::to_string(&[
        BenchmarkResult::new(1024, 30.0, 100.0),
        BenchmarkResult::oom(8192),
    ])
    .unwrap();
    connection
        .execute(
            "UPDATE benchmark_runs SET raw_metrics = ?1 WHERE id = 11",
            [metrics],
        )
        .unwrap();
    if column_exists(connection, "benchmark_runs", "oom") {
        connection
            .execute("UPDATE benchmark_runs SET oom = 1 WHERE id = 11", [])
            .unwrap();
    }
}

fn rows(connection: &Connection, sql: &str) -> Vec<Vec<rusqlite::types::Value>> {
    let mut statement = connection.prepare(sql).unwrap();
    let width = statement.column_count();
    statement
        .query_map([], |row| (0..width).map(|index| row.get(index)).collect())
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}

fn database_state(connection: &Connection) -> Vec<Vec<Vec<rusqlite::types::Value>>> {
    let mut state = vec![
        rows(connection, "PRAGMA user_version"),
        rows(
            connection,
            "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name",
        ),
    ];
    for table in [
        "hardware_fingerprints",
        "model_fingerprints",
        "benchmark_runs",
        "sqlite_sequence",
    ] {
        state.push(rows(
            connection,
            &format!("SELECT * FROM {table} ORDER BY rowid"),
        ));
    }
    state
}

#[test]
fn all_ten_git_schemas_migrate_twice_retain_data_and_save_new_runs() {
    for schema in historical_schemas() {
        let dir = tempdir().unwrap();
        let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
        historical_database(&conn, &schema);
        let hardware_before = rows(&conn, "SELECT * FROM hardware_fingerprints ORDER BY id");
        let runs_before = rows(
            &conn,
            "SELECT id, timestamp, hardware_id, flags_used, context_lengths_tested,
            raw_metrics, oom_boundary, benchmark_suite_version FROM benchmark_runs ORDER BY id",
        );
        let identity = if column_exists(&conn, "model_fingerprints", "file_path") {
            "file_path"
        } else {
            "gguf_hash"
        };
        let model_sql = format!(
            "SELECT id, model_name, {identity}, quantization, architecture_type,
            expert_count, parameters_billions FROM model_fingerprints ORDER BY id"
        );
        let model_before = rows(&conn, &model_sql);
        let mut db = Database::open(dir.path())
            .unwrap_or_else(|error| panic!("{}: {error:#}", schema.commit));
        assert_eq!(
            rows(&conn, &model_sql),
            model_before,
            "{} model retention",
            schema.commit
        );
        assert_eq!(
            rows(
                &conn,
                "SELECT id, timestamp, hardware_id, flags_used, context_lengths_tested,
            raw_metrics, oom_boundary, benchmark_suite_version FROM benchmark_runs ORDER BY id"
            ),
            runs_before
        );
        // Every old hardware column and value remains, followed only by devices_json.
        let hardware_after = rows(&conn, "SELECT * FROM hardware_fingerprints ORDER BY id");
        assert_eq!(
            &hardware_after[0][..hardware_before[0].len()],
            &hardware_before[0]
        );
        assert_eq!(
            conn.query_row(
                "SELECT model_id FROM benchmark_runs WHERE id = 12",
                [],
                |row| row.get::<_, Option<i64>>(0)
            )
            .unwrap(),
            None
        );
        assert_eq!(
            conn.query_row("SELECT oom FROM benchmark_runs WHERE id = 11", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
            1
        );
        let new_run = save(&mut db, &hardware(), 80.0);
        assert_eq!(
            db.get_recommendations("Test GPU", 0.0, 1).unwrap()[0].id,
            new_run
        );
        drop(db);
        let before_second_init = database_state(&conn);
        init_db(dir.path()).unwrap();
        assert_eq!(
            database_state(&conn),
            before_second_init,
            "{} idempotence",
            schema.commit
        );
        let mut db = Database::open(dir.path()).unwrap();
        db.store_benchmark(
            &hardware(),
            &None,
            None,
            &[BenchmarkResult::new(1024, 50.0, 100.0)],
            &[],
            "after-second-init",
        )
        .unwrap();
        assert_eq!(count(&conn, "benchmark_runs"), 4);
        assert_eq!(count(&conn, "model_fingerprints"), 2);
        assert_eq!(rows(&conn, "PRAGMA user_version"), vec![vec![4i64.into()]]);
        assert!(rows(&conn, "PRAGMA foreign_key_check").is_empty());
    }
}

#[test]
fn all_ten_git_schemas_preserve_deleted_model_sequence_with_and_without_survivors() {
    for schema in historical_schemas() {
        for survive in [false, true] {
            let dir = tempdir().unwrap();
            let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
            historical_database(&conn, &schema);
            storage_identity::assert_model_sequence_preserved(dir.path(), &conn, survive);
        }
    }
}

#[test]
fn all_ten_git_schemas_roll_back_schema_data_and_version_on_migration_failure() {
    for schema in historical_schemas() {
        for corruption in ["flags", "foreign-key"] {
            let dir = tempdir().unwrap();
            let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
            historical_database(&conn, &schema);
            let sql = if corruption == "flags" {
                "UPDATE benchmark_runs SET flags_used = 'invalid-json' WHERE id = 12"
            } else {
                "UPDATE benchmark_runs SET hardware_id = 999 WHERE id = 11"
            };
            conn.execute(sql, []).unwrap();
            let before = database_state(&conn);
            assert!(
                Database::open(dir.path()).is_err(),
                "{} {corruption}",
                schema.commit
            );
            assert_eq!(
                database_state(&conn),
                before,
                "{} {corruption} rollback",
                schema.commit
            );
        }
    }
}

#[test]
fn version_two_missing_oom_is_repaired_without_losing_snapshots_and_is_atomic() {
    for fail in [false, true] {
        let dir = tempdir().unwrap();
        let mut db = Database::open(dir.path()).unwrap();
        let id = db
            .store_benchmark(
                &hardware(),
                &Some("old-model.gguf".to_string()),
                Some(&model("immutable_model")),
                &[BenchmarkResult::oom(8192)],
                &[],
                "old-server",
            )
            .unwrap();
        drop(db);
        let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
        conn.execute_batch("ALTER TABLE benchmark_runs DROP COLUMN oom; PRAGMA user_version = 2;")
            .unwrap();
        let snapshots = rows(&conn, "SELECT * FROM model_fingerprints");
        let runs = rows(&conn, "SELECT * FROM benchmark_runs");
        if fail {
            conn.pragma_update(None, "foreign_keys", false).unwrap();
            conn.execute("UPDATE benchmark_runs SET hardware_id = 999", [])
                .unwrap();
            let before = database_state(&conn);
            assert!(Database::open(dir.path()).is_err());
            assert_eq!(database_state(&conn), before);
            conn.execute("UPDATE benchmark_runs SET hardware_id = 1", [])
                .unwrap();
        }
        let mut db = Database::open(dir.path()).unwrap();
        assert_eq!(rows(&conn, "SELECT * FROM model_fingerprints"), snapshots);
        let after = rows(&conn, "SELECT * FROM benchmark_runs");
        assert_eq!(&after[0][..runs[0].len()], &runs[0]);
        assert_eq!(
            conn.query_row(
                "SELECT oom FROM benchmark_runs WHERE id = ?1",
                [id],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
            1
        );
        save(&mut db, &hardware(), 70.0);
        let before = database_state(&conn);
        init_db(dir.path()).unwrap();
        assert_eq!(database_state(&conn), before);
        save(&mut db, &hardware(), 75.0);
        assert_eq!(count(&conn, "benchmark_runs"), 3);
        assert!(rows(&conn, "PRAGMA foreign_key_check").is_empty());
    }
}

#[test]
fn numeric_metadata_null_and_explicit_zero_remain_distinct_in_sql_and_snapshots() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let mut absent = model("unknown_numbers");
    absent.expert_count = None;
    absent.block_count = None;
    absent.parameters_billions = None;
    absent.context_length = None;
    let mut zero = absent.clone();
    zero.model_name = Some("known_zero_numbers".to_string());
    zero.expert_count = Some(0);
    zero.block_count = Some(0);
    zero.parameters_billions = Some(0.0);
    zero.context_length = Some(0);
    let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
    for expected in [&absent, &zero] {
        let id = db
            .store_benchmark(
                &hardware(),
                &None,
                Some(expected),
                &[BenchmarkResult::new(1024, 10.0, 100.0)],
                &[],
                "test",
            )
            .unwrap();
        let (expert, blocks, parameters, context, snapshot):
            (Option<u32>, Option<u32>, Option<f64>, Option<u32>, String) = conn.query_row(
            "SELECT mf.expert_count, mf.block_count, mf.parameters_billions, mf.context_length, mf.snapshot_json
             FROM model_fingerprints mf JOIN benchmark_runs br ON br.model_id = mf.id WHERE br.id = ?1", [id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?))).unwrap();
        assert_eq!(expert, expected.expert_count);
        assert_eq!(blocks, expected.block_count);
        assert_eq!(parameters, expected.parameters_billions);
        assert_eq!(context, expected.context_length);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&snapshot).unwrap(),
            serde_json::to_value(expected).unwrap()
        );
    }
}

#[test]
fn version_two_repairs_missing_additive_columns_and_accepts_a_complete_new_snapshot() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let original = save(&mut db, &hardware(), 10.0);
    drop(db);
    let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
    // Retain identity and measurements while simulating an incomplete migration.
    conn.execute_batch(
        "ALTER TABLE benchmark_runs DROP COLUMN oom;
         ALTER TABLE benchmark_runs DROP COLUMN parsed_flags;
         ALTER TABLE benchmark_runs DROP COLUMN llama_server_version;
         ALTER TABLE hardware_fingerprints DROP COLUMN devices_json;
         ALTER TABLE model_fingerprints DROP COLUMN architecture;
         ALTER TABLE model_fingerprints DROP COLUMN block_count;
         ALTER TABLE model_fingerprints DROP COLUMN context_length;
         ALTER TABLE model_fingerprints DROP COLUMN snapshot_json;
         PRAGMA user_version = 2;",
    )
    .unwrap();
    let raw_before = rows(
        &conn,
        "SELECT id, hardware_id, model_id, raw_metrics FROM benchmark_runs",
    );
    let mut db = Database::open(dir.path()).unwrap();
    assert_eq!(
        rows(
            &conn,
            "SELECT id, hardware_id, model_id, raw_metrics FROM benchmark_runs"
        ),
        raw_before
    );
    let saved = save(&mut db, &hardware(), 90.0);
    assert_ne!(original, saved);
    assert_eq!(
        db.get_recommendations("Test GPU", 0.0, 1).unwrap()[0].id,
        saved
    );
    let snapshot: String = conn.query_row(
        "SELECT mf.snapshot_json FROM benchmark_runs br JOIN model_fingerprints mf ON br.model_id = mf.id WHERE br.id = ?1",
        [saved], |row| row.get(0)).unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&snapshot).unwrap(),
        serde_json::to_value(model("test_model")).unwrap()
    );
    let before = database_state(&conn);
    init_db(dir.path()).unwrap();
    assert_eq!(database_state(&conn), before);
}

#[test]
fn init_is_versioned_idempotent_and_creates_all_tables() {
    let dir = tempdir().unwrap();
    init_db(dir.path()).unwrap();
    init_db(dir.path()).unwrap();
    let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
    assert_eq!(
        conn.pragma_query_value(None, "user_version", |row| row.get::<_, i64>(0))
            .unwrap(),
        4
    );
    for table in [
        "hardware_fingerprints",
        "model_fingerprints",
        "benchmark_runs",
    ] {
        assert_eq!(count(&conn, table), 0);
    }
}

#[test]
fn both_hash_and_path_schemas_migrate_without_rewriting_identity_or_losing_runs() {
    for identity in ["gguf_hash", "file_path"] {
        for with_parsed in [false, true] {
            let dir = tempdir().unwrap();
            let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
            legacy_schema(&conn, identity, with_parsed);
            let mut db = Database::open(dir.path()).unwrap();
            assert_eq!(count(&conn, "benchmark_runs"), 2);
            let (path, hash): (Option<String>, Option<String>) = conn
                .query_row(
                    "SELECT file_path, gguf_hash FROM model_fingerprints WHERE id = 7",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .unwrap();
            if identity == "gguf_hash" {
                assert_eq!(hash.as_deref(), Some("legacy-identity"));
                assert_eq!(path, None);
            } else {
                assert_eq!(path.as_deref(), Some("legacy-identity"));
                assert_eq!(hash, None);
            }
            let (flags, parsed): (String, String) = conn
                .query_row(
                    "SELECT flags_used, parsed_flags FROM benchmark_runs WHERE id = 11",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .unwrap();
            assert!(!flags.contains("private"));
            assert!(!parsed.contains("private"));
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&parsed).unwrap(),
                serde_json::json!({"--model":"/model.gguf", "--threads":"8"})
            );
            assert_eq!(
                conn.query_row(
                    "SELECT model_id FROM benchmark_runs WHERE id = 12",
                    [],
                    |row| row.get::<_, Option<i64>>(0)
                )
                .unwrap(),
                None
            );
            let recommendations = db.get_recommendations("Test GPU", 20.0, 1).unwrap();
            assert_eq!(recommendations[0].id, 11);
            // The old path/hash constraint must no longer force an update.
            db.store_benchmark(
                &hardware(),
                &Some("legacy-identity".to_string()),
                Some(&model("new_model")),
                &[BenchmarkResult::new(1024, 40.0, 100.0)],
                &[],
                "test",
            )
            .unwrap();
            init_db(dir.path()).unwrap();
            assert_eq!(count(&conn, "model_fingerprints"), 2);
            assert_eq!(count(&conn, "benchmark_runs"), 3);
        }
    }
}

#[test]
fn migration_errors_roll_back_schema_backfills_and_version() {
    let dir = tempdir().unwrap();
    let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
    legacy_schema(&conn, "gguf_hash", false);
    conn.execute(
        "UPDATE benchmark_runs SET flags_used = 'not-json' WHERE id = 12",
        [],
    )
    .unwrap();
    let schema_before: String = conn
        .query_row(
            "SELECT sql FROM sqlite_master WHERE name = 'model_fingerprints'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(Database::open(dir.path()).is_err());
    assert_eq!(
        conn.pragma_query_value(None, "user_version", |row| row.get::<_, i64>(0))
            .unwrap(),
        0
    );
    let schema_after: String = conn
        .query_row(
            "SELECT sql FROM sqlite_master WHERE name = 'model_fingerprints'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(schema_before, schema_after);
    assert!(conn
        .prepare("SELECT file_path FROM model_fingerprints")
        .is_err());
    assert!(conn
        .prepare("SELECT parsed_flags FROM benchmark_runs")
        .is_err());
    let flags: String = conn
        .query_row(
            "SELECT flags_used FROM benchmark_runs WHERE id = 11",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(
        flags.contains("private-secret"),
        "a failed migration must roll back earlier backfills"
    );
    assert_eq!(count(&conn, "benchmark_runs"), 2);
}

#[test]
fn failed_foreign_key_check_rolls_back_the_whole_migration() {
    let dir = tempdir().unwrap();
    let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
    legacy_schema(&conn, "file_path", true);
    conn.execute(
        "UPDATE benchmark_runs SET hardware_id = 999 WHERE id = 11",
        [],
    )
    .unwrap();
    assert!(Database::open(dir.path()).is_err());
    assert_eq!(
        conn.pragma_query_value(None, "user_version", |row| row.get::<_, i64>(0))
            .unwrap(),
        0
    );
    assert!(conn
        .prepare("SELECT devices_json FROM hardware_fingerprints")
        .is_err());
    assert_eq!(count(&conn, "benchmark_runs"), 2);
}

#[test]
fn future_schema_versions_are_rejected_without_changes() {
    let dir = tempdir().unwrap();
    let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
    conn.pragma_update(None, "user_version", 999).unwrap();
    assert!(Database::open(dir.path()).is_err());
    assert_eq!(
        conn.pragma_query_value(None, "user_version", |row| row.get::<_, i64>(0))
            .unwrap(),
        999
    );
    assert_eq!(
        conn.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'",
            [],
            |row| row.get::<_, i64>(0)
        )
        .unwrap(),
        0
    );
}

#[test]
fn replacing_a_model_path_cannot_rewrite_old_model_snapshots() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
    let path = Some(
        dir.path()
            .join("replace.gguf")
            .to_string_lossy()
            .into_owned(),
    );
    let first = model("original_model");
    let mut replacement = model("replacement_model");
    replacement.quantization = Some("Q8_0".to_string());
    replacement.file_type = Some(7);
    replacement.context_length = Some(32_768);
    std::fs::write(path.as_ref().unwrap(), b"old bytes").unwrap();
    let old_run = db
        .store_benchmark(
            &hardware(),
            &path,
            Some(&first),
            &[BenchmarkResult::new(1024, 20.0, 100.0)],
            &[],
            "test",
        )
        .unwrap();
    std::fs::write(path.as_ref().unwrap(), b"new bytes").unwrap();
    let new_run = db
        .store_benchmark(
            &hardware(),
            &path,
            Some(&replacement),
            &[BenchmarkResult::new(1024, 25.0, 100.0)],
            &[],
            "test",
        )
        .unwrap();
    for (run, expected) in [(old_run, &first), (new_run, &replacement)] {
        let snapshot: String = conn
            .query_row(
                "SELECT mf.snapshot_json FROM benchmark_runs br
            JOIN model_fingerprints mf ON br.model_id = mf.id WHERE br.id = ?1",
                [run],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&snapshot).unwrap(),
            serde_json::to_value(expected).unwrap()
        );
    }
    assert_eq!(count(&conn, "model_fingerprints"), 2);
    assert_eq!(count(&conn, "benchmark_runs"), 2);
}

#[test]
fn deleting_the_file_after_parse_does_not_lose_completed_measurements() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("removed.gguf");
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    bytes.extend(1u64.to_le_bytes());
    let key = b"general.name";
    let name = b"captured_model";
    bytes.extend((key.len() as u64).to_le_bytes());
    bytes.extend(key);
    bytes.extend(8u32.to_le_bytes());
    bytes.extend((name.len() as u64).to_le_bytes());
    bytes.extend(name);
    std::fs::write(&path, bytes).unwrap();
    let info = parse_gguf(path.to_str().unwrap()).unwrap();
    std::fs::remove_file(&path).unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    db.store_benchmark(
        &hardware(),
        &Some(path.to_string_lossy().into_owned()),
        Some(&info),
        &[BenchmarkResult::new(1024, 12.0, 50.0)],
        &[],
        "test",
    )
    .unwrap();
    assert_eq!(
        db.get_recommendations("Test GPU", 0.0, 1).unwrap()[0]
            .model_name
            .as_deref(),
        Some("captured_model")
    );
}

#[test]
fn failures_at_each_insert_stage_roll_back_all_rows_and_allow_retry() {
    for table in [
        "hardware_fingerprints",
        "model_fingerprints",
        "benchmark_runs",
    ] {
        let dir = tempdir().unwrap();
        let mut db = Database::open(dir.path()).unwrap();
        let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
        conn.execute_batch(&format!(
            "CREATE TRIGGER fail_insert BEFORE INSERT ON {table}
            BEGIN SELECT RAISE(ABORT, 'injected failure'); END;"
        ))
        .unwrap();
        assert!(db
            .store_benchmark(
                &hardware(),
                &None,
                Some(&model("snapshot")),
                &[BenchmarkResult::new(1024, 10.0, 100.0)],
                &[],
                "test"
            )
            .is_err());
        for name in [
            "hardware_fingerprints",
            "model_fingerprints",
            "benchmark_runs",
        ] {
            assert_eq!(
                count(&conn, name),
                0,
                "orphan in {name} after failure in {table}"
            );
        }
        conn.execute_batch("DROP TRIGGER fail_insert").unwrap();
        save(&mut db, &hardware(), 10.0);
        for name in [
            "hardware_fingerprints",
            "model_fingerprints",
            "benchmark_runs",
        ] {
            assert_eq!(count(&conn, name), 1);
        }
    }
}

#[test]
fn absent_metadata_uses_sql_null_even_with_a_supplied_path() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    for path in [None, Some("a-missing-model.gguf".to_string())] {
        db.store_benchmark(
            &hardware(),
            &path,
            None,
            &[BenchmarkResult::new(1024, 12.0, 50.0)],
            &[],
            "test",
        )
        .unwrap();
    }
    let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
    assert_eq!(count(&conn, "model_fingerprints"), 0);
    assert_eq!(
        conn.query_row(
            "SELECT COUNT(*) FROM benchmark_runs WHERE model_id IS NULL",
            [],
            |row| row.get::<_, i64>(0)
        )
        .unwrap(),
        2
    );
    let recommendations = db.get_recommendations("Test GPU", 0.0, 10).unwrap();
    assert_eq!(recommendations.len(), 2);
    assert!(recommendations.iter().all(|run| run.model_name.is_none()));
    assert!(conn
        .prepare("PRAGMA foreign_key_check")
        .unwrap()
        .query([])
        .unwrap()
        .next()
        .unwrap()
        .is_none());
}

#[test]
fn a_commit_failure_also_rolls_back_hardware_model_and_run() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
    conn.execute_batch(
        "CREATE TABLE deferred_parent (id INTEGER PRIMARY KEY);
         CREATE TABLE deferred_probe (parent_id INTEGER REFERENCES deferred_parent(id) DEFERRABLE INITIALLY DEFERRED);
         CREATE TRIGGER fail_commit AFTER INSERT ON benchmark_runs
         BEGIN INSERT INTO deferred_probe (parent_id) VALUES (999); END;",
    ).unwrap();
    assert!(db
        .store_benchmark(
            &hardware(),
            &None,
            Some(&model("snapshot")),
            &[BenchmarkResult::new(1024, 10.0, 100.0)],
            &[],
            "test"
        )
        .is_err());
    for table in [
        "hardware_fingerprints",
        "model_fingerprints",
        "benchmark_runs",
        "deferred_probe",
    ] {
        assert_eq!(
            count(&conn, table),
            0,
            "commit failure left rows in {table}"
        );
    }
    conn.execute_batch("DROP TRIGGER fail_commit").unwrap();
    save(&mut db, &hardware(), 10.0);
    assert_eq!(count(&conn, "benchmark_runs"), 1);
}

#[test]
fn version_one_migrates_and_invalid_nonzero_model_references_are_not_hidden() {
    let dir = tempdir().unwrap();
    let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
    legacy_schema(&conn, "file_path", true);
    conn.execute_batch(
        "ALTER TABLE model_fingerprints ADD COLUMN gguf_hash TEXT;
        UPDATE benchmark_runs SET model_id = NULL WHERE model_id = 0;
        PRAGMA user_version = 1;",
    )
    .unwrap();
    let db = Database::open(dir.path()).unwrap();
    assert_eq!(
        db.get_recommendations("Test GPU", 0.0, 1).unwrap()[0].id,
        11
    );
    assert_eq!(
        conn.pragma_query_value(None, "user_version", |row| row.get::<_, i64>(0))
            .unwrap(),
        4
    );

    let other = tempdir().unwrap();
    let broken = Connection::open(other.path().join("benchmarks.db")).unwrap();
    legacy_schema(&broken, "gguf_hash", false);
    broken
        .execute("UPDATE benchmark_runs SET model_id = 999 WHERE id = 11", [])
        .unwrap();
    assert!(Database::open(other.path()).is_err());
    assert_eq!(
        broken
            .query_row(
                "SELECT model_id FROM benchmark_runs WHERE id = 11",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
        999
    );
    assert_eq!(
        broken
            .pragma_query_value(None, "user_version", |row| row.get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[test]
fn summaries_use_only_successful_contexts_and_keep_the_lowest_oom_boundary() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let measurements = vec![
        BenchmarkResult::new(1024, 20.0, 100.0),
        BenchmarkResult::oom(8192),
        BenchmarkResult::new(2048, 40.0, 200.0),
        BenchmarkResult::oom(4096),
    ];
    let id = db
        .store_benchmark(
            &hardware(),
            &None,
            None,
            &measurements,
            &[],
            "captured-version",
        )
        .unwrap();
    let summary = &db.get_recommendations("Test GPU", 0.0, 1).unwrap()[0];
    assert_eq!(summary.max_tps, 40.0);
    assert_eq!(summary.avg_tps, 30.0);
    assert_eq!(summary.oom_boundary, Some(4096));
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&summary.raw_metrics).unwrap(),
        serde_json::to_value(&measurements).unwrap()
    );
    let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
    let (oom, version, contexts): (i64, String, String) = conn.query_row(
        "SELECT oom, llama_server_version, context_lengths_tested FROM benchmark_runs WHERE id = ?1", [id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?))).unwrap();
    assert_eq!(oom, 1);
    assert_eq!(version, "captured-version");
    assert_eq!(
        serde_json::from_str::<Vec<usize>>(&contexts).unwrap(),
        vec![1024, 8192, 2048, 4096]
    );
}

#[test]
fn every_local_device_flag_form_is_removed_without_losing_tuning_values() {
    for flag in [
        "--device",
        "-dev",
        "--spec-draft-device",
        "--device-draft",
        "-devd",
        "--device_draft",
    ] {
        for arguments in [
            vec![
                flag.to_string(),
                "private-device".to_string(),
                "--threads=8".to_string(),
            ],
            vec![format!("{flag}=private-device"), "--threads=8".to_string()],
        ] {
            assert_eq!(sanitize_flags(&arguments), vec!["--threads=8".to_string()]);
        }
    }
}

#[test]
fn recommendations_filter_successes_and_rank_before_limiting() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let best = save(&mut db, &hardware(), 50.0);
    let second = save(&mut db, &hardware(), 40.0);
    for speed in [1.0, 2.0, 3.0] {
        save(&mut db, &hardware(), speed);
    }
    for status in ["oom", "error", "cancelled"] {
        let result = BenchmarkResult {
            context_length: 1024,
            tokens_per_second: 0.0,
            prompt_per_second: 0.0,
            status: status.to_string(),
            error: None,
        };
        db.store_benchmark(&hardware(), &None, None, &[result], &[], "test")
            .unwrap();
    }
    assert_eq!(
        db.get_recommendations("Test GPU", 10.0, 1).unwrap()[0].id,
        best
    );
    let selected = db.get_recommendations("Test GPU", 10.0, 2).unwrap();
    assert_eq!(
        selected.iter().map(|row| row.id).collect::<Vec<_>>(),
        vec![best, second]
    );
    assert_eq!(
        db.get_recommendations("Test GPU", 0.0, 1000).unwrap().len(),
        5
    );
    assert!(db
        .get_recommendations("Other GPU", 0.0, 1)
        .unwrap()
        .is_empty());
}

#[test]
fn recommendation_limits_thresholds_nullable_flags_and_corrupt_metrics_are_checked() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    save(&mut db, &hardware(), 10.0);
    for limit in [-1, 0, 1001, i64::MAX] {
        assert!(db.get_recommendations("Test GPU", 0.0, limit).is_err());
    }
    for threshold in [f64::NAN, f64::INFINITY, -1.0] {
        assert!(db.get_recommendations("Test GPU", threshold, 1).is_err());
    }
    let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
    conn.execute(
        "UPDATE benchmark_runs SET parsed_flags = NULL, flags_used = NULL",
        [],
    )
    .unwrap();
    let rows = db.get_recommendations("Test GPU", 0.0, 1).unwrap();
    assert_eq!(rows[0].parsed_flags, "{}");
    assert_eq!(rows[0].flags_used, "[]");
    conn.execute("UPDATE benchmark_runs SET raw_metrics = 'invalid-json'", [])
        .unwrap();
    assert!(db.get_recommendations("Test GPU", 0.0, 1).is_err());
}

#[test]
fn rates_and_integer_ranges_are_validated_before_writing() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let mut invalid_hardware = hardware();
    invalid_hardware.system_ram_gb = u64::MAX;
    assert!(db
        .store_benchmark(&invalid_hardware, &None, None, &[], &[], "test")
        .is_err());
    for rate in [f64::NAN, f64::INFINITY, -1.0] {
        assert!(db
            .store_benchmark(
                &hardware(),
                &None,
                None,
                &[BenchmarkResult::new(1024, rate, 1.0)],
                &[],
                "test"
            )
            .is_err());
    }
    let mut invalid_model = model("invalid");
    invalid_model.parameters_billions = Some(f64::NAN);
    assert!(db
        .store_benchmark(&hardware(), &None, Some(&invalid_model), &[], &[], "test")
        .is_err());
    assert!(db
        .store_benchmark(
            &hardware(),
            &None,
            None,
            &[BenchmarkResult::new(usize::MAX, 1.0, 1.0)],
            &[],
            "test"
        )
        .is_err());
    let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
    for table in [
        "hardware_fingerprints",
        "model_fingerprints",
        "benchmark_runs",
    ] {
        assert_eq!(count(&conn, table), 0);
    }
}

#[test]
fn delete_all_uses_ram_os_and_complete_order_independent_device_inventory() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let target = hardware();
    let mut variants = Vec::new();
    let mut variant = target.clone();
    variant.system_ram_gb = 32;
    variants.push(variant);
    let mut variant = target.clone();
    variant.os_platform = "windows".to_string();
    variants.push(variant);
    let mut variant = target.clone();
    variant.devices[0].id = "local-device-B".to_string();
    variants.push(variant);
    let mut variant = target.clone();
    variant.devices.clear();
    variants.push(variant);
    let mut variant = target.clone();
    variant.devices[0].memory_gb = 8.0;
    variants.push(variant);
    let mut variant = target.clone();
    variant.devices[0].name = "Other GPU".to_string();
    variants.push(variant);
    for variant in &variants {
        save(&mut db, variant, 10.0);
    }
    save(&mut db, &target, 10.0);
    save(&mut db, &target, 10.0);
    assert_eq!(db.delete_all_for_hardware(&target).unwrap(), 2);
    let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
    assert_eq!(count(&conn, "benchmark_runs"), variants.len() as i64);
    let mut multi = target.clone();
    multi.devices.push(HardwareDevice {
        id: "local-device-C".to_string(),
        name: "Test GPU".to_string(),
        memory_gb: 16.0,
    });
    save(&mut db, &multi, 10.0);
    multi.devices.reverse();
    save(&mut db, &multi, 10.0);
    assert_eq!(db.delete_all_for_hardware(&multi).unwrap(), 2);
    assert_eq!(count(&conn, "benchmark_runs"), variants.len() as i64);
}

#[test]
fn delete_last_uses_timestamp_then_id_for_ties() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let first = save(&mut db, &hardware(), 10.0);
    let second = save(&mut db, &hardware(), 10.0);
    let third = save(&mut db, &hardware(), 10.0);
    let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
    conn.execute(
        "UPDATE benchmark_runs SET timestamp = '2020-01-02 00:00:00'",
        [],
    )
    .unwrap();
    conn.execute(
        "UPDATE benchmark_runs SET timestamp = '2020-01-01 00:00:00' WHERE id = ?1",
        [third],
    )
    .unwrap();
    assert_eq!(db.delete_last_entry().unwrap(), 1);
    assert_eq!(
        conn.query_row(
            "SELECT COUNT(*) FROM benchmark_runs WHERE id = ?1",
            [second],
            |row| row.get::<_, i64>(0)
        )
        .unwrap(),
        0
    );
    assert_eq!(db.delete_last_entry().unwrap(), 1);
    assert_eq!(
        conn.query_row("SELECT id FROM benchmark_runs", [], |row| row
            .get::<_, i64>(0))
            .unwrap(),
        third
    );
    assert!(third > first);
    assert_eq!(db.delete_last_entry().unwrap(), 1);
    assert_eq!(db.delete_last_entry().unwrap(), 0);
}

#[test]
fn stored_and_read_flags_keep_the_same_sanitized_json_without_device_ids() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let flags: Vec<String> = [
        "--api_key",
        "-secret",
        "--hf-token=secret",
        "--device=local-device-A",
        "--spec-draft-device",
        "local-device-B",
        "--model-url",
        "https://user:password@example.test/private/model.gguf?token=secret,other#fragment",
        "--lora",
        "C:\\private\\a.gguf,/private/b.gguf",
        "--gpu_layers",
        "-1",
        "--threads=8",
    ]
    .iter()
    .map(|value| value.to_string())
    .collect();
    let id = db
        .store_benchmark(
            &hardware(),
            &None,
            None,
            &[BenchmarkResult::new(1024, 10.0, 100.0)],
            &flags,
            "test",
        )
        .unwrap();
    let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
    let (raw, parsed, devices): (String, String, String) = conn
        .query_row(
            "SELECT br.flags_used, br.parsed_flags, hf.devices_json FROM benchmark_runs br
         JOIN hardware_fingerprints hf ON br.hardware_id = hf.id WHERE br.id = ?1",
            [id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    let sanitized = sanitize_flags(&flags);
    assert_eq!(
        serde_json::from_str::<Vec<String>>(&raw).unwrap(),
        sanitized
    );
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&parsed).unwrap(),
        parse_flags(&sanitized)
    );
    assert!(devices.contains("local-device-A"));
    for forbidden in [
        "secret",
        "password",
        "private",
        "local-device-A",
        "local-device-B",
    ] {
        assert!(!raw.contains(forbidden));
        assert!(!parsed.contains(forbidden));
    }
    // Reapply privacy filtering when reading legacy raw flags; never trust stale parsed_flags.
    conn.execute(
        "UPDATE benchmark_runs SET flags_used = ?1, parsed_flags = '{\"--api-key\":\"secret\"}'",
        [serde_json::to_string(&flags).unwrap()],
    )
    .unwrap();
    let exported =
        serde_json::to_string(&db.get_recommendations("Test GPU", 0.0, 1).unwrap()).unwrap();
    for forbidden in [
        "secret",
        "password",
        "private",
        "local-device-A",
        "local-device-B",
    ] {
        assert!(!exported.contains(forbidden));
    }
}
