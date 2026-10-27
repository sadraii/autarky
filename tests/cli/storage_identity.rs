use super::*;
use autarky::db::{RunMeasurements, RunOutcome};
use serde_json::{json, Value};

fn version_three(connection: &Connection) {
    // The actual v3 referenced model table, including its NOT NULL name.
    connection.execute_batch("PRAGMA foreign_keys = OFF;
        CREATE TABLE hardware_fingerprints (id INTEGER PRIMARY KEY AUTOINCREMENT, gpu_family TEXT NOT NULL,
          vram_gb REAL NOT NULL, system_ram_gb INTEGER, os_platform TEXT, devices_json TEXT NOT NULL DEFAULT '[]');
        CREATE TABLE model_fingerprints (id INTEGER PRIMARY KEY AUTOINCREMENT, model_name TEXT NOT NULL,
          file_path TEXT, gguf_hash TEXT, quantization TEXT, architecture_type TEXT,
          expert_count INTEGER DEFAULT 0, parameters_billions REAL, architecture TEXT,
          block_count INTEGER, context_length INTEGER, snapshot_json TEXT);
        CREATE INDEX idx_models_path ON model_fingerprints(file_path);
        CREATE TABLE benchmark_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
          hardware_id INTEGER REFERENCES hardware_fingerprints(id), model_id INTEGER REFERENCES model_fingerprints(id),
          flags_used TEXT, context_lengths_tested TEXT, raw_metrics TEXT, oom_boundary INTEGER,
          oom INTEGER DEFAULT 0, benchmark_suite_version TEXT, parsed_flags TEXT, llama_server_version TEXT);
        PRAGMA user_version = 3;
        INSERT INTO hardware_fingerprints (id,gpu_family,vram_gb,system_ram_gb,os_platform) VALUES(9,'Test GPU',16,64,'linux');
        INSERT INTO model_fingerprints(id,model_name,quantization,parameters_billions,architecture) VALUES(7,'unknown','Unknown',0.001,'unknown');
        INSERT INTO benchmark_runs(id,hardware_id,model_id,flags_used,context_lengths_tested,raw_metrics) VALUES(11,9,7,'[]','[1]','[]');").unwrap();
    connection
        .execute(
            "UPDATE benchmark_runs SET raw_metrics=?1",
            [json!([BenchmarkResult::new(1, 30.0, 60.0)]).to_string()],
        )
        .unwrap();
}

#[test]
fn version_four_rebuilds_nullable_identity_once_preserves_snapshots_ids_and_rolls_back() {
    for has_snapshot in [false, true] {
        let dir = tempdir().unwrap();
        let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
        version_three(&conn);
        let original =
            "{\"model_name\":\"unknown\",\"quantization\":\"Unknown\",\"private_extension\":42}";
        if has_snapshot {
            conn.execute("UPDATE model_fingerprints SET snapshot_json=?1", [original])
                .unwrap();
        }
        conn.execute("UPDATE benchmark_runs SET hardware_id=999", [])
            .unwrap();
        let before = database_state(&conn);
        assert!(Database::open(dir.path()).is_err());
        assert_eq!(
            database_state(&conn),
            before,
            "FK failure must roll back the full v4 rebuild"
        );
        conn.execute("UPDATE benchmark_runs SET hardware_id=9", [])
            .unwrap();
        let mut db = Database::open(dir.path()).unwrap();
        let (name, quant, architecture, snapshot): (Option<String>, Option<String>, Option<String>, String) = conn.query_row("SELECT model_name,quantization,architecture,snapshot_json FROM model_fingerprints WHERE id=7", [], |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?))).unwrap();
        assert_eq!((name, quant, architecture), (None, None, None));
        if has_snapshot {
            assert_eq!(snapshot, original);
        } else {
            let snapshot: Value = serde_json::from_str(&snapshot).unwrap();
            assert_eq!(snapshot["model_name"], "unknown");
            assert_eq!(snapshot["quantization"], "Unknown");
            assert_eq!(snapshot["parameters_billions"], 0.001);
        }
        assert_eq!(
            conn.query_row(
                "SELECT model_id FROM benchmark_runs WHERE id=11",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
            7
        );
        assert_eq!(
            conn.query_row("SELECT status FROM benchmark_runs WHERE id=11", [], |row| {
                row.get::<_, String>(0)
            })
            .unwrap(),
            "completed"
        );
        let columns = rows(&conn, "PRAGMA table_info(model_fingerprints)");
        assert_eq!(
            columns
                .iter()
                .find(|column| column[1] == rusqlite::types::Value::Text("model_name".into()))
                .unwrap()[3],
            0i64.into(),
            "The SQL type must allow NULL, not just a sentinel workaround"
        );
        let literal = model("unknown");
        db.store_benchmark(
            &hardware(),
            &None,
            Some(&literal),
            &[BenchmarkResult::new(1, 40.0, 80.0)],
            &[],
            "new",
        )
        .unwrap();
        let before = database_state(&conn);
        init_db(dir.path()).unwrap();
        assert_eq!(database_state(&conn), before);
        assert_eq!(
            rows(
                &conn,
                "SELECT model_name FROM model_fingerprints ORDER BY id DESC LIMIT 1"
            ),
            vec![vec![rusqlite::types::Value::Text("unknown".into())]],
            "Migration classification must never run on new data"
        );
        assert!(rows(&conn, "PRAGMA foreign_key_check").is_empty());
    }
}

pub(super) fn assert_model_sequence_preserved(
    dir: &std::path::Path,
    conn: &Connection,
    survive: bool,
) {
    // A deleted ID is not visible in max(id), but must never be issued again.
    conn.execute_batch(
        "INSERT INTO model_fingerprints(id,model_name) VALUES(500,'deleted');
        DELETE FROM model_fingerprints WHERE id=500;
        UPDATE benchmark_runs SET model_id=NULL WHERE model_id=0;",
    )
    .unwrap();
    if !survive {
        conn.execute_batch(
            "UPDATE benchmark_runs SET model_id=NULL; DELETE FROM model_fingerprints;",
        )
        .unwrap();
    }
    let retained = rows(
        conn,
        "SELECT id,model_id,raw_metrics FROM benchmark_runs ORDER BY id",
    );
    conn.execute("UPDATE benchmark_runs SET hardware_id=999 WHERE id=11", [])
        .unwrap();
    let before = database_state(conn);
    let error = Database::open(dir).err().expect("Migration must fail");
    assert!(error.to_string().contains("invalid foreign keys"));
    assert_eq!(
        database_state(conn),
        before,
        "Failure must roll back sqlite_sequence too"
    );
    assert_eq!(
        rows(
            conn,
            "SELECT seq FROM sqlite_sequence WHERE name='model_fingerprints'"
        ),
        vec![vec![500i64.into()]]
    );
    conn.execute("UPDATE benchmark_runs SET hardware_id=9 WHERE id=11", [])
        .unwrap();
    let mut db = Database::open(dir).unwrap();
    assert_eq!(
        rows(
            conn,
            "SELECT seq FROM sqlite_sequence WHERE name='model_fingerprints'"
        ),
        vec![vec![500i64.into()]]
    );
    assert_eq!(
        rows(
            conn,
            "SELECT id,model_id,raw_metrics FROM benchmark_runs ORDER BY id"
        ),
        retained
    );
    assert_eq!(
        rows(conn, "SELECT id FROM model_fingerprints"),
        if survive {
            vec![vec![7i64.into()]]
        } else {
            vec![]
        }
    );
    let run = save(&mut db, &hardware(), 20.0);
    assert_eq!(
        conn.query_row(
            "SELECT model_id FROM benchmark_runs WHERE id=?1",
            [run],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        501
    );
    assert!(rows(conn, "PRAGMA foreign_key_check").is_empty());
    let before = database_state(conn);
    init_db(dir).unwrap();
    assert_eq!(database_state(conn), before);
}

#[test]
fn rebuilt_model_sequence_never_reuses_deleted_identifiers_in_versions_zero_through_three() {
    for version in 0..=3 {
        for survive in [false, true] {
            let dir = tempdir().unwrap();
            let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
            if version < 2 {
                legacy_schema(&conn, "file_path", true);
                conn.execute_batch("ALTER TABLE model_fingerprints ADD COLUMN gguf_hash TEXT;")
                    .unwrap();
            } else {
                version_three(&conn);
            }
            conn.pragma_update(None, "user_version", version).unwrap();
            assert_model_sequence_preserved(dir.path(), &conn, survive);
        }
    }
}

#[test]
fn legacy_status_backfill_preserves_corrupt_metrics_and_rejects_nonobject_shapes() {
    let cases = [
        (None, "unknown"),
        (Some(""), "unknown"),
        (Some("not-json"), "unknown"),
        (Some("null"), "unknown"),
        (Some("true"), "unknown"),
        (Some("42"), "unknown"),
        (Some("\"bad\""), "unknown"),
        (Some("{}"), "unknown"),
        (Some("[]"), "unknown"),
        (Some("[\"bad\"]"), "unknown"),
        (Some(r#"["{\"status\":\"success\"}"]"#), "unknown"),
        (Some("[null]"), "unknown"),
        (Some("[true]"), "unknown"),
        (Some("[42]"), "unknown"),
        (Some("[[{\"status\":\"success\"}]]"), "unknown"),
        (Some("[{\"status\":\"success\"},\"bad\"]"), "unknown"),
        (Some("[null,{\"status\":\"cancelled\"}]"), "unknown"),
        (Some("[{\"status\":\"success\"}]"), "completed"),
        (Some("[{\"status\":\"error\"}]"), "failed"),
        (Some("[{\"status\":\"cancelled\"}]"), "cancelled"),
        (
            Some("[{\"status\":\"success\"},{\"status\":\"error\"}]"),
            "failed",
        ),
        (
            Some("[{\"status\":\"error\"},{\"status\":\"cancelled\"}]"),
            "cancelled",
        ),
    ];
    let dir = tempdir().unwrap();
    let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
    version_three(&conn);
    conn.execute("DELETE FROM benchmark_runs", []).unwrap();
    for (raw, _) in cases {
        conn.execute(
            "INSERT INTO benchmark_runs(hardware_id,model_id,raw_metrics) VALUES(9,7,?1)",
            [raw],
        )
        .unwrap();
    }
    conn.execute("UPDATE benchmark_runs SET hardware_id=999", [])
        .unwrap();
    let before = database_state(&conn);
    let error = Database::open(dir.path())
        .err()
        .expect("Migration must fail");
    assert!(error.to_string().contains("invalid foreign keys"));
    assert_eq!(database_state(&conn), before);
    conn.execute("UPDATE benchmark_runs SET hardware_id=9", [])
        .unwrap();
    Database::open(dir.path()).unwrap();
    let actual: Vec<(Option<String>, String)> = conn
        .prepare("SELECT raw_metrics,status FROM benchmark_runs ORDER BY id")
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(
        actual,
        cases
            .into_iter()
            .map(|(raw, status)| (raw.map(str::to_owned), status.to_owned()))
            .collect::<Vec<_>>()
    );
    let before = database_state(&conn);
    init_db(dir.path()).unwrap();
    assert_eq!(database_state(&conn), before);
}

#[test]
fn absent_identity_and_unknown_codes_keep_full_partial_sql_and_json_snapshots() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
    for (name, quant, raw) in [
        (None, None, None),
        (None, Some("Q4_K_M"), Some(15)),
        (Some("unknown"), None, Some(9999)),
        (Some("unknown"), Some("Q4_K_M"), Some(15)),
    ] {
        let mut partial = model("unused");
        partial.model_name = name.map(str::to_string);
        partial.quantization = quant.map(str::to_string);
        partial.architecture = None;
        partial.file_type = raw;
        let run = db
            .store_benchmark(
                &hardware(),
                &None,
                Some(&partial),
                &[BenchmarkResult::new(1, 25.0, 50.0)],
                &[],
                "test",
            )
            .unwrap();
        let (stored_name,stored_quant,code,snapshot): (Option<String>,Option<String>,Option<u32>,String) = conn.query_row("SELECT model_name,quantization,file_type,snapshot_json FROM model_fingerprints mf JOIN benchmark_runs br ON br.model_id=mf.id WHERE br.id=?1", [run], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?))).unwrap();
        assert_eq!(stored_name.as_deref(), name);
        assert_eq!(stored_quant.as_deref(), quant);
        assert_eq!(code, raw);
        assert_eq!(
            serde_json::from_str::<Value>(&snapshot).unwrap(),
            serde_json::to_value(&partial).unwrap()
        );
    }
}

#[test]
fn failed_output_is_a_failed_run_without_erasing_a_completed_point() {
    let dir = tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    let metrics = [BenchmarkResult::new(1, 25.0, 50.0)];
    for outcome in [RunOutcome::Failed("Broken pipe"), RunOutcome::Cancelled] {
        db.store_benchmark_run(
            &hardware(),
            &None,
            None,
            &RunMeasurements {
                results: &metrics,
                outcome,
            },
            &[],
            "test",
        )
        .unwrap();
    }
    let conn = Connection::open(dir.path().join("benchmarks.db")).unwrap();
    assert_eq!(
        rows(&conn, "SELECT status FROM benchmark_runs ORDER BY id"),
        vec![
            vec![rusqlite::types::Value::Text("failed".into())],
            vec![rusqlite::types::Value::Text("cancelled".into())]
        ]
    );
    for row in rows(&conn, "SELECT raw_metrics FROM benchmark_runs") {
        let rusqlite::types::Value::Text(raw) = &row[0] else {
            panic!("metrics must be JSON")
        };
        assert_eq!(serde_json::from_str::<Value>(raw).unwrap(), json!(metrics));
    }
}
