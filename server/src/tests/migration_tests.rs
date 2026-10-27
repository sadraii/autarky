use super::*;

#[actix_web::test]
async fn version_two_repairs_the_missing_server_version_column() {
    let fixture = Fixture::new();
    let path = fixture._dir.path().join("version-two.db");
    create_legacy(&path);
    let previous = db::Executor::open(&path, 16 * 1024 * 1024).unwrap();
    let before = previous
        .call(|conn, _| {
            // The previous migration could mark the database as version 2
            // without adding the column required by reads and submissions.
            conn.execute_batch(
                "ALTER TABLE benchmark_submissions DROP COLUMN llama_server_version;
                 PRAGMA user_version=2;",
            )?;
            Ok(conn.query_row(
                "SELECT raw_metrics FROM benchmark_submissions WHERE id=1",
                [],
                |row| row.get::<_, String>(0),
            )?)
        })
        .await
        .unwrap();
    drop(previous);
    for _ in 0..2 {
        let upgraded = db::Executor::open(&path, 16 * 1024 * 1024).unwrap();
        let expected = before.clone();
        upgraded
            .call(move |conn, _| {
                let stored: (String, Option<String>) = conn.query_row(
                    "SELECT raw_metrics,llama_server_version FROM benchmark_submissions WHERE id=1",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )?;
                assert_eq!(stored, (expected, None));
                Ok(())
            })
            .await
            .unwrap();
    }
    let fixture = Fixture {
        _dir: fixture._dir,
        state: web::Data::new(AppState {
            db: db::Executor::open(&path, 16 * 1024 * 1024).unwrap(),
            admission: Arc::new(Admission::new(1000, 1000, 60).unwrap()),
            submission_quota: 1000,
        }),
        path,
    };
    let app = app!(fixture);
    assert_eq!(get!(app, "/api/v1/benchmarks")["count"], 2);
    assert_eq!(post!(app, "new", &payload()), 3);
    assert_eq!(get!(app, "/api/v1/benchmarks")["count"], 3);
}

#[actix_web::test]
async fn initial_schema_preserves_unknown_versions_and_accepts_new_submissions_after_reopen() {
    let fixture = Fixture::new();
    let path = fixture._dir.path().join("initial.db");
    create_legacy(&path);
    let before = {
        let conn = Connection::open(&path).unwrap();
        conn.query_row(
            "SELECT raw_metrics FROM benchmark_submissions WHERE id=1",
            [],
            |row| row.get::<_, String>(0),
        )
        .unwrap()
    };
    for _ in 0..2 {
        let database = db::Executor::open(&path, 16 * 1024 * 1024).unwrap();
        database
            .call(|conn, _| {
                let unknown: i64 = conn.query_row(
                    "SELECT COUNT(*) FROM benchmark_submissions WHERE llama_server_version IS NULL",
                    [],
                    |row| row.get(0),
                )?;
                assert_eq!(unknown, 2, "Migration must not invent historical versions");
                assert!(conn
                    .query_row("PRAGMA foreign_key_check", [], |_| Ok(()))
                    .optional()?
                    .is_none());
                Ok(())
            })
            .await
            .unwrap();
    }
    let database = db::Executor::open(&path, 16 * 1024 * 1024).unwrap();
    let fixture = Fixture {
        _dir: fixture._dir,
        path,
        state: web::Data::new(AppState {
            db: database,
            admission: Arc::new(Admission::new(1000, 1000, 60).unwrap()),
            submission_quota: 1000,
        }),
    };
    let app = app!(fixture);
    assert_eq!(post!(app, "new", &payload()), 3);
    fixture
        .state
        .db
        .call(move |conn, _| {
            let raw: String = conn.query_row(
                "SELECT raw_metrics FROM benchmark_submissions WHERE id=1",
                [],
                |row| row.get(0),
            )?;
            assert_eq!(raw, before, "Migration must preserve original metrics");
            let version: String = conn.query_row(
                "SELECT llama_server_version FROM benchmark_submissions WHERE id=3",
                [],
                |row| row.get(0),
            )?;
            assert_eq!(version, "test");
            Ok(())
        })
        .await
        .unwrap();
}

#[actix_web::test]
async fn sqlite_file_quota_rolls_back_the_entire_rejected_submission() {
    let fixture = Fixture::new();
    let path = fixture._dir.path().join("small.db");
    let database = db::Executor::open(&path, 1024 * 1024).unwrap();
    let fixture = Fixture {
        _dir: fixture._dir,
        path,
        state: web::Data::new(AppState {
            db: database,
            admission: Arc::new(Admission::new(1000, 1000, 60).unwrap()),
            submission_quota: 1000,
        }),
    };
    let app = app!(fixture);
    let mut body = payload();
    body["flags_used"] = json!(vec!["x".repeat(500); 100]);
    let mut accepted = 0;
    for i in 0..100 {
        let response = test::call_service(
            &app,
            test::TestRequest::post()
                .uri("/api/v1/benchmark")
                .insert_header(("X-API-Key", format!("key-{i}")))
                .set_json(&body)
                .to_request(),
        )
        .await;
        if response.status() == StatusCode::INSUFFICIENT_STORAGE {
            break;
        }
        assert_eq!(response.status(), StatusCode::OK);
        accepted += 1;
    }
    assert!((1..100).contains(&accepted));
    assert_eq!(fixture.counts().await, vec![accepted, 1, 1, accepted]);
    fixture
        .state
        .db
        .call(|conn, _| {
            let pages: u64 = conn.pragma_query_value(None, "page_count", |r| r.get(0))?;
            let size: u64 = conn.pragma_query_value(None, "page_size", |r| r.get(0))?;
            assert!(pages * size <= 1024 * 1024);
            Ok(())
        })
        .await
        .unwrap();
    assert_eq!(get!(app, "/ready")["status"], "ready");
}

#[actix_web::test]
async fn duplicate_migration_crosses_batch_boundaries_and_normalizes_exact_lookup_fields() {
    let fixture = Fixture::new();
    let path = fixture._dir.path().join("many-legacy.db");
    create_legacy(&path);
    let mut conn = Connection::open(&path).unwrap();
    let transaction = conn.transaction().unwrap();
    transaction
        .execute("UPDATE model_fingerprints SET model_name=' model '", [])
        .unwrap();
    transaction
        .execute("UPDATE hardware_fingerprints SET os_platform=' LINUX '", [])
        .unwrap();
    for id in 3..=600 {
        transaction.execute("INSERT INTO hardware_fingerprints(id,gpu_family,vram_gb,system_ram_gb,os_platform) VALUES (?,'CPU',0,32,'linux')",[id]).unwrap();
        transaction.execute("INSERT INTO model_fingerprints(id,model_name,quantization,architecture_type,expert_count,parameters_billions) VALUES (?,'model','Q4','dense',0,7)",[id]).unwrap();
        transaction.execute("INSERT INTO benchmark_submissions(id,api_key_id,hardware_id,model_id,flags_used,context_lengths_tested,raw_metrics) SELECT ?,1,?,?,flags_used,context_lengths_tested,raw_metrics FROM benchmark_submissions WHERE id=1",params![id,id,id]).unwrap();
    }
    transaction.commit().unwrap();
    drop(conn);
    let database = db::Executor::open(&path, 16 * 1024 * 1024).unwrap();
    let fixture = Fixture {
        _dir: fixture._dir,
        path,
        state: web::Data::new(AppState {
            db: database,
            admission: Arc::new(Admission::new(1000, 1000, 60).unwrap()),
            submission_quota: 1000,
        }),
    };
    let app = app!(fixture);
    assert_eq!(fixture.counts().await, vec![1, 1, 1, 600]);
    assert_eq!(get!(app, "/api/v1/models/model?limit=50")["count"], 50);
    assert_eq!(
        get!(app, "/api/v1/benchmarks?os=linux&offset=599")["count"],
        1
    );
    assert_eq!(
        get!(app, "/api/v1/hardware/1/benchmarks?offset=599")["count"],
        1
    );
}

#[actix_web::test]
async fn failed_migration_rolls_back_schema_and_data_instead_of_resetting_the_database() {
    let fixture = Fixture::new();
    let path = fixture._dir.path().join("failing-legacy.db");
    create_legacy(&path);
    let conn = Connection::open(&path).unwrap();
    conn.execute_batch("CREATE TRIGGER fail_merge BEFORE UPDATE ON benchmark_submissions BEGIN SELECT RAISE(ABORT,'test migration failure'); END;").unwrap();
    assert!(db::Executor::open(&path, 16 * 1024 * 1024).is_err());
    assert_eq!(
        conn.pragma_query_value(None, "user_version", |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
    assert_eq!(
        conn.query_row("SELECT COUNT(*) FROM benchmark_submissions", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        2
    );
    assert_eq!(
        conn.query_row("SELECT COUNT(*) FROM hardware_fingerprints", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        2
    );
    let mut info = conn
        .prepare("PRAGMA table_info(hardware_fingerprints)")
        .unwrap();
    let columns = info
        .query_map([], |r| r.get::<_, String>(1))
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap();
    assert!(!columns.iter().any(|c| c == "identity"));
}

#[actix_web::test]
async fn future_schema_is_rejected_and_partial_known_columns_upgrade_without_data_loss() {
    let fixture = Fixture::new();
    let app = app!(fixture);
    post!(app, "key", &payload());
    fixture
        .state
        .db
        .call(|conn, _| {
            conn.pragma_update(None, "user_version", 1)?;
            Ok(())
        })
        .await
        .unwrap();
    let upgraded = db::Executor::open(&fixture.path, 16 * 1024 * 1024).unwrap();
    upgraded
        .call(|conn, _| {
            assert_eq!(
                conn.pragma_query_value(None, "user_version", |r| r.get::<_, i64>(0))?,
                3
            );
            conn.pragma_update(None, "user_version", 999)?;
            Ok(())
        })
        .await
        .unwrap();
    assert!(db::Executor::open(&fixture.path, 16 * 1024 * 1024).is_err());
    assert_eq!(fixture.counts().await, vec![1, 1, 1, 1]);
}
