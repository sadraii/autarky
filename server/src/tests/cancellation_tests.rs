use super::*;
use std::future::Future;

fn typed_payload() -> db::SubmitPayload {
    serde_json::from_value(payload()).unwrap()
}

#[actix_web::test]
async fn dropping_active_future_at_sqlite_precommit_gate_rolls_back_every_row() {
    let fixture = Fixture::new();
    let operation = db::Operation::new(Instant::now() + db::DB_DEADLINE);
    let (entered, resume) = operation.pause_commit(false);
    let database = fixture.state.db.clone();
    let task = actix_web::rt::spawn(async move {
        database
            .call_with(operation, |conn, deadline| {
                db::store_submission(conn, "cancel-active", &typed_payload(), 100, deadline)
            })
            .await
    });
    // All four table writes exist in the transaction, but none is committed.
    entered.await.unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    resume.send(()).unwrap();
    assert_eq!(fixture.counts().await, vec![0, 0, 0, 0]);
}

#[actix_web::test]
async fn dropping_pending_future_never_enters_the_worker_operation() {
    let fixture = Fixture::new();
    let database = fixture.state.db.clone();
    let (entered, receive) = tokio::sync::oneshot::channel();
    let (resume, wait) = std::sync::mpsc::channel();
    let blocker = actix_web::rt::spawn(async move {
        database
            .call(move |_, _| {
                entered.send(()).unwrap();
                wait.recv_timeout(Duration::from_secs(5)).unwrap();
                Ok(())
            })
            .await
    });
    receive.await.unwrap();
    let database = fixture.state.db.clone();
    let entered = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let observed = entered.clone();
    let mut pending = Box::pin(database.call(move |conn, deadline| {
        observed.store(true, std::sync::atomic::Ordering::SeqCst);
        db::store_submission(conn, "cancel-pending", &typed_payload(), 100, deadline)
    }));
    assert!(
        std::future::poll_fn(|cx| std::task::Poll::Ready(pending.as_mut().poll(cx)))
            .await
            .is_pending()
    );
    drop(pending);
    resume.send(()).unwrap();
    blocker.await.unwrap().unwrap();
    assert_eq!(fixture.counts().await, vec![0, 0, 0, 0]);
    assert!(!entered.load(std::sync::atomic::Ordering::SeqCst));
}

#[actix_web::test]
async fn expiry_at_precommit_gate_returns_retryable_503_without_later_rows() {
    let fixture = Fixture::new();
    let deadline = Instant::now() + Duration::from_millis(80);
    let operation = db::Operation::new(deadline);
    let (entered, resume) = operation.pause_commit(false);
    let database = fixture.state.db.clone();
    let task = actix_web::rt::spawn(async move {
        database
            .call_with(operation, |conn, deadline| {
                db::store_submission(conn, "expired-gate", &typed_payload(), 100, deadline)
            })
            .await
    });
    entered.await.unwrap();
    let response = db_failure(task.await.unwrap().unwrap_err());
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(response.headers().contains_key("Retry-After"));
    resume.send(()).unwrap();
    assert_eq!(fixture.counts().await, vec![0, 0, 0, 0]);
}

#[actix_web::test]
async fn commit_gate_winner_outlives_both_deadlines_and_returns_real_success() {
    let fixture = Fixture::new();
    let deadline = Instant::now() + Duration::from_millis(100);
    let operation = db::Operation::new(deadline);
    let (entered, resume) = operation.pause_commit(true);
    let database = fixture.state.db.clone();
    let outer = operation.clone();
    let task = actix_web::rt::spawn(async move {
        request_outcome(
            async move {
                let result = database
                    .call_with(operation, |conn, deadline| {
                        let id = db::store_submission(
                            conn,
                            "commit-wins",
                            &typed_payload(),
                            100,
                            deadline,
                        )?;
                        db::encode(&json!({"id":id}))
                    })
                    .await;
                Ok(json_response(result))
            },
            outer,
            deadline,
        )
        .await
    });
    entered.await.unwrap();
    tokio::time::sleep_until((deadline + Duration::from_millis(20)).into()).await;
    assert!(
        !task.is_finished(),
        "neither timeout may conceal a commit in progress"
    );
    resume.send(()).unwrap();
    let response = task.await.unwrap().unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = actix_web::body::to_bytes(response.into_body())
        .await
        .unwrap();
    let response: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(fixture.counts().await, vec![1, 1, 1, 1]);
    let app = app!(fixture);
    let rows = get!(app, "/api/v1/benchmarks");
    assert_eq!(rows["results"][0]["id"], response["id"]);
    assert_eq!(rows["results"][0]["raw_metrics"], payload()["raw_metrics"]);
}

#[actix_web::test]
async fn dropping_future_after_commit_gate_cannot_undo_the_transaction() {
    let fixture = Fixture::new();
    let operation = db::Operation::new(Instant::now() + db::DB_DEADLINE);
    let (entered, resume) = operation.pause_commit(true);
    let database = fixture.state.db.clone();
    let task = actix_web::rt::spawn(async move {
        database
            .call_with(operation, |conn, deadline| {
                db::store_submission(conn, "late-cancel", &typed_payload(), 100, deadline)
            })
            .await
    });
    entered.await.unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    resume.send(()).unwrap();
    assert_eq!(fixture.counts().await, vec![1, 1, 1, 1]);
}

#[actix_web::test]
async fn active_sql_progress_cancellation_rolls_back_and_releases_the_worker() {
    let fixture = Fixture::new();
    let database = fixture.state.db.clone();
    let (entered, receive) = tokio::sync::oneshot::channel();
    let (resume, wait) = std::sync::mpsc::channel();
    let task = actix_web::rt::spawn(async move {
        database.call(move |conn, _| {
            let tx = conn.transaction()?;
            tx.execute("INSERT INTO api_keys(key) VALUES ('progress-cancel')", [])?;
            entered.send(()).unwrap();
            wait.recv_timeout(Duration::from_secs(5)).unwrap();
            // Without active cancellation, this exceeds the job deadline.
            tx.query_row("WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<1000000000) SELECT sum(x) FROM n", [], |r| r.get::<_, i64>(0))?;
            tx.commit()?;
            Ok(())
        }).await
    });
    receive.await.unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    resume.send(()).unwrap();
    let start = Instant::now();
    assert_eq!(fixture.counts().await, vec![0, 0, 0, 0]);
    assert!(start.elapsed() < Duration::from_millis(500));
}

#[actix_web::test]
async fn request_timeout_cancels_an_active_precommit_transaction() {
    let fixture = Fixture::new();
    let operation = db::Operation::new(Instant::now() + db::DB_DEADLINE);
    let (entered, resume) = operation.pause_commit(false);
    let database = fixture.state.db.clone();
    let outer = operation.clone();
    let task = actix_web::rt::spawn(async move {
        request_outcome(
            async move {
                let result = database
                    .call_with(operation, |conn, deadline| {
                        db::store_submission(
                            conn,
                            "request-expired",
                            &typed_payload(),
                            100,
                            deadline,
                        )
                    })
                    .await;
                Ok(result)
            },
            outer,
            Instant::now() + Duration::from_millis(80),
        )
        .await
    });
    entered.await.unwrap();
    let error = task.await.unwrap().unwrap_err();
    assert_eq!(
        error.as_response_error().status_code(),
        StatusCode::REQUEST_TIMEOUT
    );
    resume.send(()).unwrap();
    assert_eq!(fixture.counts().await, vec![0, 0, 0, 0]);
}

#[actix_web::test]
async fn lost_worker_result_after_commit_is_not_a_retryable_503() {
    let fixture = Fixture::new();
    let result = fixture
        .state
        .db
        .call(|conn, deadline| -> db::Result<()> {
            db::store_submission(conn, "lost-result", &typed_payload(), 100, deadline)?;
            panic!("controlled worker failure after durable commit");
        })
        .await;
    let response = db_failure(result.unwrap_err());
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert!(!response.headers().contains_key("Retry-After"));
    let conn = Connection::open(&fixture.path).unwrap();
    assert_eq!(
        conn.query_row("SELECT COUNT(*) FROM benchmark_submissions", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
}

#[actix_web::test]
async fn commit_lock_wait_that_outlives_deadline_is_rejected_by_sqlite_hook() {
    let fixture = Fixture::new();
    let reader = Connection::open(&fixture.path).unwrap();
    reader
        .execute_batch("BEGIN; SELECT * FROM api_keys;")
        .unwrap();
    let deadline = Instant::now() + Duration::from_millis(400);
    let operation = db::Operation::new(deadline);
    let database = fixture.state.db.clone();
    let (entered, receive) = tokio::sync::oneshot::channel();
    let (resume, wait) = std::sync::mpsc::channel();
    let task = actix_web::rt::spawn(async move {
        database
            .call_with(operation, move |conn, deadline| {
                // Busy budget is set at worker admission, not at each statement.
                // This models bounded preparation before a late COMMIT lock wait.
                entered.send(()).unwrap();
                wait.recv_timeout(Duration::from_secs(5)).unwrap();
                db::store_submission(conn, "late-reader", &typed_payload(), 100, deadline)
            })
            .await
    });
    receive.await.unwrap();
    tokio::time::sleep_until((deadline - Duration::from_millis(100)).into()).await;
    resume.send(()).unwrap();
    let response = db_failure(task.await.unwrap().unwrap_err());
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    tokio::time::sleep_until((deadline + Duration::from_millis(50)).into()).await;
    reader.execute_batch("ROLLBACK").unwrap();
    assert_eq!(fixture.counts().await, vec![0, 0, 0, 0]);
}

#[actix_web::test]
async fn sqlite_reader_lock_crossing_http_deadline_never_commits_after_503() {
    use super::load_tests::wire_request;
    let fixture = Fixture::new();
    let database = fixture.state.db.clone();
    let (entered, receive) = tokio::sync::oneshot::channel();
    let (resume, wait) = std::sync::mpsc::channel();
    let blocker = actix_web::rt::spawn(async move {
        database
            .call(move |_, _| {
                entered.send(()).unwrap();
                wait.recv_timeout(Duration::from_secs(5)).unwrap();
                Ok(())
            })
            .await
    });
    receive.await.unwrap();
    let reader = Connection::open(&fixture.path).unwrap();
    reader
        .execute_batch("BEGIN; SELECT * FROM api_keys;")
        .unwrap();
    let (accepted, accepted_rx) = tokio::sync::oneshot::channel();
    let accepted = Arc::new(Mutex::new(Some(accepted)));
    let state = fixture.state.clone();
    let listener = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let server = HttpServer::new(move || {
        let accepted = accepted.clone();
        App::new()
            .app_data(state.clone())
            .wrap(middleware::from_fn(admission))
            .route("/health", web::get().to(health))
            .route(
                "/api/v1/benchmark",
                web::post().to(
                    move |state: web::Data<AppState>,
                          req: HttpRequest,
                          payload: web::Json<db::SubmitPayload>| {
                        let accepted = accepted.clone();
                        async move {
                            if let Some(send) = accepted.lock().unwrap().take() {
                                send.send(Instant::now()).unwrap();
                            }
                            submit(state, req, payload).await
                        }
                    },
                ),
            )
    })
    .workers(1)
    .listen(listener)
    .unwrap()
    .run();
    let handle = server.handle();
    let running = actix_web::rt::spawn(server);
    let body = payload().to_string();
    let request = actix_web::rt::spawn(wire_request(address, format!("POST /api/v1/benchmark HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nX-API-Key: deadline-lock\r\nContent-Length: {}\r\n\r\n{body}", body.len())));
    let accepted = accepted_rx.await.unwrap();
    // Admit the real POST to the worker near its two-second deadline. Its
    // COMMIT needs the reader's shared lock, which remains held past expiry.
    tokio::time::sleep_until((accepted + Duration::from_millis(1900)).into()).await;
    resume.send(()).unwrap();
    let health_start = Instant::now();
    let live = wire_request(
        address,
        "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".into(),
    )
    .await;
    assert!(live.starts_with("HTTP/1.1 200"));
    assert!(health_start.elapsed() < Duration::from_millis(150));
    let response = request.await.unwrap();
    assert!(response.starts_with("HTTP/1.1 503"), "{response}");
    tokio::time::sleep_until((accepted + Duration::from_millis(2050)).into()).await;
    reader.execute_batch("ROLLBACK").unwrap();
    let _ = blocker.await.unwrap();
    // A queued read fences the worker. This checks the exact post-failure state,
    // not a read that races an unfinished COMMIT.
    assert_eq!(fixture.counts().await, vec![0, 0, 0, 0]);
    handle.stop(true).await;
    running.await.unwrap().unwrap();
}
