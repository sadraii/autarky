use super::*;

#[actix_web::test]
async fn response_size_has_a_hard_bound_and_a_smaller_page_still_works() {
    let fixture = Fixture::new();
    let app = app!(fixture);
    let mut body = payload();
    body["flags_used"] = json!(vec!["x".repeat(500); 100]);
    for _ in 0..45 {
        post!(app, "key", &body);
    }
    let response = test::call_service(
        &app,
        test::TestRequest::get()
            .uri("/api/v1/benchmarks?limit=50")
            .to_request(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(test::read_body(response).await.len() < 1024);
    assert_eq!(get!(app, "/api/v1/benchmarks?limit=1")["count"], 1);
    fixture
        .state
        .db
        .call(|conn, _| {
            conn.execute(
                "UPDATE benchmark_submissions SET raw_metrics=? WHERE id=1",
                ["x".repeat(db::MAX_BODY + 1)],
            )?;
            Ok(())
        })
        .await
        .unwrap();
    let response = test::call_service(
        &app,
        test::TestRequest::get()
            .uri("/api/v1/benchmarks?limit=1&offset=44")
            .to_request(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
}

#[actix_web::test]
async fn bounded_database_queue_fails_fast_and_expired_jobs_do_not_write() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let fixture = Fixture::new();
    let app = app!(fixture);
    let database = fixture.state.db.clone();
    let (entered_send, entered_recv) = tokio::sync::oneshot::channel();
    let blocker = actix_web::rt::spawn(async move {
        database
            .call(move |_, _| {
                let _ = entered_send.send(());
                std::thread::sleep(Duration::from_millis(2200));
                Ok(())
            })
            .await
    });
    entered_recv.await.unwrap();
    let writes = Arc::new(AtomicUsize::new(0));
    let mut queued = Vec::new();
    for _ in 0..16 {
        let database = fixture.state.db.clone();
        let writes = writes.clone();
        queued.push(actix_web::rt::spawn(async move {
            database
                .call(move |_, _| {
                    writes.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                })
                .await
        }));
    }
    tokio::time::sleep(Duration::from_millis(20)).await;
    let start = Instant::now();
    let response = test::call_service(
        &app,
        test::TestRequest::get()
            .uri("/api/v1/benchmarks")
            .to_request(),
    )
    .await;
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(start.elapsed() < Duration::from_millis(100));
    assert_eq!(get!(app, "/health")["status"], "ok");
    assert!(matches!(
        blocker.await.unwrap(),
        Err(db::Error::Unavailable)
    ));
    for queued in queued {
        assert!(matches!(queued.await.unwrap(), Err(db::Error::Unavailable)));
    }
    tokio::time::sleep(Duration::from_millis(250)).await;
    assert_eq!(writes.load(Ordering::SeqCst), 0);
    assert_eq!(get!(app, "/ready")["status"], "ready");
}

pub(super) async fn wire_request(address: std::net::SocketAddr, request: String) -> String {
    tokio::task::spawn_blocking(move || {
        use std::io::{Read, Write};
        let mut stream =
            std::net::TcpStream::connect_timeout(&address, Duration::from_secs(2)).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(7)))
            .unwrap();
        stream.write_all(request.as_bytes()).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        response
    })
    .await
    .unwrap()
}

#[actix_web::test]
async fn real_loopback_http_routes_store_and_read_back_the_exact_submission() {
    let fixture = Fixture::new();
    let state = fixture.state.clone();
    let listener = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let server = HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .wrap(middleware::from_fn(admission))
            .configure(routes)
    })
    .workers(1)
    .listen(listener)
    .unwrap()
    .run();
    let handle = server.handle();
    let running = actix_web::rt::spawn(server);
    let body = payload().to_string();
    let response=wire_request(address,format!("POST /api/v1/benchmark HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nX-API-Key: wire-key\r\nContent-Length: {}\r\n\r\n{body}",body.len())).await;
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    let result: Value = serde_json::from_str(response.split_once("\r\n\r\n").unwrap().1).unwrap();
    let id = result["id"].as_i64().unwrap();
    assert!(id > 0);
    let response = wire_request(
        address,
        "GET /api/v1/benchmarks HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".into(),
    )
    .await;
    assert!(response.starts_with("HTTP/1.1 200"));
    let result: Value = serde_json::from_str(response.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(result["results"][0]["id"], id);
    assert_eq!(
        result["results"][0]["raw_metrics"],
        payload()["raw_metrics"]
    );
    let lock = Connection::open(&fixture.path).unwrap();
    lock.execute_batch("BEGIN EXCLUSIVE;").unwrap();
    let blocked=wire_request(address,format!("POST /api/v1/benchmark HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nX-API-Key: contention-key\r\nContent-Length: {}\r\n\r\n{body}",body.len()));
    let health = async {
        tokio::time::sleep(Duration::from_millis(20)).await;
        let start = Instant::now();
        let response = wire_request(
            address,
            "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".into(),
        )
        .await;
        assert!(response.starts_with("HTTP/1.1 200"));
        assert!(start.elapsed() < Duration::from_millis(150));
    };
    let (blocked, ()) = tokio::join!(blocked, health);
    assert!(blocked.starts_with("HTTP/1.1 503"));
    lock.execute_batch("ROLLBACK;").unwrap();
    assert_eq!(fixture.counts().await, vec![1, 1, 1, 1]);
    handle.stop(true).await;
    running.await.unwrap().unwrap();
}

#[actix_web::test]
async fn slow_json_body_has_a_request_deadline_and_does_not_block_health() {
    let fixture = Fixture::new();
    let state = fixture.state.clone();
    let listener = std::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let address = listener.local_addr().unwrap();
    let server = HttpServer::new(move || {
        App::new()
            .app_data(state.clone())
            .wrap(middleware::from_fn(admission))
            .configure(routes)
    })
    .workers(1)
    .listen(listener)
    .unwrap()
    .run();
    let handle = server.handle();
    let running = actix_web::rt::spawn(server);
    let slow=wire_request(address,"POST /api/v1/benchmark HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nX-API-Key: slow-key\r\nContent-Length: 1000\r\n\r\n{}".into());
    let health = async {
        tokio::time::sleep(Duration::from_millis(50)).await;
        let response = wire_request(
            address,
            "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n".into(),
        )
        .await;
        assert!(response.starts_with("HTTP/1.1 200"));
    };
    let start = Instant::now();
    let (response, ()) = tokio::join!(slow, health);
    assert!(response.starts_with("HTTP/1.1 408"), "{response}");
    assert!(start.elapsed() < Duration::from_secs(7));
    assert_eq!(fixture.counts().await, vec![0, 0, 0, 0]);
    handle.stop(true).await;
    running.await.unwrap().unwrap();
}
