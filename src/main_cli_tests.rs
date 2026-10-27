use super::*;

#[test]
fn model_and_expert_metadata_respect_required_value_boundaries() {
    let flags = |items: &[&str]| {
        items
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        parse_model_flag(&flags(&[
            "--model",
            "real.gguf",
            "--alias",
            "--model=not-a-path"
        ])),
        Some("real.gguf".into())
    );
    assert_eq!(
        parse_model_flag(&flags(&["--model", "first.gguf", "-m=last.gguf"])),
        Some("last.gguf".into())
    );
    assert_eq!(
        parse_model_flag(&flags(&[
            "--alias",
            "--model",
            "--model",
            "--host=filename"
        ])),
        Some("--host=filename".into())
    );
    assert_eq!(
        parse_model_flag(&flags(&["--alias", "--model=fake.gguf"])),
        None
    );
    assert_eq!(
        extract_cpu_moe_layers(&flags(&["--alias", "--cpu-moe"]), 22),
        None
    );
    assert_eq!(
        extract_cpu_moe_layers(&flags(&["--n_cpu_moe=2", "--alias", "--n-cpu-moe=20"]), 22),
        Some(2)
    );
    // Native handlers append CPU tensor overrides; a later smaller prefix does not undo them.
    assert_eq!(
        extract_cpu_moe_layers(&flags(&["--n-cpu-moe", "5", "--n-cpu-moe", "2"]), 22),
        Some(5)
    );
    assert_eq!(
        extract_cpu_moe_layers(&flags(&["--cpu-moe", "--n-cpu-moe", "2"]), 22),
        Some(22)
    );
}

use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::test]
async fn submission_rejects_corrupt_saved_data_before_opening_a_connection() {
    for (column, value) in [
        ("flags_used", "not-json"),
        ("context_lengths_tested", "[]"),
        ("raw_metrics", "{}"),
        ("raw_metrics", "[]"),
    ] {
        let dir = submission_tests::benchmark_db();
        let conn = rusqlite::Connection::open(dir.path().join("benchmarks.db")).unwrap();
        conn.execute(&format!("UPDATE benchmark_runs SET {column}=?1"), [value])
            .unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        assert!(submission_tests::submit_to_fixture(&dir, &url)
            .await
            .is_err());
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(10), listener.accept())
                .await
                .is_err()
        );
        assert_eq!(
            conn.query_row("SELECT count(*) FROM benchmark_runs", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            1
        );
    }
}

#[tokio::test]
async fn submission_success_requires_a_positive_integer_id() {
    for body in [
        r#"{"id":0}"#,
        r#"{"id":-1}"#,
        r#"{"id":1.5}"#,
        r#"{"id":"7"}"#,
    ] {
        let dir = submission_tests::benchmark_db();
        let (url, server) = submission_tests::submission_server("200 OK", body).await;
        assert!(submission_tests::submit_to_fixture(&dir, &url)
            .await
            .is_err());
        server.await.unwrap();
    }
}

#[test]
fn compare_selector_contract_is_explicit() {
    for args in [
        vec!["autarky", "compare", "--id2", "1"],
        vec!["autarky", "compare", "--remote", "--id1", "1"],
        vec!["autarky", "compare", "--id1", "1", "--model", "x"],
    ] {
        assert!(Cli::try_parse_from(args).is_err());
    }
    assert!(Cli::try_parse_from(["autarky", "compare", "--model", "x", "--gpu", "y"]).is_ok());
}

#[test]
fn date_range_parser_checks_calendar_order_and_exact_pair() {
    assert_eq!(
        parse_date_range(Some(" 2024-02-29 , 2026-10-25 ")).unwrap(),
        Some(("2024-02-29".into(), "2026-10-25".into()))
    );
    for dates in [
        "",
        "2026-02-29,2026-03-01",
        "1900-02-29,1900-03-01",
        "2026-02-01,2026-01-01",
        "0000-01-01,2026-01-01",
        "2026-01-01,2026-01-02,2026-01-03",
        "模型模型,2026-01-01",
    ] {
        assert!(parse_date_range(Some(dates)).is_err(), "{dates}");
    }
    assert!(parse_date_range(Some("2000-02-29,2000-03-01")).is_ok());
}

#[test]
fn previous_benchmark_binds_id_model_and_gpu_without_losing_null_models() {
    let dir = submission_tests::benchmark_db();
    let conn = rusqlite::Connection::open(dir.path().join("benchmarks.db")).unwrap();
    conn.execute("INSERT INTO benchmark_runs(id,hardware_id,model_id,flags_used,raw_metrics) SELECT 2,hardware_id,NULL,flags_used,raw_metrics FROM benchmark_runs WHERE id=1", []).unwrap();
    for model in [None, Some("Test".to_string())] {
        for gpu in [None, Some("GPU".to_string())] {
            assert_eq!(load_previous(&conn, 2, model.clone(), gpu).unwrap().id, 1);
        }
    }
    assert!(load_benchmark(&conn, 2).unwrap().model.is_none());
}

async fn community_fixture(data: &Value) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/prefix/", listener.local_addr().unwrap());
    let body = data.to_string();
    let task = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        while !request.windows(4).any(|window| window == b"\r\n\r\n") {
            let mut chunk = [0; 4096];
            let size = stream.read(&mut chunk).await.unwrap();
            assert!(size > 0);
            request.extend_from_slice(&chunk[..size]);
        }
        assert!(String::from_utf8_lossy(&request).starts_with("GET /prefix/api/v1/benchmarks?"));
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(response.as_bytes()).await.unwrap();
    });
    (url, task)
}

fn remote_entry() -> Value {
    json!({"id":1,"submitted_at":"2026-10-25","model_name":"Test Model","quantization":"Q4", "gpu_family":"Test GPU", "vram_gb":16.0,"system_ram_gb":32,"os_platform":"linux","raw_metrics":[BenchmarkResult::new(1,50.0,50.0),BenchmarkResult::new(2,90.0,300.0)]})
}

#[tokio::test]
async fn community_suggestion_keeps_winning_context_metric_and_relative_improvement() {
    let hardware = hardware::HardwareInfo {
        gpu_family: "Test GPU".into(),
        vram_gb: 16.0,
        system_ram_gb: 32,
        os_platform: "linux".into(),
        devices: vec![],
    };
    let ours = vec![
        BenchmarkResult::new(1, 100.0, 100.0),
        BenchmarkResult::new(2, 100.0, 100.0),
    ];
    let (url, server) = community_fixture(&json!({"count":1,"results":[remote_entry()]})).await;
    let (_, context, metric, label) = check_community_comparison(
        &url,
        &hardware,
        "Test Model",
        "Q4",
        &ModelDescriptors::default(),
        &ours,
        None,
    )
    .await
    .unwrap()
    .unwrap();
    server.await.unwrap();
    assert_eq!(context, "2k");
    assert_eq!(metric, "300.0 prompt/s");
    assert!(label.contains("200.0% faster"));
}

#[tokio::test]
async fn community_discovery_is_not_model_quantization_or_hardware_equivalence() {
    let hardware = hardware::HardwareInfo {
        gpu_family: "Test GPU".into(),
        vram_gb: 16.0,
        system_ram_gb: 32,
        os_platform: "linux".into(),
        devices: vec![],
    };
    let ours = vec![BenchmarkResult::new(1, 1.0, 1.0)];
    for (field, value) in [
        ("model_name", json!("Test Model XL")),
        ("quantization", json!("Q8")),
        ("gpu_family", json!("Test GPU Plus")),
        ("vram_gb", json!(24)),
        ("system_ram_gb", json!(64)),
        ("os_platform", json!("windows")),
    ] {
        let mut entry = remote_entry();
        entry[field] = value;
        let (url, server) = community_fixture(&json!({"count":1,"results":[entry]})).await;
        assert!(
            check_community_comparison(
                &url,
                &hardware,
                "Test Model",
                "Q4",
                &ModelDescriptors::default(),
                &ours,
                None
            )
            .await
            .unwrap()
            .is_none(),
            "{field}"
        );
        server.await.unwrap();
    }
}

#[tokio::test]
async fn community_suggestions_reject_known_model_metadata_contradictions() {
    let hardware = hardware::HardwareInfo {
        gpu_family: "Test GPU".into(),
        vram_gb: 16.0,
        system_ram_gb: 32,
        os_platform: "linux".into(),
        devices: vec![],
    };
    let ours = [BenchmarkResult::new(1, 1.0, 1.0)];
    let descriptors = ModelDescriptors {
        architecture: Some("llama".into()),
        architecture_type: Some("dense".into()),
        expert_count: Some(0),
        parameters_billions: Some(1.0),
        block_count: Some(32),
        context_length: Some(8192),
    };
    for (field, value) in [
        ("architecture", json!("mistral")),
        ("architecture_type", json!("moe")),
        ("expert_count", json!(8)),
        ("parameters_billions", json!(2.0)),
        ("block_count", json!(40)),
        ("context_length", json!(16384)),
    ] {
        let mut entry = remote_entry();
        for (key, value) in serde_json::to_value(&descriptors)
            .unwrap()
            .as_object()
            .unwrap()
        {
            entry[key] = value.clone();
        }
        entry[field] = value;
        let (url, server) = community_fixture(&json!({"count":1,"results":[entry]})).await;
        assert!(
            check_community_comparison(
                &url,
                &hardware,
                "Test Model",
                "Q4",
                &descriptors,
                &ours,
                None
            )
            .await
            .unwrap()
            .is_none(),
            "{field}"
        );
        server.await.unwrap();
    }
}

#[tokio::test]
async fn legacy_saved_errors_and_unknown_prompt_fields_never_leave_submission() {
    let dir = submission_tests::benchmark_db();
    let conn = rusqlite::Connection::open(dir.path().join("benchmarks.db")).unwrap();
    let mut metrics = json!([
        BenchmarkResult::new(1, 42.0, 100.0),
        BenchmarkResult::error(2, "private-server-body")
    ]);
    metrics[0]["prompt"] = json!("private-prompt");
    conn.execute(
        "UPDATE benchmark_runs SET raw_metrics=?1,context_lengths_tested='[1,2]'",
        [metrics.to_string()],
    )
    .unwrap();
    let (url, server) = submission_tests::submission_server("200 OK", r#"{"id":9}"#).await;
    assert_eq!(
        submission_tests::submit_to_fixture(&dir, &url)
            .await
            .unwrap(),
        9
    );
    let payload = server.await.unwrap().to_string();
    assert!(!payload.contains("private-server-body"));
    assert!(!payload.contains("private-prompt"));
}
