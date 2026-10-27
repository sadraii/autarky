//! Only private subprocesses and loopback fixtures are used by these tests.
#![cfg(unix)]
use crate::launcher::{launch_server, Server};
use crate::test_process::process_exists;
use std::{os::unix::fs::PermissionsExt, time::Duration};
use tempfile::TempDir;

fn fixture() -> TempDir {
    // Use an immutable executable for the whole parallel test run. No writable
    // descriptor can be inherited by another child while this fixture is built.
    let source =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inference-server.py");
    let directory = tempfile::Builder::new()
        .prefix("autarky-runtime-test-")
        .tempdir()
        .unwrap();
    std::os::unix::fs::symlink(source, directory.path().join("server.py")).unwrap();
    directory
}
async fn start(directory: &TempDir, mode: &str, extra: &[&str]) -> Server {
    let mut flags = vec![
        "--tags".into(),
        directory.path().display().to_string(),
        "--reasoning-budget-message".into(),
        mode.into(),
    ];
    flags.extend(extra.iter().map(|value| value.to_string()));
    launch_server(directory.path().join("server.py").to_str().unwrap(), &flags)
        .await
        .unwrap()
}
fn pid(directory: &TempDir) -> u32 {
    std::fs::read_to_string(directory.path().join("pid"))
        .unwrap()
        .parse()
        .unwrap()
}
async fn reaped(pid: u32) {
    tokio::time::timeout(Duration::from_secs(3), async {
        while process_exists(pid) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("owned child must be killed and reaped, not left as a zombie");
}

#[tokio::test]
async fn every_repeat_uses_its_own_complete_response_and_auth_is_not_recorded() {
    let directory = fixture();
    let mut server = start(&directory, "", &["--api-key", "fixture-not-a-real-secret"]).await;
    server.wait_ready().await.unwrap();
    assert!(server
        .full_flags
        .iter()
        .all(|flag| !flag.contains("fixture-not-a-real-secret") && flag != "--api-key"));
    let result = server.benchmark(1, 2, 1, None).await.unwrap();
    assert_eq!(result.tokens_per_second, 25.0);
    assert_eq!(result.prompt_per_second, 250.0);
    assert_eq!(
        std::fs::read_to_string(directory.path().join("requests")).unwrap(),
        "3"
    );
    let pid = pid(&directory);
    server.kill().await.unwrap();
    reaped(pid).await;
}

#[tokio::test]
async fn partial_and_stale_timings_fail_the_whole_measurement_and_reap() {
    for mode in ["partial-second", "stale-count", "error"] {
        let directory = fixture();
        let mut server = start(&directory, mode, &[]).await;
        let pid = pid(&directory);
        assert!(
            server.benchmark(1, 2, 0, None).await.is_err(),
            "{} must not produce a partial success",
            mode
        );
        reaped(pid).await;
    }
    let directory = fixture();
    let mut server = start(&directory, "oom", &[]).await;
    assert_eq!(server.benchmark(1, 1, 0, None).await.unwrap().status, "oom");
    reaped(pid(&directory)).await;
}

#[tokio::test]
async fn concurrent_servers_bind_distinct_owned_loopback_ports_and_drop_reaps() {
    let first = fixture();
    let second = fixture();
    let (first_server, second_server) =
        tokio::join!(start(&first, "", &[]), start(&second, "", &[]));
    assert_ne!(first_server.api_base(), second_server.api_base());
    assert!(first_server.api_base().starts_with("http://127.0.0.1:"));
    let first_pid = pid(&first);
    let second_pid = pid(&second);
    drop(first_server);
    drop(second_server);
    tokio::join!(reaped(first_pid), reaped(second_pid));
}

#[tokio::test]
async fn cancellation_during_active_request_kills_and_reaps_child() {
    let directory = fixture();
    let mut server = start(&directory, "request-hang", &[]).await;
    let pid = pid(&directory);
    assert!(
        tokio::time::timeout(Duration::from_millis(150), server.benchmark(1, 1, 0, None))
            .await
            .is_err()
    );
    reaped(pid).await;
}

#[tokio::test]
async fn cancellation_during_startup_kills_and_reaps_child() {
    let directory = fixture();
    let flags = vec![
        "--tags".into(),
        directory.path().display().to_string(),
        "--reasoning-budget-message".into(),
        "startup-hang".into(),
    ];
    assert!(tokio::time::timeout(
        Duration::from_millis(200),
        launch_server(directory.path().join("server.py").to_str().unwrap(), &flags)
    )
    .await
    .is_err());
    reaped(pid(&directory)).await;
}

#[tokio::test]
async fn incompatible_endpoint_and_tls_options_are_rejected_before_spawn() {
    for flags in [
        vec!["--host=0.0.0.0"],
        vec!["--port", "9999"],
        vec!["--ssl-key-file=secret.pem"],
        vec!["--reuse-port"],
        vec!["--api-prefix=/private"],
    ] {
        assert!(launch_server(
            "/does/not/exist",
            &flags
                .iter()
                .map(|flag| flag.to_string())
                .collect::<Vec<_>>()
        )
        .await
        .is_err());
    }
}

#[tokio::test]
async fn private_logs_are_distinct_and_removed_after_process_reaping() {
    let first = fixture();
    let second = fixture();
    let first_server = start(&first, "", &[]).await;
    let second_server = start(&second, "", &[]).await;
    let first_log = first_server.log.path().to_path_buf();
    let second_log = second_server.log.path().to_path_buf();
    assert_ne!(first_log, second_log);
    for path in [&first_log, &second_log] {
        assert_eq!(
            std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
    let first_pid = pid(&first);
    let second_pid = pid(&second);
    drop(first_server);
    drop(second_server);
    tokio::join!(reaped(first_pid), reaped(second_pid));
    assert!(!first_log.exists());
    assert!(!second_log.exists());
}

#[tokio::test]
async fn hung_health_response_cannot_exceed_outer_startup_deadline() {
    let directory = fixture();
    let mut server = start(&directory, "health-hang", &[]).await;
    server.startup_deadline = tokio::time::Instant::now() + Duration::from_millis(100);
    let child_pid = pid(&directory);
    let began = std::time::Instant::now();
    assert!(server.wait_ready().await.is_err());
    assert!(began.elapsed() < Duration::from_secs(2));
    reaped(child_pid).await;
}

#[tokio::test]
async fn invalid_repeat_count_stops_child_instead_of_leaking_it() {
    let directory = fixture();
    let mut server = start(&directory, "", &[]).await;
    let child_pid = pid(&directory);
    assert!(server.benchmark(1, 0, 0, None).await.is_err());
    reaped(child_pid).await;
}

#[test]
fn timings_require_both_finite_positive_rates_and_counts_from_this_response() {
    let value = serde_json::json!({"stop":true,"truncated":false,"tokens_predicted":3,"tokens_evaluated":10,
        "timings":{"cache_n":0,"prompt_n":10,"predicted_n":3,"prompt_ms":1,"predicted_ms":2,
                   "prompt_per_second":10000,"predicted_per_second":1500}});
    assert!(super::parse_response_timing(&value).is_ok());
    for field in [
        "prompt_ms",
        "predicted_ms",
        "prompt_per_second",
        "predicted_per_second",
        "prompt_n",
        "predicted_n",
        "cache_n",
    ] {
        let mut missing = value.clone();
        missing["timings"].as_object_mut().unwrap().remove(field);
        assert!(
            super::parse_response_timing(&missing).is_err(),
            "{} cannot be optional",
            field
        );
    }
    for field in ["prompt_per_second", "predicted_per_second"] {
        for rate in [
            serde_json::Value::Null,
            serde_json::json!(0),
            serde_json::json!(-1),
        ] {
            let mut invalid = value.clone();
            invalid["timings"][field] = rate;
            assert!(super::parse_response_timing(&invalid).is_err());
        }
    }
    let mut partial = value.clone();
    partial["stop"] = false.into();
    assert!(super::parse_response_timing(&partial).is_err());
    let mut cached = value;
    cached["timings"]["cache_n"] = 1.into();
    assert!(super::parse_response_timing(&cached).is_err());
}

#[tokio::test]
async fn early_child_exit_is_reported_without_waiting_for_startup_deadline() {
    let directory = fixture();
    let flags = vec![
        "--tags".into(),
        directory.path().display().to_string(),
        "--reasoning-budget-message".into(),
        "exit".into(),
    ];
    let result = tokio::time::timeout(
        Duration::from_secs(2),
        launch_server(directory.path().join("server.py").to_str().unwrap(), &flags),
    )
    .await;
    assert!(result.unwrap().is_err());
    reaped(pid(&directory)).await;
}

#[test]
fn server_drop_reaps_after_its_tokio_runtime_is_gone() {
    for drop_before_runtime in [true, false] {
        let directory = fixture();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let server = runtime.block_on(start(&directory, "", &[]));
        let child_pid = pid(&directory);
        let log = server.log.path().to_path_buf();
        if drop_before_runtime {
            drop(server);
            drop(runtime);
        } else {
            drop(runtime);
            drop(server);
        }
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        while (process_exists(child_pid) || log.exists()) && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        // The test parent stays alive. No waitpid can hide a zombie here.
        assert!(
            !process_exists(child_pid),
            "owned child {child_pid} remains after runtime teardown"
        );
        assert!(
            !log.exists(),
            "the native owner must release the private log"
        );
    }
}

#[test]
fn policy_uses_native_long_option_aliases_without_changing_forwarded_tokens() {
    let flags = [
        "--api_key",
        "--gpu-layers=0",
        "--alias",
        "--api-key-file=not-a-file",
    ]
    .map(str::to_owned);
    let (forwarded, _, key) = super::launch_arguments(&flags).unwrap();
    assert_eq!(&forwarded[..flags.len()], flags);
    assert_eq!(key.as_deref(), Some("--gpu-layers=0"));
    let flags = ["--spec-replace", "--host=target", "--api-key=draft"].map(str::to_owned);
    let (forwarded, public, _) = super::launch_arguments(&flags).unwrap();
    assert_eq!(&forwarded[..flags.len()], flags);
    assert_eq!(&public[..flags.len()], flags);

    for value in [
        "--host=fixture-model-name",
        "--api-key=not-auth",
        "--log-disable",
    ] {
        let flags = vec!["--alias".into(), value.into()];
        let (forwarded, public, _) = super::launch_arguments(&flags).unwrap();
        assert_eq!(&forwarded[..flags.len()], flags);
        assert_eq!(&public[..flags.len()], flags);
    }
    for key in ["--api-key", "--api_key"] {
        for inline in [false, true] {
            let flags = if inline {
                vec![format!("{key}=private_key-value")]
            } else {
                vec![key.into(), "private_key-value".into()]
            };
            let (args, public, key) = super::launch_arguments(&flags).unwrap();
            assert_eq!(&args[..flags.len()], flags);
            assert_eq!(key.as_deref(), Some("private_key-value"));
            assert!(!public
                .iter()
                .any(|value| value.contains("private") || value.contains("api")));
        }
    }
    for key in [
        "--host",
        "--port",
        "--reuse-port",
        "--ssl-key-file",
        "--ssl-cert-file",
        "--api-prefix",
        "--log-disable",
        "--log-file",
        "-lv",
        "--verbosity",
        "--log-verbosity",
        "-v",
        "--verbose",
        "--log-verbose",
    ] {
        let aliases = if let Some(name) = key.strip_prefix("--") {
            vec![key.to_string(), format!("--{}", name.replace('-', "_"))]
        } else {
            vec![key.to_string()]
        };
        for alias in aliases {
            for flags in [
                vec![alias.clone(), "value".into()],
                vec![format!("{alias}=value")],
            ] {
                assert!(
                    super::launch_arguments(&flags).is_err(),
                    "{flags:?} must not bypass endpoint/log policy"
                );
            }
        }
    }
}

/// Run with AUTARKY_NATIVE_SERVER and AUTARKY_NATIVE_MODEL set to owned local assets.
/// Each case gets a new test process, so environment authentication never changes
/// the environment of parallel tests or reads the user's configuration.
#[cfg(target_os = "linux")]
#[tokio::test]
#[ignore = "requires an explicitly selected native llama-server and GGUF model"]
async fn native_csv_api_key_authentication() {
    const TEST: &str = "launcher::runtime_regressions::native_csv_api_key_authentication";
    let binary = std::env::var("AUTARKY_NATIVE_SERVER").expect("set AUTARKY_NATIVE_SERVER");
    let model = std::env::var("AUTARKY_NATIVE_MODEL").expect("set AUTARKY_NATIVE_MODEL");
    let Ok(case) = std::env::var("AUTARKY_NATIVE_AUTH_CASE") else {
        for case in ["cli", "environment", "append", "complex", "raw-file"] {
            let directory = tempfile::tempdir().unwrap();
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command
                .args(["--exact", TEST, "--ignored", "--nocapture"])
                .env_clear()
                .env("PATH", "/usr/bin:/bin")
                .env("HOME", directory.path())
                .env("XDG_CONFIG_HOME", directory.path())
                .env("XDG_DATA_HOME", directory.path())
                .env("TMPDIR", directory.path())
                .env("AUTARKY_NATIVE_SERVER", &binary)
                .env("AUTARKY_NATIVE_MODEL", &model)
                .env("AUTARKY_NATIVE_AUTH_CASE", case);
            if case == "environment" || case == "append" {
                command.env("LLAMA_API_KEY", "\"OWNFAKEKEY\"");
            }
            let output = command.output().unwrap();
            assert!(
                output.status.success(),
                "native auth case {case} failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        return;
    };
    let directory = tempfile::tempdir().unwrap();
    let alias = directory.path().display().to_string();
    let mut flags = vec![
        "--model".into(),
        model,
        "--alias".into(),
        alias.clone(),
        "--n-gpu-layers".into(),
        "0".into(),
        "--threads".into(),
        "2".into(),
        "--ctx-size".into(),
        "512".into(),
        "--parallel".into(),
        "1".into(),
        "--no-warmup".into(),
    ];
    let encoded = match case.as_str() {
        "cli" => Some("\"OWNFAKEKEY\""),
        "append" => Some(",,\"\""),
        "complex" => Some(",\"OWN,FAKE\"\"KEY\",other"),
        "environment" | "raw-file" => None,
        _ => panic!("unknown native auth test case"),
    };
    if let Some(encoded) = encoded {
        flags.extend(["--api-key".into(), encoded.into()]);
    }
    if case == "raw-file" {
        let path = directory.path().join("keys");
        std::fs::write(&path, "\n\"OWN,FAKEKEY\"\n").unwrap();
        flags.extend(["--api-key-file".into(), path.display().to_string()]);
    }
    let mut server = launch_server(&binary, &flags).await.unwrap();
    // Find only our child by its unique, explicitly forwarded alias.
    let child_pid = std::fs::read_dir("/proc")
        .unwrap()
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_name().to_str()?.parse::<u32>().ok())
        .find(|pid| {
            std::fs::read(format!("/proc/{pid}/cmdline")).is_ok_and(|cmdline| {
                cmdline
                    .split(|byte| *byte == 0)
                    .any(|arg| arg == alias.as_bytes())
            })
        })
        .expect("owned native server PID");
    server.wait_ready().await.unwrap();
    assert!(server
        .client
        .get(format!("{}/health", server.api_base()))
        .send()
        .await
        .unwrap()
        .status()
        .is_success());
    let body =
        serde_json::json!({"prompt":"Hello", "n_predict":2, "temperature":0, "stream":false});
    let response = server
        .client
        .post(format!("{}/completion", server.api_base()))
        .json(&body)
        .send()
        .await
        .unwrap();
    let status = response.status();
    let completion: serde_json::Value = response.json().await.unwrap();
    assert!(
        status.is_success(),
        "native authenticated completion failed: {status}"
    );
    assert!(completion["tokens_predicted"]
        .as_u64()
        .is_some_and(|n| n > 0));
    let unauthenticated = reqwest::Client::builder()
        .no_proxy()
        .build()
        .unwrap()
        .post(format!("{}/completion", server.api_base()))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(unauthenticated.status(), reqwest::StatusCode::UNAUTHORIZED);
    assert!(!server
        .full_flags
        .iter()
        .any(|token| token.contains("OWN") || token.contains("api-key")));
    server.kill().await.unwrap();
    reaped(child_pid).await;
    eprintln!("native auth {case}: health/completion 200, unauthenticated 401, PID reaped");
}

#[tokio::test]
async fn inline_key_and_key_file_authenticate_without_public_credentials() {
    let directory = fixture();
    let mut server = start(
        &directory,
        "",
        &[
            "--alias",
            "--host=fixture-model-name",
            "--api_key",
            "--gpu-layers=0",
        ],
    )
    .await;
    assert_eq!(
        server.benchmark(1, 1, 0, None).await.unwrap().status,
        "success"
    );
    server.kill().await.unwrap();
    reaped(pid(&directory)).await;

    for name in ["--api-key", "--api_key", "--api-key-file", "--api_key_file"] {
        for inline in [true, false] {
            let directory = fixture();
            let key_file = directory.path().join("private_key_file");
            std::fs::write(&key_file, "fixture-secret_from-file\n").unwrap();
            let value = if name.ends_with("file") {
                key_file.display().to_string()
            } else {
                "fixture-secret_from-file".into()
            };
            let flags = if inline {
                vec![format!("{name}={value}")]
            } else {
                vec![name.into(), value]
            };
            let (forwarded, public, key) = super::launch_arguments(&flags).unwrap();
            assert_eq!(
                &forwarded[..flags.len()],
                flags,
                "authentication must preserve native tokens and values"
            );
            assert_eq!(key.as_deref(), Some("fixture-secret_from-file"));
            assert!(!public.iter().any(
                |token| token.contains("private_key_file") || token.contains("fixture-secret")
            ));
            let refs = flags.iter().map(String::as_str).collect::<Vec<_>>();
            let mut server = start(&directory, "", &refs).await;
            assert!(server
                .full_flags
                .iter()
                .all(|flag| !flag.contains("fixture-secret") && !flag.contains("api-key")));
            assert_eq!(
                server.benchmark(1, 1, 0, None).await.unwrap().status,
                "success"
            );
            server.kill().await.unwrap();
            reaped(pid(&directory)).await;
        }
    }
}
