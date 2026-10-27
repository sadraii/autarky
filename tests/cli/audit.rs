use super::*;
use autarky::monitor::BenchmarkResult;
use serde_json::json;

fn seed(sandbox: &Sandbox) {
    success(sandbox.run(&["version"]));
    let conn = sandbox.db();
    conn.execute("INSERT INTO hardware_fingerprints(id,gpu_family,vram_gb,system_ram_gb,os_platform) VALUES(1,?1,16,32,'linux')", ["GPU模型\u{1b}[31m\n"]).unwrap();
    conn.execute("INSERT INTO model_fingerprints(id,model_name,quantization) VALUES(1,?1,'Q4'),(2,'Other','Q8')", ["模型👩‍💻abcdefghijklmnop\u{1b}[31m"]).unwrap();
    for (id, model, metrics) in [
        (
            1,
            Some(1),
            json!([
                BenchmarkResult::new(1, 50.0, 100.0),
                BenchmarkResult::new(2, 500.0, 1000.0)
            ]),
        ),
        (
            2,
            Some(1),
            json!([
                BenchmarkResult::new(1, 100.0, 200.0),
                BenchmarkResult::oom(2)
            ]),
        ),
        (3, None, json!([BenchmarkResult::new(1, 20.0, 40.0)])),
        (4, Some(2), json!([BenchmarkResult::new(1, 999.0, 1998.0)])),
    ] {
        conn.execute("INSERT INTO benchmark_runs(id,hardware_id,model_id,flags_used,parsed_flags,context_lengths_tested,raw_metrics) VALUES(?1,1,?2,'[]',NULL,'[1,2]',?3)", rusqlite::params![id,model,metrics.to_string()]).unwrap();
    }
}

#[test]
fn history_handles_null_models_unicode_and_success_only_summaries() {
    let sandbox = Sandbox::new();
    seed(&sandbox);
    let listing = success(sandbox.run(&["list"]));
    assert!(listing.contains("unknown"));
    assert!(!listing.contains('\u{1b}'));
    let exported: Value = serde_json::from_str(&success(sandbox.run(&["export"]))).unwrap();
    let rows = exported.as_array().unwrap();
    let row = rows.iter().find(|row| row["id"] == 2).unwrap();
    assert_eq!(row["avg_tps"], 100.0);
    assert_eq!(row["avg_prompt"], 200.0);
    assert_eq!(row["max_context"], 1);
    assert!(rows.iter().find(|row| row["id"] == 3).unwrap()["model"].is_null());
    let board = success(sandbox.run(&["leaderboard", "--period", "all", "--sort", "context"]));
    assert!(!board.contains('\u{1b}'));
    assert!(board.contains("100.0"));
    assert!(board.contains("1k"));
    let csv = success(sandbox.run(&["export", "--format", "csv"]));
    assert_eq!(
        csv::Reader::from_reader(csv.as_bytes()).records().count(),
        4
    );
    let recommendations: Value = serde_json::from_str(&success(sandbox.run(&[
        "recommend",
        "--gpu",
        "GPU模型\u{1b}[31m\n",
        "--format",
        "json",
    ])))
    .unwrap();
    assert_eq!(recommendations.as_array().unwrap().len(), 4);
}

#[test]
fn compare_binds_all_filters_and_uses_only_matched_successful_contexts() {
    let sandbox = Sandbox::new();
    seed(&sandbox);
    for args in [
        vec!["compare", "--model", "模型"],
        vec!["compare", "--gpu", "GPU", "--model", "模型"],
        vec!["compare", "--id1", "2", "--id2", "1"],
        vec!["compare", "--id1", "2"],
        vec!["compare", "--similar", "--id1", "2"],
    ] {
        let output = success(sandbox.run(&args));
        assert!(
            output.contains("Matched successful contexts (K): [1]"),
            "{output}"
        );
        assert!(output.contains("100.0 / 50.0"), "{output}");
        assert!(output.contains("+100.0%"), "{output}");
        assert!(!output.contains('\u{1b}'));
    }
    let nullable = success(sandbox.run(&["compare", "--id1", "3", "--id2", "1"]));
    assert!(nullable.contains("unknown workloads"));
    assert!(!sandbox
        .run(&["compare", "--similar", "--id1", "4"])
        .status
        .success());
}

#[test]
fn invalid_dates_fail_before_initialization_and_never_overwrite_exports() {
    let sandbox = Sandbox::new();
    let target = sandbox.root.path().join("export.json");
    std::fs::write(&target, "preserve").unwrap();
    for range in [
        "bad",
        "2026-01-01",
        "2026-01-01,2026-02-30",
        "2026-02-01,2026-01-01",
        "2026-01-01,2026-02-01,2026-03-01",
        "2025-02-29,2026-01-01",
        "2026-1-01,2026-02-01",
    ] {
        for command in ["list", "export"] {
            let mut args = vec![command, "--date-range", range];
            if command == "export" {
                args.extend(["--file", target.to_str().unwrap()]);
            }
            assert!(!sandbox.run(&args).status.success(), "{range}");
            assert_eq!(std::fs::read_to_string(&target).unwrap(), "preserve");
            assert!(!sandbox.config_path().exists());
        }
    }
    assert_eq!(
        serde_json::from_str::<Value>(&success(sandbox.run(&[
            "export",
            "--date-range",
            " 2024-02-29 , 2026-10-25 "
        ])))
        .unwrap(),
        json!([])
    );
}

#[test]
fn empty_machine_outputs_replace_destinations_and_remain_machine_readable() {
    let sandbox = Sandbox::new();
    for format in ["json", "csv"] {
        let target = sandbox.root.path().join(format!("export.{format}"));
        std::fs::write(&target, "stale").unwrap();
        let output = success(sandbox.run(&[
            "export",
            "--format",
            format,
            "--file",
            target.to_str().unwrap(),
        ]));
        assert!(output.trim().is_empty());
        let data = std::fs::read_to_string(target).unwrap();
        if format == "json" {
            assert_eq!(serde_json::from_str::<Value>(&data).unwrap(), json!([]));
        } else {
            assert!(data.starts_with("id,timestamp,model,"));
            assert_eq!(data.lines().count(), 1);
        }
    }
    assert_eq!(
        serde_json::from_str::<Value>(&success(sandbox.run(&[
            "recommend",
            "--gpu",
            "unknown",
            "--format",
            "json"
        ])))
        .unwrap(),
        json!([])
    );
}

fn community_entry(id: i64, tps: f64) -> Value {
    json!({"id":id, "submitted_at":"2026-10-25", "model_name":"模型👩‍💻abcdefghijk", "quantization":"Q4", "gpu_family":"GPU模型\u{1b}[31m", "vram_gb":16.0, "system_ram_gb":32, "os_platform":"linux", "raw_metrics":[BenchmarkResult::new(1,tps,tps*2.0), BenchmarkResult::oom(2)]})
}

pub(super) fn community_fixture(
    status: &str,
    headers: &str,
    body: &str,
) -> (String, std::thread::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!("http://{}/prefix/", listener.local_addr().unwrap());
    let response = format!(
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n{headers}\r\n{body}",
        body.len()
    );
    let worker = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(5);
        let (mut stream, _) = loop {
            match listener.accept() {
                Ok(value) => break value,
                Err(error)
                    if error.kind() == std::io::ErrorKind::WouldBlock
                        && Instant::now() < deadline =>
                {
                    std::thread::sleep(Duration::from_millis(5))
                }
                Err(error) => panic!("Fixture accept failed: {error}"),
            }
        };
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut request = Vec::new();
        loop {
            let mut chunk = [0; 4096];
            let size = stream.read(&mut chunk).unwrap();
            assert!(size > 0);
            request.extend_from_slice(&chunk[..size]);
            if let Some(end) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&request[..end]).to_ascii_lowercase();
                let length = headers
                    .lines()
                    .find_map(|line| line.strip_prefix("content-length: "))
                    .map_or(0, |len| len.parse::<usize>().unwrap());
                if request.len() >= end + 4 + length {
                    break;
                }
            }
        }
        stream.write_all(response.as_bytes()).unwrap();
        String::from_utf8(request).unwrap()
    });
    (url, worker)
}

#[test]
fn search_sends_real_flag_filters_derives_metrics_and_sanitizes_unicode() {
    let sandbox = Sandbox::new();
    let (url, server) = community_fixture(
        "200 OK",
        "",
        &json!({"count":1,"results":[community_entry(7,42.0)]}).to_string(),
    );
    sandbox.set_config(&format!("server_url = '{url}'\n"));
    let output = success(sandbox.run(&["search", "Model", "--flags=--threads 8"]));
    let request = server.join().unwrap();
    assert!(request.starts_with("GET /prefix/api/v1/benchmarks?"));
    let first = request
        .lines()
        .next()
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap();
    let query = reqwest::Url::parse(&format!("http://localhost{first}")).unwrap();
    assert!(query
        .query_pairs()
        .any(|(key, value)| key == "flags" && value == "--threads 8"));
    assert!(output.contains("42.0"));
    assert!(output.contains("84.0"));
    assert!(!output.contains('\u{1b}'));
}

#[test]
fn community_status_json_schema_and_redirect_errors_are_nonzero_not_empty_success() {
    for (status, body) in [
        ("500 Internal Server Error", "private-error-body"),
        ("200 OK", "<html>bad</html>"),
        ("200 OK", "{}"),
        ("200 OK", "{\"count\":1,\"results\":[]}"),
        ("200 OK", "{\"count\":1,\"results\":[{}]}"),
    ] {
        let sandbox = Sandbox::new();
        let (url, server) = community_fixture(status, "", body);
        sandbox.set_config(&format!("server_url = '{url}'\n"));
        let output = sandbox.run(&["search"]);
        server.join().unwrap();
        assert!(!output.status.success(), "{body}");
        assert!(!String::from_utf8_lossy(&output.stdout).contains("Found 0"));
        assert!(!String::from_utf8_lossy(&output.stderr).contains(body));
    }
    let destination = TcpListener::bind("127.0.0.1:0").unwrap();
    destination.set_nonblocking(true).unwrap();
    let sandbox = Sandbox::new();
    let headers = format!(
        "Location: http://{}/leak\r\n",
        destination.local_addr().unwrap()
    );
    let (url, server) = community_fixture("302 Found", &headers, "");
    sandbox.set_config(&format!("server_url = '{url}'\n"));
    assert!(!sandbox.run(&["search"]).status.success());
    server.join().unwrap();
    assert!(destination.accept().is_err());
}

#[test]
fn remote_comparison_normalizes_prefix_and_ignores_inconsistent_server_deltas() {
    let sandbox = Sandbox::new();
    let body = json!({"benchmark_1":community_entry(1,100.0),"benchmark_2":community_entry(2,50.0),"delta":{"tps_percent":"9999%"}});
    let (url, server) = community_fixture("200 OK", "", &body.to_string());
    sandbox.set_config(&format!("server_url = '{url}'\n"));
    let output = success(sandbox.run(&["compare", "--remote", "--id1", "1", "--id2", "2"]));
    assert!(server
        .join()
        .unwrap()
        .starts_with("POST /prefix/api/v1/compare HTTP/1.1"));
    assert!(output.contains("100.0 / 50.0"));
    assert!(output.contains("+100.0%"));
    assert!(!output.contains("9999"));
    assert!(!output.contains('\u{1b}'));
}

#[test]
fn remote_comparison_rejects_http_schema_and_identity_failures() {
    for (status, body) in [
        ("500 Internal Server Error", "private response".to_string()),
        ("200 OK", "not JSON".to_string()),
        ("200 OK", "{}".to_string()),
        (
            "200 OK",
            json!({"benchmark_1":community_entry(9,10.0),"benchmark_2":community_entry(2,20.0)})
                .to_string(),
        ),
    ] {
        let sandbox = Sandbox::new();
        let (url, server) = community_fixture(status, "", &body);
        sandbox.set_config(&format!("server_url = '{url}'\n"));
        let output = sandbox.run(&["compare", "--remote", "--id1", "1", "--id2", "2"]);
        server.join().unwrap();
        assert!(!output.status.success());
        assert!(!String::from_utf8_lossy(&output.stdout).contains("Avg Tokens/s"));
        assert!(!String::from_utf8_lossy(&output.stderr).contains("private response"));
    }
}

#[test]
fn search_connection_failure_is_nonzero_and_valid_empty_response_is_success() {
    let sandbox = Sandbox::new();
    let (url, server) = community_fixture("200 OK", "", r#"{"count":0,"results":[]}"#);
    sandbox.set_config(&format!("server_url = '{url}'\n"));
    assert!(success(sandbox.run(&["search"])).contains("Found 0"));
    server.join().unwrap();
    assert!(!sandbox.run(&["search"]).status.success());
}

#[test]
fn offline_blocks_all_community_commands_without_connecting() {
    let sandbox = Sandbox::new();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    sandbox.set_config(&format!(
        "server_url = 'http://{}'\n",
        listener.local_addr().unwrap()
    ));
    for args in [
        vec!["--offline", "search"],
        vec![
            "--offline",
            "compare",
            "--remote",
            "--id1",
            "1",
            "--id2",
            "2",
        ],
    ] {
        assert!(!sandbox.run(&args).status.success());
    }
    #[cfg(unix)]
    {
        let fixture = sandbox.fixture();
        success(sandbox.run(&[
            "--offline",
            "bench",
            "--context-lengths",
            "1",
            "--repeats",
            "1",
            "--passthrough",
            fixture.to_str().unwrap(),
            "--gpu-layers",
            "0",
        ]));
    }
    assert!(listener.accept().is_err());
}

#[cfg(unix)]
#[test]
fn tui_quit_and_ctrl_c_interrupt_active_work_restore_terminal_and_save_cancellation() {
    const PROBE: &str = r#"
import os, pty, subprocess, sys, time, termios, fcntl, struct, select
master, slave = pty.openpty()
fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 40, 120, 0, 0))
before = termios.tcgetattr(slave)
env = dict(os.environ, TERM='xterm', AUTARKY_FIXTURE_MODE=sys.argv[4])
child = subprocess.Popen([sys.argv[1], '--offline', 'bench', '--tui', '--context-lengths','1','--repeats','1','--passthrough',sys.argv[2],'--gpu-layers','0'], stdin=slave, stdout=slave, stderr=slave, env=env, start_new_session=True)
root = env['AUTARKY_FIXTURE_ROOT']
output = bytearray()
deadline = time.monotonic()+10
sent=False
while child.poll() is None and time.monotonic()<deadline:
    ready,_,_=select.select([master],[],[],0.02)
    if ready: output.extend(os.read(master,65536))
    witness = '/server-pid' if sys.argv[4] == 'startup' else '/requests.jsonl'
    if not sent and os.path.exists(root+witness):
        os.write(master, bytes.fromhex(sys.argv[3])); sent=True
if child.poll() is None:
    child.kill(); child.wait(); raise AssertionError('TUI did not stop active work')
while select.select([master],[],[],0)[0]: output.extend(os.read(master,65536))
assert sent and child.returncode != 0
assert termios.tcgetattr(slave) == before, 'terminal modes were not restored'
assert b'Saved benchmark #' in output, output
assert output.index(b'\x1b[?1049l') < output.index(b'Saved benchmark #'), 'report printed before restoration'
os.close(master); os.close(slave)
print('terminal restored; cancellation saved; child reaped')
"#;
    for (key, mode) in [
        ("71", "cancel"),
        ("03", "cancel"),
        ("71", "startup"),
        ("03", "startup"),
    ] {
        let sandbox = Sandbox::new();
        let fixture = sandbox.fixture();
        let base = sandbox.command(&[]);
        let envs = base
            .get_envs()
            .filter_map(|(key, value)| value.map(|value| (key, value)));
        let output = bounded_output(
            Command::new("python3")
                .arg("-c")
                .arg(PROBE)
                .arg(env!("CARGO_BIN_EXE_autarky"))
                .arg(fixture)
                .arg(key)
                .arg(mode)
                .envs(envs)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
        success(output);
        super::durability::assert_child_gone(&sandbox);
        assert_eq!(sandbox.measurements()[0][0]["status"], "cancelled");
    }
}
