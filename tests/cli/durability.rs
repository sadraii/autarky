use super::*;

fn wait_for(path: &std::path::Path) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !path.exists() {
        assert!(Instant::now() < deadline, "No fixture witness: {path:?}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

pub(super) fn assert_child_gone(sandbox: &Sandbox) {
    let pid = std::fs::read_to_string(sandbox.root.path().join("server-pid"))
        .unwrap()
        .trim()
        .parse::<u32>()
        .unwrap();
    assert!(!process_exists(pid), "Fixture child {pid} was not reaped");
}

fn gguf(
    sandbox: &Sandbox,
    label: &str,
    name: Option<&str>,
    quant: Option<u32>,
    parameters: u64,
) -> PathBuf {
    let mut entries = vec![(
        "general.parameter_count",
        10u32,
        parameters.to_le_bytes().to_vec(),
    )];
    if let Some(name) = name {
        let mut value = (name.len() as u64).to_le_bytes().to_vec();
        value.extend(name.as_bytes());
        entries.push(("general.name", 8, value));
    }
    if let Some(quant) = quant {
        entries.push(("general.file_type", 4, quant.to_le_bytes().to_vec()));
    }
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    bytes.extend((entries.len() as u64).to_le_bytes());
    for (key, kind, value) in entries {
        bytes.extend((key.len() as u64).to_le_bytes());
        bytes.extend(key.as_bytes());
        bytes.extend(kind.to_le_bytes());
        bytes.extend(value);
    }
    let path = sandbox.root.path().join(format!("{label}-Q8_0.gguf"));
    std::fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn completed_measurement_survives_broken_stdout_without_panic_and_reaps_child() {
    let sandbox = Sandbox::new();
    let fixture = sandbox.fixture();
    let mut child = sandbox
        .command(&[
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
        ])
        .env("AUTARKY_FIXTURE_MODE", "gated_success")
        .spawn()
        .unwrap();
    wait_for(&sandbox.root.path().join("requests.jsonl"));
    // Close the consumer before the measured response completes. The next output
    // is the completed result, which must enter durable storage before returning.
    drop(child.stdout.take());
    std::fs::write(sandbox.root.path().join("release-response"), "go").unwrap();
    let output = bounded_output(child);
    assert_eq!(
        output.status.code(),
        Some(1),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!String::from_utf8_lossy(&output.stderr).contains("panicked"));
    let records = sandbox.measurements();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0][0]["status"], "success");
    assert_eq!(records[0][0]["tokens_per_second"], 50.0);
    let exported: Value = serde_json::from_str(&success(sandbox.run(&["export"]))).unwrap();
    assert_eq!(exported[0]["status"], "failed");
    assert!(exported[0]["failure_reason"]
        .as_str()
        .unwrap()
        .contains("pipe"));
    assert_child_gone(&sandbox);
}

#[test]
fn startup_output_and_tui_setup_failures_use_the_same_durable_failure_path() {
    for tui in [false, true] {
        let sandbox = Sandbox::new();
        let fixture = sandbox.fixture();
        let mut args = vec![
            "--offline",
            "bench",
            "--context-lengths",
            "1",
            "--repeats",
            "1",
        ];
        if tui {
            args.push("--tui");
        }
        args.extend([
            "--passthrough",
            fixture.to_str().unwrap(),
            "--gpu-layers",
            "0",
        ]);
        let mut child = sandbox.command(&args).spawn().unwrap();
        drop(child.stdout.take());
        let output = bounded_output(child);
        assert_eq!(output.status.code(), Some(1));
        assert!(!String::from_utf8_lossy(&output.stderr).contains("panicked"));
        assert_eq!(sandbox.measurements().len(), 1);
        assert_eq!(sandbox.measurements()[0][0]["status"], "error");
        assert!(!sandbox.root.path().join("server-pid").exists());
    }
}

#[test]
fn uploads_preserve_nullable_identity_and_never_compare_missing_or_unsupported_quantization() {
    for (name, quant) in [
        (None, Some(15)),
        (Some("unknown"), None),
        (Some("unknown"), Some(9999)),
        (Some("unknown"), Some(15)),
    ] {
        let sandbox = Sandbox::new();
        let fixture = sandbox.fixture();
        let path = gguf(&sandbox, "partial", name, quant, 123_000_000);
        let (url, server) = audit::community_fixture("200 OK", "", r#"{"id":7}"#);
        sandbox.set_config(&format!("server_url = '{url}'\n"));
        let output = success(sandbox.run(&[
            "bench",
            "--context-lengths",
            "1",
            "--repeats",
            "1",
            "--passthrough",
            fixture.to_str().unwrap(),
            "--model",
            path.to_str().unwrap(),
            "--gpu-layers",
            "0",
        ]));
        let request = server.join().unwrap();
        let payload: Value =
            serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
        if let Some(name) = name {
            assert_eq!(payload["model"]["model_name"], name);
            assert_eq!(
                payload["model"]["quantization"],
                if quant == Some(15) {
                    Value::String("Q4_K_M".into())
                } else {
                    Value::Null
                }
            );
            assert_eq!(payload["model"]["parameters_billions"], 0.123);
        } else {
            assert!(payload["model"].is_null());
        }
        assert_eq!(
            output.contains("Checking community"),
            name.is_some() && quant == Some(15)
        );
        let snapshot: String = sandbox
            .db()
            .query_row("SELECT snapshot_json FROM model_fingerprints", [], |r| {
                r.get(0)
            })
            .unwrap();
        let snapshot: Value = serde_json::from_str(&snapshot).unwrap();
        assert_eq!(
            snapshot["file_type"],
            quant.map_or(Value::Null, Value::from)
        );
        assert_eq!(snapshot["parameters_billions"], 0.123);
    }
}

#[test]
fn incomplete_identity_never_becomes_a_matched_workload_but_literal_unknown_is_known() {
    for (name, quant) in [
        (None, None),
        (None, Some(15)),
        (Some("Named"), None),
        (Some("unknown"), Some(15)),
    ] {
        let sandbox = Sandbox::new();
        let fixture = sandbox.fixture();
        for n in [1, 2] {
            let parameters = if name.is_some() && quant.is_some() {
                1_000_000
            } else {
                n * 1_000_000
            };
            let path = gguf(&sandbox, &format!("model-{n}"), name, quant, parameters);
            success(sandbox.run(&[
                "--offline",
                "bench",
                "--context-lengths",
                "1",
                "--repeats",
                "1",
                "--passthrough",
                fixture.to_str().unwrap(),
                "--model",
                path.to_str().unwrap(),
                "--gpu-layers",
                "0",
            ]));
        }
        let output = success(sandbox.run(&["compare", "--id1", "2", "--id2", "1"]));
        let similar = sandbox.run(&["compare", "--similar", "--id1", "2"]);
        if name.is_some() && quant.is_some() {
            assert!(output.contains("+0.0%"), "{output}");
            assert!(success(similar).contains("+0.0%"));
        } else {
            assert!(output.contains("unknown workloads"), "{output}");
            assert!(!output.contains('%'));
            assert!(!similar.status.success());
            let rows: Value = serde_json::from_str(&success(sandbox.run(&["export"]))).unwrap();
            assert_eq!(
                rows[0]["model"],
                name.map_or(Value::Null, |n| Value::String(n.into()))
            );
            assert_eq!(rows[0]["quantization"].is_null(), quant.is_none());
            success(sandbox.run(&["list"]));
            let gpu: String = sandbox
                .db()
                .query_row(
                    "SELECT gpu_family FROM hardware_fingerprints LIMIT 1",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            let recommendations: Value = serde_json::from_str(&success(sandbox.run(&[
                "recommend",
                "--gpu",
                &gpu,
                "--format",
                "json",
            ])))
            .unwrap();
            assert_eq!(recommendations.as_array().unwrap().len(), 2);
            assert_eq!(
                recommendations[0]["model_name"],
                name.map_or(Value::Null, Value::from)
            );
        }
    }
}

#[test]
fn sigint_during_stalled_upload_exits_promptly_keeps_local_record_and_sends_no_followup() {
    sigint_during_stalled_community_call(false);
}

#[test]
fn sigint_during_stalled_comparison_keeps_local_record_and_sends_no_followup() {
    sigint_during_stalled_community_call(true);
}

#[test]
fn remote_comparison_uses_presence_not_placeholder_text_for_identity() {
    for (name, quant) in [
        (None, Some("Q4_K_M")),
        (Some("unknown"), None),
        (Some("unknown"), Some("Q4_K_M")),
    ] {
        let sandbox = Sandbox::new();
        let entry = |id| serde_json::json!({"id":id,"submitted_at":"2026-10-25","model_name":name,"quantization":quant,"gpu_family":"Fixture CPU","vram_gb":0.0,"system_ram_gb":8,"os_platform":"linux","raw_metrics":[autarky::monitor::BenchmarkResult::new(1,50.0,900.0)]});
        let body = serde_json::json!({"benchmark_1":entry(1),"benchmark_2":entry(2)});
        let (url, server) = audit::community_fixture("200 OK", "", &body.to_string());
        sandbox.set_config(&format!("server_url = '{url}'\n"));
        let output = success(sandbox.run(&["compare", "--remote", "--id1", "1", "--id2", "2"]));
        server.join().unwrap();
        assert_eq!(output.contains("+0.0%"), name.is_some() && quant.is_some());
        if name.is_none() || quant.is_none() {
            assert!(!output.contains('%'));
        }
    }
}

#[test]
fn sigint_during_community_prompt_restores_tui_and_exits_without_waiting_for_input() {
    const PROBE: &str = r#"
import http.server, json, os, pathlib, pty, select, signal, struct, subprocess, sys, termios, threading, time, fcntl
root = pathlib.Path(os.environ['AUTARKY_FIXTURE_ROOT'])
payload = None
requests = []
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def reply(self, value):
        body = json.dumps(value).encode()
        self.send_response(200); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        global payload
        requests.append('POST')
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.reply({'id':7})
    def do_GET(self):
        requests.append('GET')
        entry = dict(id=9, submitted_at='2026-10-25', model_name=payload['model']['model_name'], quantization=payload['model']['quantization'], **payload['hardware'])
        entry['raw_metrics'] = [dict(context_length=1,tokens_per_second=100.0,prompt_per_second=1800.0,status='success',error=None)]
        self.reply({'count':1,'results':[entry]})
server = http.server.HTTPServer(('127.0.0.1',0),Handler)
thread = threading.Thread(target=server.serve_forever); thread.start()
config = pathlib.Path(os.environ['AUTARKY_FIXTURE_CONFIG']); config.parent.mkdir(parents=True,exist_ok=True)
config.write_text("server_url = 'http://127.0.0.1:%d'\n" % server.server_port)
master, slave = pty.openpty(); fcntl.ioctl(slave,termios.TIOCSWINSZ,struct.pack('HHHH',40,120,0,0))
before = termios.tcgetattr(slave)
child = subprocess.Popen([sys.argv[1],'bench','--tui','--context-lengths','1','--repeats','1','--passthrough',sys.argv[2],'--model',sys.argv[3],'--gpu-layers','0'],stdin=slave,stdout=slave,stderr=slave,env=dict(os.environ,TERM='xterm'),start_new_session=True)
output = bytearray(); deadline = time.monotonic()+10
try:
    while b'Want to see it?' not in output and child.poll() is None and time.monotonic()<deadline:
        if select.select([master],[],[],0.02)[0]: output.extend(os.read(master,65536))
    assert b'Want to see it?' in output, output
    assert termios.tcgetattr(slave)==before, 'TUI not restored before community prompt'
    assert output.index(b'\x1b[?1049l') < output.index(b'Saved benchmark #')
    child.send_signal(signal.SIGINT)
    assert child.wait(timeout=3)==1
    assert requests == ['POST','GET'], requests
finally:
    if child.poll() is None: child.kill(); child.wait()
    server.shutdown(); thread.join(); server.server_close(); os.close(master); os.close(slave)
print('prompt cancelled; terminal restored; one upload and one comparison')
"#;
    let sandbox = Sandbox::new();
    let fixture = sandbox.fixture();
    let model = gguf(&sandbox, "known", Some("Known Model"), Some(15), 1_000_000);
    let base = sandbox.command(&[]);
    let envs = base
        .get_envs()
        .filter_map(|(key, value)| value.map(|value| (key, value)));
    success(bounded_output(
        Command::new("python3")
            .arg("-c")
            .arg(PROBE)
            .arg(env!("CARGO_BIN_EXE_autarky"))
            .arg(fixture)
            .arg(model)
            .envs(envs)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    ));
    assert_eq!(sandbox.measurements().len(), 1);
    assert_eq!(sandbox.measurements()[0][0]["status"], "success");
    assert_child_gone(&sandbox);
}

#[test]
fn remote_comparison_never_ignores_known_descriptor_contradictions() {
    for (field, left, right) in [
        ("parameters_billions", Value::from(1.0), Value::from(2.0)),
        ("expert_count", Value::from(0), Value::from(8)),
        ("architecture", Value::from("llama"), Value::from("mistral")),
    ] {
        let sandbox = Sandbox::new();
        let entry = |id, descriptor| {
            let mut entry = serde_json::json!({"id":id,"submitted_at":"2026-10-25","model_name":"unknown","quantization":"Q4_K_M","gpu_family":"Fixture CPU","vram_gb":0.0,"system_ram_gb":8,"os_platform":"linux","raw_metrics":[autarky::monitor::BenchmarkResult::new(1,50.0,900.0)]});
            entry[field] = descriptor;
            entry
        };
        let body = serde_json::json!({"benchmark_1":entry(1,left),"benchmark_2":entry(2,right)});
        let (url, server) = audit::community_fixture("200 OK", "", &body.to_string());
        sandbox.set_config(&format!("server_url = '{url}'\n"));
        let output = success(sandbox.run(&["compare", "--remote", "--id1", "1", "--id2", "2"]));
        server.join().unwrap();
        assert!(!output.contains('%'), "{field}: {output}");
    }
}

fn add_metadata(path: &std::path::Path, key: &str, kind: u32, value: Vec<u8>) {
    let mut bytes = std::fs::read(path).unwrap();
    let count = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    bytes[16..24].copy_from_slice(&(count + 1).to_le_bytes());
    bytes.extend((key.len() as u64).to_le_bytes());
    bytes.extend(key.as_bytes());
    bytes.extend(kind.to_le_bytes());
    bytes.extend(value);
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn matching_names_and_quants_never_override_known_metadata_contradictions() {
    for descriptor in [
        "parameters",
        "expert_count",
        "architecture",
        "block_count",
        "context_length",
        "absent-experts",
    ] {
        let sandbox = Sandbox::new();
        let fixture = sandbox.fixture();
        for n in [1u32, 2] {
            let parameters = if descriptor == "parameters" {
                u64::from(n) * 1_000_000
            } else {
                1_000_000
            };
            let path = gguf(
                &sandbox,
                &format!("variant-{n}"),
                Some("Same Name"),
                Some(15),
                parameters,
            );
            match descriptor {
                "architecture" => {
                    let value = if n == 1 { "llama" } else { "mistral" };
                    let mut bytes = (value.len() as u64).to_le_bytes().to_vec();
                    bytes.extend(value.as_bytes());
                    add_metadata(&path, "general.architecture", 8, bytes);
                }
                "expert_count" | "block_count" | "context_length" => add_metadata(
                    &path,
                    &format!("llama.{descriptor}"),
                    4,
                    (n * 8192).to_le_bytes().to_vec(),
                ),
                "absent-experts" if n == 2 => {
                    add_metadata(&path, "llama.expert_count", 4, 0u32.to_le_bytes().to_vec())
                }
                _ => {}
            }
            let output = success(sandbox.run(&[
                "--offline",
                "bench",
                "--context-lengths",
                "1",
                "--repeats",
                "1",
                "--passthrough",
                fixture.to_str().unwrap(),
                "--model",
                path.to_str().unwrap(),
                "--gpu-layers",
                "0",
            ]));
            if n == 1
                && matches!(
                    descriptor,
                    "parameters" | "block_count" | "context_length" | "absent-experts"
                )
            {
                assert!(
                    !output.contains("(dense)"),
                    "Unknown expert/architecture must not be labelled dense"
                );
            }
        }
        let direct = success(sandbox.run(&["compare", "--id1", "2", "--id2", "1"]));
        let similar = sandbox.run(&["compare", "--id1", "2", "--similar"]);
        if descriptor == "absent-experts" {
            assert!(direct.contains("+0.0%"));
            assert!(success(similar).contains("+0.0%"));
        } else {
            assert!(!direct.contains('%'), "{descriptor}: {direct}");
            assert!(!similar.status.success(), "{descriptor}");
        }
    }
}

fn sigint_during_stalled_community_call(comparison: bool) {
    let sandbox = Sandbox::new();
    let fixture = sandbox.fixture();
    let model = gguf(&sandbox, "known", Some("Known Model"), Some(15), 1_000_000);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    sandbox.set_config(&format!(
        "server_url = 'http://{}'\napi_key = 'fixture-key'\n",
        listener.local_addr().unwrap()
    ));
    let mut child = sandbox
        .command(&[
            "bench",
            "--context-lengths",
            "1",
            "--repeats",
            "1",
            "--passthrough",
            fixture.to_str().unwrap(),
            "--model",
            model.to_str().unwrap(),
            "--gpu-layers",
            "0",
        ])
        .spawn()
        .unwrap();
    let mut stalled = Vec::new();
    for request_index in 0..=usize::from(comparison) {
        let deadline = Instant::now() + Duration::from_secs(10);
        let (mut stream, _) = loop {
            match listener.accept() {
                Ok(stream) => break stream,
                Err(e)
                    if e.kind() == std::io::ErrorKind::WouldBlock && Instant::now() < deadline =>
                {
                    std::thread::sleep(Duration::from_millis(5))
                }
                Err(e) => {
                    let _ = child.kill();
                    panic!("No upload: {e}");
                }
            }
        };
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let mut request = Vec::new();
        loop {
            let mut bytes = [0; 4096];
            let n = stream.read(&mut bytes).unwrap();
            assert!(n > 0);
            request.extend(&bytes[..n]);
            if let Some(end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                let headers = String::from_utf8_lossy(&request[..end]).to_ascii_lowercase();
                let length: usize = headers
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length: "))
                    .map_or(0, |length| length.parse().unwrap());
                if request.len() >= end + 4 + length {
                    break;
                }
            }
        }
        if comparison && request_index == 0 {
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\nConnection: close\r\n\r\n{\"id\":7}",
                )
                .unwrap();
        } else {
            stalled.push(stream);
        }
    }
    assert_eq!(stalled.len(), 1);
    assert_eq!(sandbox.measurements()[0][0]["status"], "success");
    assert_child_gone(&sandbox);
    assert!(Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .unwrap()
        .success());
    let deadline = Instant::now() + Duration::from_secs(3);
    while child.try_wait().unwrap().is_none() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    if child.try_wait().unwrap().is_none() {
        child.kill().unwrap();
        child.wait().unwrap();
        panic!("Ctrl+C was ignored during a saved-result community call");
    }
    let output = child.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    let text = String::from_utf8(output.stdout).unwrap();
    assert_eq!(text.contains("ID #7"), comparison);
    assert_eq!(text.contains("Checking community"), comparison);
    assert!(!text.contains("none found"));
    assert!(
        listener.accept().is_err(),
        "Cancellation must not start another community call"
    );
    assert_eq!(sandbox.measurements().len(), 1);
}
