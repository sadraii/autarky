//! Public CLI boundaries must not use the real HOME, config, DB, or a live service.
// Windows Known Folders do not follow HOME/XDG overrides. Do not run these
// fixtures there until their native directories can be isolated safely.
#![cfg(unix)]

#[path = "support/process.rs"]
mod process;
use process::process_exists;
#[path = "cli/audit.rs"]
mod audit;
#[cfg(unix)]
#[path = "cli/durability.rs"]
mod durability;

use rusqlite::Connection;
use serde_json::Value;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

struct Sandbox {
    root: tempfile::TempDir,
}

impl Sandbox {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        for name in ["home", "config", "data", "tmp"] {
            std::fs::create_dir(root.path().join(name)).unwrap();
        }
        // Publish only a link to immutable source, never a writable executable
        // that another concurrently forked fixture can inherit.
        let source =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cli-server.py");
        std::os::unix::fs::symlink(source, root.path().join("fixture-server")).unwrap();
        Self { root }
    }

    fn command(&self, args: &[&str]) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_autarky"));
        command
            .args(args)
            .current_dir(self.root.path())
            .env("HOME", self.root.path().join("home"))
            .env("XDG_CONFIG_HOME", self.root.path().join("config"))
            .env("XDG_DATA_HOME", self.root.path().join("data"))
            .env("TMPDIR", self.root.path().join("tmp"))
            .env_remove("LLAMA_ARG_DEVICE")
            .env_remove("LLAMA_ARG_N_GPU_LAYERS")
            .env_remove("LLAMA_API_KEY")
            .env("AUTARKY_FIXTURE_ROOT", self.root.path())
            .env("AUTARKY_FIXTURE_CONFIG", self.config_path())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        command
    }

    fn run(&self, args: &[&str]) -> Output {
        bounded_output(self.command(args).spawn().unwrap())
    }

    fn config_root(&self) -> PathBuf {
        if cfg!(target_os = "macos") {
            self.root.path().join("home/Library/Application Support")
        } else {
            self.root.path().join("config")
        }
    }

    fn data_root(&self) -> PathBuf {
        if cfg!(target_os = "macos") {
            self.config_root()
        } else {
            self.root.path().join("data")
        }
    }

    fn config_path(&self) -> PathBuf {
        self.config_root().join("autarky/config.toml")
    }

    fn set_config(&self, text: &str) {
        std::fs::create_dir_all(self.config_path().parent().unwrap()).unwrap();
        std::fs::write(self.config_path(), text).unwrap();
    }

    fn db(&self) -> Connection {
        Connection::open(self.data_root().join("autarky/benchmarks.db")).unwrap()
    }

    fn measurements(&self) -> Vec<Value> {
        let conn = self.db();
        let mut stmt = conn
            .prepare("SELECT raw_metrics FROM benchmark_runs ORDER BY id")
            .unwrap();
        stmt.query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .map(|row| serde_json::from_str(&row.unwrap()).unwrap())
            .collect()
    }

    #[cfg(unix)]
    fn fixture(&self) -> PathBuf {
        self.root.path().join("fixture-server")
    }
}

fn bounded_output(mut child: Child) -> Output {
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if child.try_wait().unwrap().is_some() {
            return child.wait_with_output().unwrap();
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let output = child.wait_with_output().unwrap();
            panic!(
                "CLI exceeded its test deadline: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn success(output: Output) -> String {
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn process_evidence_sees_the_live_test_process() {
    assert!(process_exists(std::process::id()));
}

#[test]
fn native_cli_initialization_uses_only_the_sandbox_directories() {
    let sandbox = Sandbox::new();
    success(sandbox.run(&["version"]));
    for path in [
        sandbox.config_path(),
        sandbox.data_root().join("autarky/benchmarks.db"),
    ] {
        assert!(path.is_file(), "Missing native fixture file: {path:?}");
        assert!(path
            .canonicalize()
            .unwrap()
            .starts_with(sandbox.root.path().canonicalize().unwrap()));
    }
    if cfg!(target_os = "macos") {
        assert_eq!(
            sandbox.config_root(),
            sandbox.root.path().join("home/Library/Application Support")
        );
        assert_eq!(sandbox.data_root(), sandbox.config_root());
        assert!(!sandbox.root.path().join("config/autarky").exists());
        assert!(!sandbox.root.path().join("data/autarky").exists());
    }
}

#[test]
fn cli_rejects_invalid_repeat_counts_and_conflicting_selectors_before_startup() {
    let sandbox = Sandbox::new();
    for args in [
        vec!["bench", "--repeats", "0", "--passthrough", "does-not-exist"],
        vec!["compare", "--id2", "1"],
        vec!["compare", "--id1", "1", "--id2", "2", "--model", "Tiny"],
        vec![
            "compare",
            "--remote",
            "--similar",
            "--id1",
            "1",
            "--id2",
            "2",
        ],
        vec!["recommend", "--min-tps", "NaN"],
        vec!["recommend", "--limit", "0"],
        vec!["config", "--value", "orphan"],
    ] {
        let output = sandbox.run(&args);
        assert!(!output.status.success(), "{args:?}");
        assert!(
            !sandbox.config_path().exists(),
            "Clap must reject before initialization: {args:?}"
        );
    }
}

#[test]
fn config_keys_are_readable_and_api_key_output_is_masked() {
    let sandbox = Sandbox::new();
    let secret = "cli-test-private-credential";
    for args in [
        vec!["config", "--key", "api_key", "--value", secret],
        vec!["config", "--key", "api_key"],
        vec!["config"],
    ] {
        let output = success(sandbox.run(&args));
        assert!(!output.contains(secret));
        assert!(output.contains("<redacted>"));
    }
    for (key, value) in [
        ("server_url", "https://community.example/base/"),
        ("default_repeats", "2"),
        ("default_prefill_context", "1"),
        ("default_context_lengths", "1,2"),
        ("extra_flags", "[\"--threads\",\"4\"]"),
    ] {
        success(sandbox.run(&["config", "--key", key, "--value", value]));
        let output = success(sandbox.run(&["config", "--key", key]));
        assert!(output.contains(key));
    }
    assert!(!sandbox
        .run(&["config", "--key", "models_dir"])
        .status
        .success());
}

#[test]
fn fresh_install_has_no_placeholder_endpoint_and_help_uses_k_indices() {
    let sandbox = Sandbox::new();
    success(sandbox.run(&["version"]));
    let config: toml::Value =
        toml::from_str(&std::fs::read_to_string(sandbox.config_path()).unwrap()).unwrap();
    assert!(config.get("server_url").is_none());
    let help = success(sandbox.run(&["--help"]));
    assert!(help.contains("--context-lengths 1,2 --passthrough"));
    assert!(!help.contains("--context-lengths 512,1024"));
}

#[cfg(unix)]
#[test]
fn benchmark_resolves_config_defaults_and_explicit_cli_overrides() {
    for explicit in [false, true] {
        let sandbox = Sandbox::new();
        sandbox.set_config("default_context_lengths = '2,1,2'\ndefault_repeats = 2\ndefault_prefill_context = 1\nextra_flags = ['--threads', '4', '--gpu-layers', '0']\n");
        let fixture = sandbox.fixture();
        let mut args = vec!["bench"];
        if explicit {
            args.extend([
                "--context-lengths",
                "1",
                "--repeats",
                "1",
                "--prefill-context",
                "0",
            ]);
        }
        args.extend(["--passthrough", fixture.to_str().unwrap(), "--threads", "8"]);
        let text = success(sandbox.run(&args));
        assert!(
            !text.contains("Submitting"),
            "A fresh endpoint is local-only"
        );
        let metrics = sandbox.measurements();
        let contexts: Vec<_> = metrics[0]
            .as_array()
            .unwrap()
            .iter()
            .map(|metric| metric["context_length"].as_u64().unwrap())
            .collect();
        assert_eq!(contexts, if explicit { vec![1] } else { vec![1, 2] });
        let requests = std::fs::read_to_string(sandbox.root.path().join("requests.jsonl")).unwrap();
        assert_eq!(requests.lines().count(), if explicit { 1 } else { 6 });
        let args: Vec<String> = serde_json::from_str(
            &std::fs::read_to_string(sandbox.root.path().join("launch-args.json")).unwrap(),
        )
        .unwrap();
        assert!(args.windows(2).any(|pair| pair == ["--threads", "4"]));
        assert_eq!(
            args.iter()
                .rposition(|arg| arg == "--threads")
                .map(|i| args[i + 1].as_str()),
            Some("8")
        );
        assert!(sandbox
            .db()
            .query_row("SELECT model_id IS NULL FROM benchmark_runs", [], |row| row
                .get::<_, bool>(0))
            .unwrap());
    }
}

#[cfg(unix)]
#[test]
fn failed_partial_and_oom_runs_are_saved_and_return_failure() {
    for mode in ["error", "partial", "oom"] {
        let sandbox = Sandbox::new();
        let fixture = sandbox.fixture();
        let output = bounded_output(
            sandbox
                .command(&[
                    "bench",
                    "--context-lengths",
                    "2,1,2",
                    "--repeats",
                    "1",
                    "--passthrough",
                    fixture.to_str().unwrap(),
                    "--gpu-layers",
                    "0",
                ])
                .env("AUTARKY_FIXTURE_MODE", mode)
                .spawn()
                .unwrap(),
        );
        assert!(!output.status.success(), "{mode}");
        let records = sandbox.measurements();
        assert_eq!(records.len(), 1);
        let metrics = records[0].as_array().unwrap();
        assert_eq!(metrics[0]["context_length"], 1);
        assert_eq!(
            metrics.last().unwrap()["status"],
            if mode == "oom" { "oom" } else { "error" }
        );
        assert_eq!(metrics.len(), if mode == "partial" { 2 } else { 1 });
        assert!(!String::from_utf8_lossy(&output.stdout).contains("Submitting"));
    }
}

#[cfg(unix)]
#[test]
fn legacy_insecure_endpoint_does_not_invalidate_a_local_benchmark() {
    let sandbox = Sandbox::new();
    sandbox.set_config("server_url = 'http://community.example'\n");
    let fixture = sandbox.fixture();
    let output = sandbox.run(&[
        "bench",
        "--context-lengths",
        "1",
        "--repeats",
        "1",
        "--passthrough",
        fixture.to_str().unwrap(),
        "--gpu-layers",
        "0",
    ]);
    success(output);
    assert_eq!(sandbox.measurements().len(), 1);
}

// A real loopback socket inspects the payload; no public service sees test data.
#[cfg(unix)]
#[test]
fn automatic_submission_uses_configured_prefix_and_omits_private_device_ids() {
    let sandbox = Sandbox::new();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/prefix/", listener.local_addr().unwrap());
    listener.set_nonblocking(true).unwrap();
    sandbox.set_config(&format!(
        "server_url = '{url}'\napi_key = 'fixture-private-key'\n"
    ));
    let server = std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(10);
        let (mut stream, _) = loop {
            match listener.accept() {
                Ok(value) => break value,
                Err(error)
                    if error.kind() == std::io::ErrorKind::WouldBlock
                        && Instant::now() < deadline =>
                {
                    std::thread::sleep(Duration::from_millis(10))
                }
                Err(error) => panic!("No submission received: {error}"),
            }
        };
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut bytes = Vec::new();
        let (start, len) = loop {
            let mut chunk = [0; 4096];
            let read = stream.read(&mut chunk).unwrap();
            assert!(read > 0);
            bytes.extend_from_slice(&chunk[..read]);
            if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
                let header = String::from_utf8_lossy(&bytes[..end]).to_ascii_lowercase();
                assert!(header.starts_with("post /prefix/api/v1/benchmark http/1.1"));
                assert!(header.contains("x-api-key: fixture-private-key"));
                let len = header
                    .lines()
                    .find_map(|line| line.strip_prefix("content-length: "))
                    .unwrap()
                    .parse::<usize>()
                    .unwrap();
                break (end + 4, len);
            }
        };
        while bytes.len() < start + len {
            let mut chunk = [0; 4096];
            let read = stream.read(&mut chunk).unwrap();
            assert!(read > 0);
            bytes.extend_from_slice(&chunk[..read]);
        }
        let payload: Value = serde_json::from_slice(&bytes[start..start + len]).unwrap();
        stream
            .write_all(
                b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\nConnection: close\r\n\r\n{\"id\":7}",
            )
            .unwrap();
        payload
    });
    let fixture = sandbox.fixture();
    let output = success(sandbox.run(&[
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
    assert!(output.contains("ID #7"));
    let payload = server.join().unwrap();
    assert!(payload["hardware"].get("devices").is_none());
    assert!(payload["model"].is_null());
    assert_eq!(payload["raw_metrics"][0]["tokens_per_second"], 50.0);
    assert!(payload["raw_metrics"][0]["error"].is_null());
    assert!(!payload.to_string().contains("fixture-private-key"));
    assert_eq!(sandbox.measurements().len(), 1);
}
