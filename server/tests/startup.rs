use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Invalid safety settings must stop startup, not silently select defaults.
#[test]
fn invalid_environment_values_fail_before_any_database_write() {
    let scratch = std::env::var_os("TMPDIR").expect("TMPDIR must name the test scratch directory");
    let dir = tempfile::Builder::new()
        .prefix("autarky-server-startup-")
        .tempdir_in(scratch)
        .unwrap();
    for (name, value) in [
        ("AUTARKY_RATE_MAX", "0"),
        ("AUTARKY_RATE_MAX", "4294967295"),
        ("AUTARKY_RATE_WINDOW", "0"),
        ("AUTARKY_RATE_WINDOW", "18446744073709551615"),
        ("AUTARKY_GLOBAL_RATE_MAX", "0"),
        ("AUTARKY_MAX_SUBMISSIONS", "0"),
        ("AUTARKY_DB_MAX_MB", "0"),
        ("PORT", "0"),
        ("PORT", "65536"),
        ("BIND_ADDR", "not-an-IP-address"),
    ] {
        let path = dir.path().join(format!("{name}-{value}.db"));
        let mut child = Command::new(env!("CARGO_BIN_EXE_autarky-server"))
            .env_clear()
            .env("AUTARKY_SERVER_DB", &path)
            .env("BIND_ADDR", "127.0.0.1")
            .env(name, value)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if child.try_wait().unwrap().is_some() {
                break;
            }
            if Instant::now() >= deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("{name}={value} did not stop startup");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let output = child.wait_with_output().unwrap();
        assert!(!output.status.success(), "{name}={value} must fail");
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(
            error.contains(name),
            "configuration error must name the setting, not its value"
        );
        assert!(
            !path.exists(),
            "{name}={value} must fail before database initialization"
        );
    }
}
