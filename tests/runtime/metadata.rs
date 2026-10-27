//! Unix owned-process fixtures. No shell, descendants, network, or real config.
#![cfg(unix)]

use super::bounded_command_output;
use crate::test_process::process_exists;
use crate::{hardware::detect_hardware_for_server, launcher::get_server_version};
use std::{io, time::Duration};
use tempfile::TempDir;
use tokio::process::Command;

fn fixture(mode: &str) -> TempDir {
    // Never create an executable through a writable FD while other tests fork:
    // another child can inherit that FD and briefly prevent exec (ETXTBSY).
    // The checked-in executable is immutable for the entire test run. Symlinks
    // retain per-case __file__ paths without requiring the same filesystem.
    let source =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/metadata-server.py");
    let folder = tempfile::Builder::new()
        .prefix("autarky-metadata-test-")
        .tempdir()
        .unwrap();
    let script = folder.path().join("server.py");
    std::os::unix::fs::symlink(source, script).unwrap();
    std::fs::write(folder.path().join("mode"), mode).unwrap();
    folder
}

fn command(folder: &TempDir) -> Command {
    let mut command = Command::new("python3");
    command.arg(folder.path().join("server.py"));
    command
}

async fn fixture_pid(folder: &TempDir) -> u32 {
    tokio::time::timeout(Duration::from_secs(3), async {
        loop {
            if let Ok(text) = tokio::fs::read_to_string(folder.path().join("pid")).await {
                if let Ok(pid) = text.parse() {
                    return pid;
                }
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("owned fixture must start")
}

async fn assert_reaped(pid: u32) {
    tokio::time::timeout(Duration::from_secs(3), async {
        while process_exists(pid) {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("owned child must be killed and reaped, not orphaned or left as a zombie");
}

#[tokio::test]
async fn metadata_preserves_both_binary_streams_and_exit_status() {
    for mode in ["success", "nonzero"] {
        let folder = fixture(mode);
        let output = bounded_command_output(&mut command(&folder), Duration::from_secs(3), 4096)
            .await
            .unwrap();
        if mode == "success" {
            assert!(output.status.success());
            assert_eq!(output.stdout, b"out\x00\xff");
            assert_eq!(output.stderr, b"err\x00\xfe");
        } else {
            assert_eq!(output.status.code(), Some(7));
            assert!(!output.stdout.is_empty());
            assert!(!output.stderr.is_empty());
        }
        assert_reaped(fixture_pid(&folder).await).await;
    }
}

#[tokio::test]
async fn metadata_drains_both_pipes_concurrently_and_accepts_exact_limits() {
    let folder = fixture("large-both");
    let output = bounded_command_output(&mut command(&folder), Duration::from_secs(3), 65536)
        .await
        .unwrap();
    assert_eq!(output.stdout, vec![b'o'; 65536]);
    assert_eq!(output.stderr, vec![b'e'; 65536]);
    assert_reaped(fixture_pid(&folder).await).await;
}

#[tokio::test]
async fn metadata_caps_stdout_and_stderr_independently_and_reaps() {
    for mode in ["flood-stdout", "flood-stderr"] {
        let folder = fixture(mode);
        let error = bounded_command_output(&mut command(&folder), Duration::from_secs(3), 4096)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error
            .to_string()
            .contains(mode.strip_prefix("flood-").unwrap()));
        assert_reaped(fixture_pid(&folder).await).await;
    }
}

#[tokio::test]
async fn metadata_deadline_covers_reads_and_wait_even_after_both_pipes_close() {
    for mode in ["hang", "closed-pipes-hang"] {
        let folder = fixture(mode);
        let error = tokio::time::timeout(
            Duration::from_secs(3),
            bounded_command_output(&mut command(&folder), Duration::from_secs(1), 4096),
        )
        .await
        .expect("timeout plus cleanup must be bounded")
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        assert_reaped(fixture_pid(&folder).await).await;
    }
}

#[tokio::test]
async fn metadata_future_cancellation_kills_and_reaps_without_a_detached_task() {
    let folder = fixture("hang");
    let mut command = command(&folder);
    let task = tokio::spawn(async move {
        bounded_command_output(&mut command, Duration::from_secs(60), 4096).await
    });
    let pid = fixture_pid(&folder).await;
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert_reaped(pid).await;
}

#[tokio::test]
async fn metadata_direct_future_drop_kills_and_reaps() {
    let folder = fixture("hang");
    let mut command = command(&folder);
    let mut output = Box::pin(bounded_command_output(
        &mut command,
        Duration::from_secs(60),
        4096,
    ));
    let pid = tokio::select! {
        result = &mut output => panic!("hanging fixture unexpectedly returned: {result:?}"),
        pid = fixture_pid(&folder) => pid,
    };
    drop(output);
    assert_reaped(pid).await;
}

#[test]
fn metadata_runtime_teardown_reaps_without_tokio_or_manual_waitpid() {
    let folder = fixture("hang");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let pid = runtime.block_on(async {
        let mut command = command(&folder);
        tokio::spawn(async move {
            bounded_command_output(&mut command, Duration::from_secs(60), 4096).await
        });
        fixture_pid(&folder).await
    });
    drop(runtime);
    let deadline = std::time::Instant::now() + Duration::from_secs(1);
    while process_exists(pid) && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(
        !process_exists(pid),
        "metadata child {pid} must not remain as a zombie after runtime teardown"
    );
}

#[tokio::test]
async fn metadata_zero_byte_limit_accepts_empty_and_rejects_any_output() {
    assert_eq!(
        super::read_bounded(&b""[..], 0, "stdout").await.unwrap(),
        b""
    );
    let folder = fixture("success");
    let error = bounded_command_output(&mut command(&folder), Duration::from_secs(3), 0)
        .await
        .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    assert_reaped(fixture_pid(&folder).await).await;
}

#[tokio::test]
async fn metadata_version_uses_either_stream_and_rejects_nonzero_without_echoing_output() {
    for mode in ["version-stdout", "version-stderr"] {
        let folder = fixture(mode);
        assert_eq!(
            get_server_version(folder.path().join("server.py").to_str().unwrap())
                .await
                .unwrap(),
            "9013"
        );
        assert_reaped(fixture_pid(&folder).await).await;
    }
    let folder = fixture("nonzero");
    let error = get_server_version(folder.path().join("server.py").to_str().unwrap())
        .await
        .unwrap_err();
    assert!(!format!("{error:#}").contains("fixture-private-output"));
    assert_reaped(fixture_pid(&folder).await).await;
}

#[tokio::test]
async fn metadata_version_and_enumeration_enforce_both_output_caps() {
    for mode in ["flood-stdout", "flood-stderr"] {
        let folder = fixture(mode);
        assert!(
            get_server_version(folder.path().join("server.py").to_str().unwrap())
                .await
                .is_err()
        );
        assert_reaped(fixture_pid(&folder).await).await;
        let folder = fixture(mode);
        let server = folder.path().join("server.py");
        assert!(
            detect_hardware_for_server(server.to_str().unwrap(), &["--device=CUDA0".into()])
                .await
                .is_err()
        );
        assert_reaped(fixture_pid(&folder).await).await;
    }
}

#[tokio::test]
async fn metadata_enumeration_parses_native_devices_and_keeps_safe_fallback() {
    let folder = fixture("devices");
    let info = detect_hardware_for_server(
        folder.path().join("server.py").to_str().unwrap(),
        &["--device=CUDA0".into()],
    )
    .await
    .unwrap();
    assert_eq!(info.devices.len(), 1);
    assert_eq!(info.devices[0].id, "CUDA0");
    assert_eq!(info.vram_gb, 8.0);
    assert_reaped(fixture_pid(&folder).await).await;
    let folder = fixture("flood-stderr");
    let info = detect_hardware_for_server(folder.path().join("server.py").to_str().unwrap(), &[])
        .await
        .unwrap();
    assert!(info.gpu_family.starts_with("Unknown"));
    assert!(info.devices.is_empty());
    assert_reaped(fixture_pid(&folder).await).await;
}

#[tokio::test]
async fn metadata_version_and_enumeration_deadlines_kill_and_reap() {
    let version_folder = fixture("hang");
    let device_folder = fixture("hang");
    let version_path = version_folder.path().join("server.py");
    let device_path = device_folder.path().join("server.py");
    let flags = ["--device=CUDA0".into()];
    let (version, devices) = tokio::time::timeout(Duration::from_secs(7), async {
        tokio::join!(
            get_server_version(version_path.to_str().unwrap()),
            detect_hardware_for_server(device_path.to_str().unwrap(), &flags),
        )
    })
    .await
    .expect("metadata callers must include bounded cleanup in their deadlines");
    assert!(version.is_err());
    assert!(devices.is_err());
    assert_reaped(fixture_pid(&version_folder).await).await;
    assert_reaped(fixture_pid(&device_folder).await).await;
}

#[tokio::test]
async fn metadata_cancellation_through_version_and_enumeration_reaps() {
    let folder = fixture("hang");
    let path = folder.path().join("server.py");
    let mut version = Box::pin(get_server_version(path.to_str().unwrap()));
    let pid = tokio::select! {
        result = &mut version => panic!("hanging version returned: {result:?}"),
        pid = fixture_pid(&folder) => pid,
    };
    drop(version);
    assert_reaped(pid).await;

    let folder = fixture("hang");
    let path = folder.path().join("server.py");
    let mut devices = Box::pin(detect_hardware_for_server(path.to_str().unwrap(), &[]));
    let pid = tokio::select! {
        result = &mut devices => panic!("hanging enumeration returned: {result:?}"),
        pid = fixture_pid(&folder) => pid,
    };
    drop(devices);
    assert_reaped(pid).await;
}
