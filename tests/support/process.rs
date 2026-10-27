//! Process evidence for owned Unix test fixtures. Zombies still count as present.

pub fn process_exists(pid: u32) -> bool {
    assert!(pid > 0, "Process evidence requires a positive PID");
    #[cfg(target_os = "linux")]
    {
        std::path::Path::new(&format!("/proc/{pid}")).exists()
    }
    #[cfg(all(unix, not(target_os = "linux")))]
    {
        // Signal zero does not terminate the process. These fixtures retain the
        // test user's UID, so an unsuccessful probe means the owned PID is gone.
        let status = std::process::Command::new("/bin/kill")
            .args(["-0", &pid.to_string()])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .expect("Cannot run the native process-presence probe");
        match status.code() {
            Some(0) => true,
            Some(1) => false,
            other => panic!("Unexpected process-presence probe status: {other:?}"),
        }
    }
}
