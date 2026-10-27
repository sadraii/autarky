//! Exercise the native owner with no Tokio runtime available for cleanup.
#![cfg(target_os = "linux")]
use super::*;
use std::{
    io::{BufRead, BufReader},
    path::Path,
    process::Stdio,
    sync::mpsc,
    time::Instant,
};

struct Released(mpsc::Sender<()>);
impl Drop for Released {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

#[test]
fn immediate_exit_helper() {
    if std::env::var_os("AUTARKY_PROCESS_IMMEDIATE_EXIT_TEST").is_none() {
        return;
    }
    use std::io::Write;
    let mut command = Command::new("python3");
    command
        .args([
            "-c",
            "import os,time; print(os.getpid(), flush=True); time.sleep(120)",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut owner = OwnedProcess::spawn(&mut command, ()).unwrap();
    let mut pid = String::new();
    BufReader::new(owner.stdout.take().unwrap())
        .read_line(&mut pid)
        .unwrap();
    println!("OWNED_CHILD_PID={}", pid.trim());
    std::io::stdout().flush().unwrap();
    drop(owner);
    // No runtime teardown, join, sleep, or scheduling opportunity is required
    // between Drop and exit for the OS kill to have been initiated.
    std::process::exit(0);
}

#[test]
fn immediate_parent_exit_cannot_leave_the_owned_child_running() {
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "process::regressions::immediate_exit_helper",
            "--nocapture",
        ])
        .env("AUTARKY_PROCESS_IMMEDIATE_EXIT_TEST", "1")
        .output()
        .unwrap();
    assert!(output.status.success());
    let output = String::from_utf8(output.stdout).unwrap();
    let pid: u32 = output
        .lines()
        .find_map(|line| line.strip_prefix("OWNED_CHILD_PID="))
        .unwrap()
        .parse()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let status = std::fs::read_to_string(format!("/proc/{pid}/status"));
        let stopped = match status {
            Ok(status) => status.lines().any(|line| {
                line.starts_with("State:")
                    && (line.contains("Z (zombie)") || line.contains("X (dead)"))
            }),
            Err(_) if !Path::new(&format!("/proc/{pid}")).exists() => true,
            Err(error) => panic!("cannot inspect owned child: {error}"),
        };
        if stopped {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "Drop must initiate kill before the parent can exit"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    // Once the parent exits, only the OS adopter can reap an orphaned zombie.
    // The separate runtime-teardown tests require PID absence, not this fallback.
}

#[test]
fn spawn_errors_do_not_leave_a_reaper_or_retained_resources() {
    for _ in 0..8 {
        let (released, receiver) = mpsc::channel();
        assert!(
            OwnedProcess::spawn(&mut Command::new("/does/not/exist"), Released(released)).is_err()
        );
        receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("failed spawn must release the preallocated native reaper");
    }
}

#[test]
fn native_owners_reap_and_release_threads_pipes_and_resources_without_a_runtime() {
    for exits_naturally in [false, true] {
        let mut owners = Vec::new();
        for _ in 0..8 {
            let (released, receiver) = mpsc::channel();
            let mut command = Command::new("python3");
            command
                .args([
                    "-c",
                    if exits_naturally {
                        "import os; print(os.getpid(), flush=True)"
                    } else {
                        "import os,time; print(os.getpid(), flush=True); time.sleep(120)"
                    },
                ])
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            let mut owner = OwnedProcess::spawn(&mut command, Released(released)).unwrap();
            let mut text = String::new();
            BufReader::new(owner.stdout.take().unwrap())
                .read_line(&mut text)
                .unwrap();
            let pid: u32 = text.trim().parse().unwrap();
            let control = owner.control();
            let weak = Arc::downgrade(&control);
            drop(control);
            owners.push((owner, pid, weak, receiver));
        }
        for (owner, pid, weak, released) in owners {
            drop(owner);
            released
                .recv_timeout(Duration::from_secs(1))
                .expect("the native reaper must release retained resources");
            let deadline = Instant::now() + Duration::from_secs(1);
            while weak.upgrade().is_some() && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(1));
            }
            assert!(
                weak.upgrade().is_none(),
                "no process owner may remain in a detached thread"
            );
            assert!(
                !Path::new(&format!("/proc/{pid}")).exists(),
                "actual PID must be gone; a zombie is not a completed cleanup"
            );
        }
    }
}
