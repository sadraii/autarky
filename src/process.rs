//! Runtime-independent ownership for every subprocess started by Autarky.
//!
//! Drop sends the OS kill synchronously. A native thread retains the actual
//! Child handle until try_wait reaps it, even after the Tokio runtime is gone.
use std::io;
use std::process::{Child, ChildStderr, ChildStdout, Command, ExitStatus};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;
use tokio::sync::watch;

const REAP_INTERVAL: Duration = Duration::from_millis(10);
type ProcessExit = Result<ExitStatus, String>;

struct State {
    starting: bool,
    stopping: bool,
    child: Option<Child>,
}

pub(crate) struct ProcessControl {
    state: Mutex<State>,
    wake: Condvar,
}

impl ProcessControl {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|error| error.into_inner())
    }

    /// Only nonblocking child operations run under this mutex. Never wait here:
    /// cancellation must not block an async executor behind an unkillable child.
    pub(crate) fn terminate(&self) {
        let mut state = self.lock();
        state.stopping = true;
        state.starting = false;
        if let Some(child) = state.child.as_mut() {
            // An exit can race cancellation. The owner still polls and reaps.
            let _ = child.kill();
        }
        self.wake.notify_one();
    }

    pub(crate) fn is_stopping(&self) -> bool {
        self.lock().stopping
    }
}

pub(crate) struct OwnedProcess {
    control: Arc<ProcessControl>,
    pub(crate) exited: watch::Receiver<Option<ProcessExit>>,
    pub(crate) stdout: Option<ChildStdout>,
    pub(crate) stderr: Option<ChildStderr>,
}

impl OwnedProcess {
    pub(crate) fn spawn(
        command: &mut Command,
        keep_alive: impl Send + 'static,
    ) -> io::Result<Self> {
        let control = Arc::new(ProcessControl {
            state: Mutex::new(State {
                starting: true,
                stopping: false,
                child: None,
            }),
            wake: Condvar::new(),
        });
        let (exited_tx, exited) = watch::channel(None);
        let reaper = control.clone();
        // Allocate the reaper BEFORE spawning a process. A thread creation error
        // therefore cannot strand a child with no owner capable of reaping it.
        std::thread::Builder::new()
            .name("autarky-child-reaper".into())
            .spawn(move || {
                let status = {
                    let mut state = reaper.lock();
                    loop {
                        if !state.starting {
                            let Some(child) = state.child.as_mut() else {
                                return;
                            };
                            match child.try_wait() {
                                Ok(Some(status)) => {
                                    state.child.take();
                                    break Ok(status);
                                }
                                Ok(None) => {}
                                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                                Err(error) => {
                                    state.child.take();
                                    break Err(format!("Process wait failed: {error}"));
                                }
                            }
                        }
                        state = reaper
                            .wake
                            .wait_timeout(state, REAP_INTERVAL)
                            .unwrap_or_else(|error| error.into_inner())
                            .0;
                    }
                };
                // Release private logs and handles before advertising completion.
                drop(keep_alive);
                let _ = exited_tx.send(Some(status));
            })?;
        let mut owner = Self {
            control,
            exited,
            stdout: None,
            stderr: None,
        };
        // If spawn fails, owner's Drop wakes the preallocated thread to exit.
        let mut child = command.spawn()?;
        owner.stdout = child.stdout.take();
        owner.stderr = child.stderr.take();
        {
            let mut state = owner.control.lock();
            state.child = Some(child);
            state.starting = false;
        }
        owner.control.wake.notify_one();
        Ok(owner)
    }

    pub(crate) fn control(&self) -> Arc<ProcessControl> {
        self.control.clone()
    }

    pub(crate) async fn wait(&self) -> io::Result<ExitStatus> {
        let mut exited = self.exited.clone();
        loop {
            if let Some(status) = exited.borrow().as_ref() {
                return status.clone().map_err(io::Error::other);
            }
            exited
                .changed()
                .await
                .map_err(|_| io::Error::other("Process reaper stopped before exit"))?;
        }
    }
}

#[cfg(test)]
#[path = "../tests/runtime/process.rs"]
mod regressions;

impl Drop for OwnedProcess {
    fn drop(&mut self) {
        self.control.terminate();
    }
}
