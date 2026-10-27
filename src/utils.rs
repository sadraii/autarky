//! Utils module
//! Helpers, formatters, auto-discovery logic

use std::io;
use std::process::{Output, Stdio};
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;

const SUBPROCESS_CLEANUP_TIMEOUT: Duration = Duration::from_secs(1);

/// Capture both streams independently, with one deadline for reads and exit.
///
/// The byte limit applies to each stream. Failure kills and reaps the child,
/// allowing at most one extra second for cleanup. Cancellation drops the owned
/// native process owner, which reaps independently of the Tokio runtime.
pub(crate) async fn bounded_command_output(
    command: &mut Command,
    timeout: Duration,
    max_bytes_per_stream: usize,
) -> io::Result<Output> {
    let deadline = tokio::time::Instant::now() + timeout;
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = crate::process::OwnedProcess::spawn(command.as_std_mut(), ())?;
    let stdout =
        tokio::process::ChildStdout::from_std(child.stdout.take().expect("stdout is piped"))?;
    let stderr =
        tokio::process::ChildStderr::from_std(child.stderr.take().expect("stderr is piped"))?;
    let result = tokio::time::timeout_at(deadline, async {
        tokio::try_join!(
            read_bounded(stdout, max_bytes_per_stream, "stdout"),
            read_bounded(stderr, max_bytes_per_stream, "stderr"),
            child.wait(),
        )
    })
    .await;
    let error = match result {
        Ok(Ok((stdout, stderr, status))) => {
            return Ok(Output {
                status,
                stdout,
                stderr,
            });
        }
        Ok(Err(error)) => error,
        Err(_) => io::Error::new(io::ErrorKind::TimedOut, "Metadata subprocess timed out"),
    };
    if let Err(cleanup) = terminate_and_reap(&mut child).await {
        return Err(io::Error::new(error.kind(), format!("{error}; {cleanup}")));
    }
    Err(error)
}

async fn read_bounded(
    mut stream: impl AsyncRead + Unpin,
    limit: usize,
    name: &'static str,
) -> io::Result<Vec<u8>> {
    let mut output = Vec::new();
    let mut buffer = [0; 8192];
    loop {
        let count = stream.read(&mut buffer).await?;
        if count == 0 {
            return Ok(output);
        }
        if count > limit.saturating_sub(output.len()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Metadata subprocess {name} exceeded its byte limit"),
            ));
        }
        output.extend_from_slice(&buffer[..count]);
    }
}

async fn terminate_and_reap(child: &mut crate::process::OwnedProcess) -> io::Result<()> {
    // Keep ownership in the native reaper even if this bounded wait expires.
    child.control().terminate();
    match tokio::time::timeout(SUBPROCESS_CLEANUP_TIMEOUT, child.wait()).await {
        Ok(result) => result.map(|_| ()),
        Err(_) => Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "Metadata subprocess cleanup timed out",
        )),
    }
}

/// Format bytes to human-readable string
pub fn format_bytes(bytes: u64) -> String {
    if bytes > 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes > 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1_000_000.0)
    } else if bytes > 1_000 {
        format!("{:.1} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Format duration to human-readable string
pub fn format_duration(secs: f64) -> String {
    if secs > 60.0 {
        format!("{:.1} min", secs / 60.0)
    } else if secs > 1.0 {
        format!("{:.1} sec", secs)
    } else {
        format!("{:.0} ms", secs * 1000.0)
    }
}

/// Remove CSI/OSC terminal controls without indexing UTF-8 bytes.
pub fn strip_ansi(text: &str) -> String {
    let mut chars = text.chars().peekable();
    let mut output = String::with_capacity(text.len());
    while let Some(character) = chars.next() {
        if character != '\u{1b}' {
            if !character.is_control() {
                output.push(character);
            }
            continue;
        }
        match chars.next() {
            Some('[') => {
                for ch in chars.by_ref() {
                    if ('@'..='~').contains(&ch) {
                        break;
                    }
                }
            }
            Some(']') => {
                while let Some(ch) = chars.next() {
                    if ch == '\u{7}' {
                        break;
                    }
                    if ch == '\u{1b}' && chars.peek() == Some(&'\\') {
                        chars.next();
                        break;
                    }
                }
            }
            _ => {}
        }
    }
    output
}

pub fn display_width(text: &str) -> usize {
    unicode_width::UnicodeWidthStr::width(strip_ansi(text).as_str())
}

/// Pad by terminal cells, not bytes, scalar values, or ANSI sequence length.
pub fn pad_display(text: &str, width: usize) -> String {
    format!(
        "{}{}",
        text,
        " ".repeat(width.saturating_sub(display_width(text)))
    )
}

/// Return plain text that fits the given display width. Never split UTF-8.
pub fn truncate_display(text: &str, width: usize) -> String {
    let plain = strip_ansi(text);
    use unicode_truncate::UnicodeTruncateStr;
    // The maintained grapheme iterator keeps ZWJ sequences intact without
    // rescanning every prefix of untrusted, potentially large model metadata.
    plain.unicode_truncate(width).0.to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn truncation_preserves_zero_width_and_complete_graphemes() {
        assert_eq!(truncate_display("", 0), "");
        assert_eq!(truncate_display("e\u{301}", 0), "");
        assert_eq!(truncate_display("\u{301}👩‍💻x", 0), "\u{301}");
        assert_eq!(truncate_display("👩‍💻x", 1), "");
        assert_eq!(truncate_display("👩‍💻x", 2), "👩‍💻");
        assert_eq!(truncate_display("👨‍👩‍👧‍👦模型", 3), "👨‍👩‍👧‍👦");
        assert_eq!(truncate_display("e\u{301}模", 1), "e\u{301}");
    }

    #[test]
    fn long_model_metadata_truncation_has_bounded_linear_work() {
        // GGUF names may be 16 MiB. A width-24 display must not rescan every
        // complete prefix: that quadratic algorithm cannot finish this input.
        let text = "a".repeat(16 * 1024 * 1024);
        let began = std::time::Instant::now();
        assert_eq!(truncate_display(&text, 24), "a".repeat(24));
        assert!(began.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn ansi_padding_and_unicode_truncation_use_display_cells() {
        let text = "\u{1b}[1m模型e\u{301}\u{1b}[0m";
        assert_eq!(display_width(text), 5);
        assert_eq!(display_width(&pad_display(text, 9)), 9);
        assert_eq!(truncate_display(text, 3), "模");
        assert_eq!(truncate_display(text, 5), "模型e\u{301}");
        assert_eq!(truncate_display("👩‍💻!", 2), "👩‍💻");
        assert_eq!(
            strip_ansi("\u{1b}]8;;https://invalid\u{7}link\u{1b}]8;;\u{1b}\\!"),
            "link!"
        );
    }
}

#[cfg(test)]
#[path = "../tests/runtime/metadata.rs"]
mod metadata_regressions;
