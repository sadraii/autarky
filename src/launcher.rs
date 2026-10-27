//! Launcher module
//! Spawns llama-server as child process and provides HTTP API interaction

use super::monitor::BenchmarkResult;
use crate::process::{OwnedProcess, ProcessControl};
use anyhow::{Context, Result};
use std::sync::Arc;
use std::time::Duration;
use tempfile::NamedTempFile;
use tokio::process::Command;
use tokio::sync::watch;
use tokio::time::Instant;

pub struct Server {
    api_base: String,
    client: reqwest::Client,
    log: Arc<NamedTempFile>,
    process: OwnedProcess,
    shutdown: Arc<ProcessControl>,
    exited: watch::Receiver<Option<Result<std::process::ExitStatus, String>>>,
    startup_deadline: Instant,
    /// Complete list of flags passed to the llama-server process,
    /// including injected infrastructure flags, excluding credentials.
    pub full_flags: Vec<String>,
}

struct ServerTiming {
    prompt_per_second: f64,
    tokens_per_second: f64,
}

const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;

// Dropping an active benchmark future stops the owned child as well.
struct RequestGuard(Option<Arc<ProcessControl>>);
impl Drop for RequestGuard {
    fn drop(&mut self) {
        if let Some(shutdown) = self.0.take() {
            shutdown.terminate();
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.shutdown.terminate();
    }
}

pub const SUPPORTED_CONTEXT_LENGTHS: [usize; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];

const BENCHMARK_PROMPTS: [&str; 9] = [
    include_str!("../prompts/1k.txt"),
    include_str!("../prompts/2k.txt"),
    include_str!("../prompts/4k.txt"),
    include_str!("../prompts/8k.txt"),
    include_str!("../prompts/16k.txt"),
    include_str!("../prompts/32k.txt"),
    include_str!("../prompts/64k.txt"),
    include_str!("../prompts/128k.txt"),
    include_str!("../prompts/256k.txt"),
];

pub fn validate_context_lengths(lengths: &[usize]) -> Result<()> {
    if lengths.is_empty() {
        anyhow::bail!("No supported context lengths fit the model's context window");
    }
    for &length in lengths {
        embedded_prompt(length)?;
    }
    Ok(())
}

fn embedded_prompt(context_length: usize) -> Result<&'static str> {
    let index = SUPPORTED_CONTEXT_LENGTHS
        .iter()
        .position(|&k| k == context_length)
        .with_context(|| {
            format!(
                "Unsupported context length {}k. Use 1, 2, 4, 8, 16, 32, 64, 128, or 256.",
                context_length
            )
        })?;
    let content = BENCHMARK_PROMPTS[index];
    if content.trim().is_empty() {
        anyhow::bail!("Embedded benchmark prompt for {}k is empty", context_length);
    }
    Ok(content)
}

impl Server {
    pub async fn benchmark(
        &mut self,
        context_length: usize,
        num_prompts: usize,
        warmup: usize,
        model_context_length: Option<usize>,
    ) -> Result<BenchmarkResult> {
        let mut guard = RequestGuard(Some(self.shutdown.clone()));
        let result = self
            .measure(context_length, num_prompts, warmup, model_context_length)
            .await;
        if result.is_err() || result.as_ref().is_ok_and(|r| r.status != "success") {
            self.stop().await?;
        }
        guard.0 = None;
        result
    }

    async fn measure(
        &self,
        context_length: usize,
        repeats: usize,
        warmup: usize,
        model_context_length: Option<usize>,
    ) -> Result<BenchmarkResult> {
        anyhow::ensure!(repeats > 0, "At least one measured repeat is required");
        let total = repeats
            .checked_add(warmup)
            .context("Repeat count overflow")?;
        let prompt = generate_test_prompt(context_length, model_context_length)?;
        let mut generation = 0.0;
        let mut prefill = 0.0;
        for index in 0..total {
            let timing = match self.send_prompt(&prompt).await {
                Ok(timing) => timing,
                Err(error) if error.downcast_ref::<OutOfMemory>().is_some() => {
                    return Ok(BenchmarkResult::oom(context_length));
                }
                Err(error) => {
                    return Err(error.context(format!(
                        "Benchmark request {} of {} failed",
                        index + 1,
                        total
                    )))
                }
            };
            if index >= warmup {
                // Divide before adding to avoid overflow when averaging finite rates.
                generation += timing.tokens_per_second / repeats as f64;
                prefill += timing.prompt_per_second / repeats as f64;
            }
        }
        anyhow::ensure!(
            generation.is_finite() && prefill.is_finite(),
            "Invalid aggregate timing"
        );
        Ok(BenchmarkResult::new(context_length, generation, prefill))
    }

    async fn send_prompt(&self, prompt: &str) -> Result<ServerTiming> {
        self.ensure_alive()?;
        let request = async {
            let body = serde_json::json!({
                "prompt": prompt,
                "n_predict": 100,
                "temperature": 0,
                "seed": 1,
                "stream": false,
                "cache_prompt": false,
                "ignore_eos": true,
            });
            let mut response = self
                .client
                .post(format!("{}/completion", self.api_base))
                .json(&body)
                .send()
                .await
                .context("Completion request failed")?;
            let status = response.status();
            let mut bytes = Vec::new();
            while let Some(chunk) = response.chunk().await.context("Completion body failed")? {
                anyhow::ensure!(
                    bytes.len() + chunk.len() <= MAX_RESPONSE_BYTES,
                    "Completion response is too large"
                );
                bytes.extend_from_slice(&chunk);
            }
            let value: serde_json::Value =
                serde_json::from_slice(&bytes).context("Completion response is not valid JSON")?;
            if !status.is_success() {
                if structured_oom(&value) {
                    return Err(OutOfMemory.into());
                }
                // Do not echo bodies: a service can reflect credentials or private prompts.
                anyhow::bail!(
                    "llama-server returned HTTP {} (not classified as OOM)",
                    status
                );
            }
            parse_response_timing(&value)
        };
        let mut exited = self.exited.clone();
        tokio::select! {
            result = tokio::time::timeout(REQUEST_TIMEOUT, request) => {
                result.context("Completion request exceeded 120 seconds")?
            }
            _ = exited.changed() => anyhow::bail!("llama-server exited during the completion request"),
        }
    }

    fn ensure_alive(&self) -> Result<()> {
        if let Some(status) = self.exited.borrow().as_ref() {
            anyhow::bail!("llama-server exited: {:?}", status);
        }
        anyhow::ensure!(
            !self.shutdown.is_stopping(),
            "llama-server is shutting down"
        );
        Ok(())
    }

    pub async fn wait_ready(&self) -> Result<()> {
        // This guard also handles cancellation while waiting for a hung health endpoint.
        let mut guard = RequestGuard(Some(self.shutdown.clone()));
        let result = tokio::time::timeout_at(self.startup_deadline, async {
            loop {
                self.ensure_alive()?;
                let mut exited = self.exited.clone();
                tokio::select! {
                    response = self.client.get(format!("{}/health", self.api_base))
                        .timeout(Duration::from_secs(1)).send() => {
                        if response.is_ok_and(|r| r.status().is_success()) {
                            self.ensure_alive()?;
                            return Ok(());
                        }
                    }
                    _ = exited.changed() => anyhow::bail!("llama-server exited before readiness"),
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        })
        .await
        .context("Server startup exceeded 30 seconds")
        .and_then(|result| result);
        if result.is_err() {
            self.stop().await?;
        }
        guard.0 = None;
        result
    }

    async fn stop(&self) -> Result<()> {
        self.shutdown.terminate();
        tokio::time::timeout(SHUTDOWN_TIMEOUT, self.process.wait())
            .await
            .context("Process shutdown exceeded 5 seconds; native owner still owns reaping")?
            .context("Process reaping failed")?;
        Ok(())
    }

    pub async fn kill(&mut self) -> Result<()> {
        self.stop().await
    }

    /// The actual loopback endpoint reported after the owned process bound port zero.
    pub fn api_base(&self) -> &str {
        &self.api_base
    }
}

#[derive(Debug)]
struct OutOfMemory;
impl std::fmt::Display for OutOfMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("llama-server reported out of memory")
    }
}
impl std::error::Error for OutOfMemory {}

fn structured_oom(value: &serde_json::Value) -> bool {
    let Some(error) = value.get("error").and_then(|v| v.as_object()) else {
        return false;
    };
    if ["type", "code"].iter().any(|key| {
        matches!(
            error.get(*key).and_then(|v| v.as_str()),
            Some("out_of_memory" | "out_of_memory_error" | "cuda_out_of_memory")
        )
    }) {
        return true;
    }
    // A more specific non-memory classification outranks free-form message text.
    if !matches!(
        error.get("type").and_then(|v| v.as_str()),
        None | Some("server_error")
    ) || error
        .get("code")
        .is_some_and(|code| !code.is_null() && code.as_i64() != Some(500))
    {
        return false;
    }
    // Older native servers expose these exception messages without an OOM code.
    // Do not classify reflected prompts, negations, or arbitrary substrings.
    error
        .get("message")
        .and_then(|v| v.as_str())
        .is_some_and(|message| {
            matches!(
                message.trim().to_ascii_lowercase().as_str(),
                "out of memory" | "std::bad_alloc" | "cuda error: out of memory"
            )
        })
}

fn parse_response_timing(value: &serde_json::Value) -> Result<ServerTiming> {
    anyhow::ensure!(
        value.get("stop").and_then(|v| v.as_bool()) == Some(true),
        "Completion is partial or missing its final marker"
    );
    anyhow::ensure!(
        value.get("truncated").and_then(|v| v.as_bool()) == Some(false),
        "Completion was truncated or lacks truncation state"
    );
    let timings = value
        .get("timings")
        .context("Current completion has no native timings")?;
    let count = |key: &str| -> Result<u64> {
        timings
            .get(key)
            .and_then(|v| v.as_u64())
            .with_context(|| format!("Missing native timing count: {}", key))
    };
    let positive = |key: &str| -> Result<f64> {
        let number = timings
            .get(key)
            .and_then(|v| v.as_f64())
            .with_context(|| format!("Missing native timing: {}", key))?;
        anyhow::ensure!(
            number.is_finite() && number > 0.0,
            "Native timing {} is not finite and positive",
            key
        );
        Ok(number)
    };
    let prompt_n = count("prompt_n")?;
    let predicted_n = count("predicted_n")?;
    anyhow::ensure!(
        prompt_n > 0 && predicted_n > 1 && count("cache_n")? == 0,
        "Native timing must include uncached prefill and generation"
    );
    anyhow::ensure!(
        value.get("tokens_predicted").and_then(|v| v.as_u64()) == Some(predicted_n),
        "Generation count does not match current completion"
    );
    anyhow::ensure!(
        value.get("tokens_evaluated").and_then(|v| v.as_u64()) == Some(prompt_n),
        "Prefill count does not match current completion"
    );
    positive("prompt_ms")?;
    positive("predicted_ms")?;
    Ok(ServerTiming {
        prompt_per_second: positive("prompt_per_second")?,
        tokens_per_second: positive("predicted_per_second")?,
    })
}

// Match common/arg.cpp::parse_csv_row, including its permissive quote rules.
// Unlike general CSV readers, quotes only open an empty field, spaces are data,
// and an unfinished quoted field is accepted.
fn parse_api_key_csv(input: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '"' if in_quotes => {
                if chars.peek() == Some(&'"') {
                    field.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            }
            '"' if field.is_empty() => in_quotes = true,
            ',' if !in_quotes => fields.push(std::mem::take(&mut field)),
            _ => field.push(ch),
        }
    }
    fields.push(field);
    fields
}

fn launch_arguments(flags: &[String]) -> Result<(Vec<String>, Vec<String>, Option<String>)> {
    let mut args = Vec::new();
    let mut public = Vec::new();
    // Native handlers append environment keys before processing CLI keys. Any
    // appended key works; keep an earlier one when a later option adds nothing.
    let mut api_key = std::env::var("LLAMA_API_KEY").ok().and_then(|keys| {
        parse_api_key_csv(&keys)
            .into_iter()
            .find(|key| !key.is_empty())
    });
    for argument in crate::arguments::parse(flags)? {
        let policy_key = argument.key.as_str();
        anyhow::ensure!(
            !matches!(
                policy_key,
                "--host"
                    | "--port"
                    | "--reuse-port"
                    | "--ssl-key-file"
                    | "--ssl-cert-file"
                    | "--api-prefix"
                    | "--log-disable"
                    | "--log-file"
                    | "-lv"
                    | "--verbosity"
                    | "--log-verbosity"
                    | "-v"
                    | "--verbose"
                    | "--log-verbose"
            ),
            "Native option conflicts with the owned loopback benchmark server"
        );
        if matches!(policy_key, "--api-key" | "--api-key-file") {
            let value = argument.values[0];
            let candidate = if policy_key == "--api-key" {
                parse_api_key_csv(value)
                    .into_iter()
                    .find(|key| !key.is_empty())
            } else {
                let metadata = std::fs::metadata(value).context("Cannot inspect API key file")?;
                anyhow::ensure!(
                    metadata.is_file() && metadata.len() <= 65536,
                    "API key file must be a regular file of at most 64 KiB"
                );
                let content = std::fs::read_to_string(value).context("Cannot read API key file")?;
                // std::getline removes only '\n'; quotes, commas, spaces and '\r'
                // remain part of the native file key, unlike CSV or str::lines.
                content
                    .split('\n')
                    .find(|line| !line.is_empty())
                    .map(str::to_owned)
            };
            if candidate.is_some() {
                api_key = candidate;
            }
        } else {
            public.extend_from_slice(argument.tokens);
        }
        args.extend_from_slice(argument.tokens);
    }
    for values in [&mut args, &mut public] {
        values.extend([
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            "0".into(),
            "--log-colors".into(),
            "off".into(),
            "--log-verbosity".into(),
            "3".into(),
        ]);
    }
    Ok((args, public, api_key))
}

pub async fn launch_server(server_path: &str, flags: &[String]) -> Result<Server> {
    let (args, full_flags, api_key) = launch_arguments(flags)?;
    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(key) = api_key {
        let mut header = reqwest::header::HeaderValue::from_str(&format!("Bearer {}", key))
            .context("API key is not a valid HTTP header")?;
        header.set_sensitive(true);
        headers.insert(reqwest::header::AUTHORIZATION, header);
    }
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .default_headers(headers)
        .connect_timeout(Duration::from_secs(1))
        .timeout(REQUEST_TIMEOUT)
        .build()?;
    let log = Arc::new(
        tempfile::Builder::new()
            .prefix("autarky-server-")
            .suffix(".log")
            .tempfile()?,
    );
    let startup_deadline = Instant::now() + STARTUP_TIMEOUT;
    let mut command = Command::new(server_path);
    command
        .args(args)
        .env_remove("LLAMA_ARG_HOST")
        .env_remove("LLAMA_ARG_PORT")
        .env_remove("LLAMA_ARG_REUSE_PORT")
        .env_remove("LLAMA_ARG_API_PREFIX")
        .env_remove("LLAMA_ARG_SSL_KEY_FILE")
        .env_remove("LLAMA_ARG_SSL_CERT_FILE")
        .env_remove("LLAMA_LOG_FILE")
        .env_remove("LLAMA_LOG_VERBOSITY")
        .stdin(std::process::Stdio::null())
        .stdout(log.as_file().try_clone()?)
        .stderr(log.as_file().try_clone()?);
    let process = OwnedProcess::spawn(command.as_std_mut(), log.clone())
        .context("Failed to spawn llama-server")?;
    let shutdown = process.control();
    let exited = process.exited.clone();
    let mut server = Server {
        api_base: String::new(),
        client,
        log,
        process,
        shutdown,
        exited,
        startup_deadline,
        full_flags,
    };
    let result = tokio::time::timeout_at(startup_deadline, async {
        loop {
            server.ensure_alive()?;
            use tokio::io::AsyncReadExt;
            let mut output = Vec::new();
            tokio::fs::File::open(server.log.path())
                .await
                .context("Cannot read private server log")?
                .take(8 * 1024 * 1024 + 1)
                .read_to_end(&mut output)
                .await?;
            anyhow::ensure!(
                output.len() <= 8 * 1024 * 1024,
                "Server startup log exceeded 8 MiB"
            );
            if let Some(endpoint) = bound_endpoint(&String::from_utf8_lossy(&output)) {
                server.api_base = endpoint;
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .context("Server did not report its bound loopback endpoint within 30 seconds")
    .and_then(|r| r);
    if let Err(error) = result {
        server.stop().await?;
        return Err(error);
    }
    Ok(server)
}

fn bound_endpoint(log: &str) -> Option<String> {
    log.lines().find_map(|line| {
        let address = line
            .split_once("server is listening on http://127.0.0.1:")?
            .1;
        let port = address.split_whitespace().next()?.parse::<u16>().ok()?;
        (port != 0).then(|| format!("http://127.0.0.1:{}", port))
    })
}

fn generate_test_prompt(
    context_length: usize,
    model_context_length: Option<usize>,
) -> Result<String> {
    let content = embedded_prompt(context_length)?;
    let Some(max_ctx) = model_context_length else {
        return Ok(content.to_string());
    };

    // Estimate how many words we can fit: leave 150 tokens headroom for
    // output tokens + chat template wrapping. English prose tokenizes at
    // roughly 1.5 tokens/word, so:
    //   target_words = (max_ctx - 150) / 1.5
    let headroom = 150usize;
    if max_ctx <= headroom {
        anyhow::bail!(
            "Model context window must exceed {} tokens to fit a benchmark prompt and output",
            headroom
        );
    }
    let max_tokens = max_ctx - headroom;
    let target_words = (max_tokens as f64 / 1.5) as usize;

    let words: Vec<&str> = content.split_whitespace().collect();

    if words.len() > target_words {
        if target_words == 0 {
            anyhow::bail!("Model context window is too small for a benchmark prompt");
        }
        Ok(words[..target_words].join(" "))
    } else {
        Ok(content.to_string())
    }
}

/// Get the version string from a llama-server binary by running `--version`.
pub async fn get_server_version(server_path: &str) -> Result<String> {
    let output = crate::utils::bounded_command_output(
        Command::new(server_path).arg("--version"),
        Duration::from_secs(5),
        64 * 1024,
    )
    .await
    .context("Failed to run llama-server --version")?;
    anyhow::ensure!(output.status.success(), "llama-server --version failed");

    // Combine stdout and stderr (CUDA init noise can be on either stream)
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );

    // Extract the "version: NNNN (hash)" line specifically
    for line in combined.lines() {
        if line.trim().starts_with("version:") {
            // Extract just the build number from "version: 9013 (hash)"
            let version = line.trim();
            let num = version
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.split(|c: char| !c.is_alphanumeric()).next())
                .unwrap_or(version);
            return Ok(num.to_string());
        }
    }

    // Fallback: use the last non-empty line
    if let Some(last) = combined.lines().rfind(|l| !l.trim().is_empty()) {
        return Ok(last.trim().to_string());
    }

    Ok("unknown".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_fields_match_native_quote_state_and_keep_empty_fields() {
        for (input, expected) in [
            ("", vec![""]),
            (",,", vec!["", "", ""]),
            ("\"\",x,", vec!["", "x", ""]),
            ("\"a,b\",c", vec!["a,b", "c"]),
            ("\"a\"\"b\"", vec!["a\"b"]),
            ("a\"b,c", vec!["a\"b", "c"]),
            (" \"a,b\"", vec![" \"a", "b\""]),
            ("\"\"\"\"", vec!["\""]),
            ("\"\"\"", vec!["\""]),
            ("\"a\"tail,b", vec!["atail", "b"]),
            ("\"\"\"\"tail,b", vec!["\"tail", "b"]),
            ("\"unclosed,a", vec!["unclosed,a"]),
            ("a,a", vec!["a", "a"]),
            ("é,\"鍵,值\"", vec!["é", "鍵,值"]),
        ] {
            assert_eq!(parse_api_key_csv(input), expected);
        }
    }

    #[test]
    fn api_keys_follow_native_csv_without_changing_forwarding_or_redaction() {
        for (input, expected) in [
            ("\"owned-key\"", "owned-key"),
            ("\"owned,key\",other", "owned,key"),
            ("\"owned\"\"key\"", "owned\"key"),
            (",,\"\",owned-key,", "owned-key"),
            ("  owned-key  ,other", "  owned-key  "),
            ("own\"ed,key", "own\"ed"),
            ("\"owned,key", "owned,key"),
            ("\"owned\"tail,other", "ownedtail"),
        ] {
            let flags = vec!["--api-key".into(), input.into()];
            let (forwarded, public, key) = launch_arguments(&flags).unwrap();
            assert_eq!(&forwarded[..flags.len()], flags);
            assert!(!public
                .iter()
                .any(|token| token == "--api-key" || token == input));
            assert_eq!(key.as_deref(), Some(expected));
        }
    }

    #[test]
    fn empty_api_key_options_do_not_erase_a_previously_appended_key() {
        let flags = ["--api-key", "owned-key", "--api-key", ",,\"\""].map(str::to_owned);
        assert_eq!(
            launch_arguments(&flags).unwrap().2.as_deref(),
            Some("owned-key")
        );
    }

    #[test]
    fn key_files_preserve_raw_lines_instead_of_decoding_csv_or_trimming() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        use std::io::Write;
        file.write_all(b"\n\"owned,key\" \r\nsecond\n").unwrap();
        let flags = vec!["--api-key-file".into(), file.path().display().to_string()];
        assert_eq!(
            launch_arguments(&flags).unwrap().2.as_deref(),
            Some("\"owned,key\" \r")
        );
    }

    #[test]
    fn oom_requires_positive_classification_not_incidental_error_text() {
        for (kind, message) in [
            ("server_error", "not out of memory"),
            (
                "server_error",
                "the reflected prompt mentions out of memory",
            ),
            ("server_error", "this failure is not std::bad_alloc"),
            ("invalid_request_error", "out of memory"),
        ] {
            assert!(!structured_oom(&serde_json::json!({
                "error": {"type": kind, "message": message}
            })));
        }
        for kind in ["out_of_memory", "out_of_memory_error", "cuda_out_of_memory"] {
            assert!(structured_oom(
                &serde_json::json!({"error": {"type": kind}})
            ));
        }
        for message in [
            "out of memory",
            "std::bad_alloc",
            "CUDA error: out of memory",
        ] {
            assert!(structured_oom(&serde_json::json!({
                "error": {"type": "server_error", "message": message}
            })));
        }
    }

    #[test]
    fn every_supported_context_has_an_embedded_prompt() {
        validate_context_lengths(&SUPPORTED_CONTEXT_LENGTHS).unwrap();
        for length in SUPPORTED_CONTEXT_LENGTHS {
            let prompt = generate_test_prompt(length, None).unwrap();
            assert!(
                prompt.len() > length * 1000,
                "{}k prompt is missing data",
                length
            );
        }
    }

    #[test]
    fn unsupported_contexts_fail_instead_of_benchmarking_empty_prompts() {
        for length in [0, 3, 131, 512] {
            assert!(generate_test_prompt(length, None).is_err());
            assert!(validate_context_lengths(&[length]).is_err());
        }
        assert!(validate_context_lengths(&[]).is_err());
    }

    #[test]
    fn prompt_trimming_preserves_data_and_rejects_impossible_windows() {
        let full = generate_test_prompt(2, None).unwrap();
        let trimmed = generate_test_prompt(2, Some(2048)).unwrap();
        assert!(!trimmed.is_empty());
        assert!(trimmed.len() < full.len());
        for window in [0, 150, 151] {
            assert!(generate_test_prompt(1, Some(window)).is_err());
        }
    }
}

#[cfg(test)]
#[path = "../tests/runtime/inference.rs"]
mod runtime_regressions;
