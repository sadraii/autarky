use anyhow::{Context, Result};
use autarky::community;
use autarky::config;
use autarky::db;
use autarky::hardware;
use autarky::launcher;
use autarky::monitor::{self, BenchmarkResult};
use autarky::utils::strip_ansi as terminal_text;

use clap::{Parser, Subcommand};
use std::io::{IsTerminal, Write};

#[derive(Parser, Debug)]
#[command(
    name = "autarky",
    version,
    about = "Local LLM benchmark launcher for llama-server",
    long_about = "Autarky benchmarks local LLM models via llama-server, tracking performance across context lengths. Runs stay local unless a community server is configured. Use --offline to disable all community calls.\n\nExamples:\n  autarky bench --context-lengths 1,2 --passthrough ./llama-server --model ~/models/mymodel.gguf\n  autarky list --model tinyllama"
)]
struct Cli {
    /// Skip community calls: no uploads, no server queries, no remote comparisons
    #[arg(long, global = true)]
    offline: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Launch llama-server and benchmark it
    Bench {
        /// Comma-separated context lengths (K values: 1, 2, 4, ... up to model max)
        ///
        /// Each value matches a prompt file (1=1k.txt, 2=2k.txt, 4=4k.txt, etc.)
        #[arg(long)]
        context_lengths: Option<String>,

        /// Number of measurement repeats per context length
        #[arg(long, value_parser = positive_usize)]
        repeats: Option<usize>,

        /// Run one unmeasured warmup before cold-cache measurements (0 or 1)
        #[arg(long, value_parser = clap::value_parser!(u8).range(0..=1))]
        prefill_context: Option<u8>,

        /// Enable TUI dashboard
        #[arg(long)]
        tui: bool,

        /// Full llama-server command to execute. Everything after
        /// --passthrough is forwarded as-is to the server process.
        /// Example: --passthrough ./llama-server --model model.gguf --ctx-size 4096
        #[arg(long, allow_hyphen_values = true, num_args = 1..)]
        passthrough: Vec<String>,
    },

    /// List benchmark history
    List {
        /// Filter by model name (partial match)
        #[arg(long)]
        model: Option<String>,

        /// Filter by GPU model (partial match)
        #[arg(long)]
        gpu: Option<String>,

        /// Filter by quantization type (partial match, e.g., Q4_K_M)
        #[arg(long)]
        quant: Option<String>,

        /// Filter by date range (format: YYYY-MM-DD,YYYY-MM-DD)
        #[arg(long, value_parser = date_range_arg)]
        date_range: Option<String>,
    },

    /// Compare benchmarks
    Compare {
        /// First benchmark ID
        #[arg(long, conflicts_with_all = ["model", "gpu"], value_parser = positive_id)]
        id1: Option<i64>,

        /// Second benchmark ID
        #[arg(long, requires = "id1", conflicts_with_all = ["similar", "model", "gpu"], value_parser = positive_id)]
        id2: Option<i64>,

        /// Model name to compare (compares latest two for that model)
        #[arg(long)]
        model: Option<String>,

        /// GPU model filter (compares latest two for that GPU)
        #[arg(long)]
        gpu: Option<String>,

        /// Compare against similar hardware setups
        #[arg(long, conflicts_with = "remote")]
        similar: bool,

        /// Compare against remote benchmarks on the community server
        #[arg(long, requires_all = ["id1", "id2"], conflicts_with_all = ["similar", "model", "gpu"])]
        remote: bool,
    },

    /// Export benchmarks to JSON or CSV
    Export {
        /// Export format: json or csv
        #[arg(long, default_value = "json", value_parser = ["json", "csv"])]
        format: String,

        /// Output file path (prints to stdout if not specified)
        #[arg(long)]
        file: Option<String>,

        /// Filter by model name (partial match)
        #[arg(long)]
        model: Option<String>,

        /// Filter by GPU model (partial match)
        #[arg(long)]
        gpu: Option<String>,

        /// Filter by quantization type (partial match)
        #[arg(long)]
        quant: Option<String>,

        /// Filter by date range (format: YYYY-MM-DD,YYYY-MM-DD)
        #[arg(long, value_parser = date_range_arg)]
        date_range: Option<String>,
    },

    /// Show leaderboard
    Leaderboard {
        /// Time period: 24h, 7d, 30d, all
        #[arg(long, default_value = "7d", value_parser = ["24h", "7d", "30d", "all"])]
        period: String,

        /// Filter by model
        #[arg(long)]
        model: Option<String>,

        /// Filter by GPU model
        #[arg(long)]
        gpu: Option<String>,

        /// Sort by: tps, prompt, context
        #[arg(long, default_value = "tps", value_parser = ["tps", "prompt", "context", "ctx"])]
        sort: String,

        /// Show top N results
        #[arg(long, default_value = "10", value_parser = positive_usize)]
        top: usize,

        /// Group results: none, model, gpu, hardware
        #[arg(long, default_value = "none", value_parser = ["none", "model", "gpu", "hardware"])]
        group_by: String,
    },

    /// Search benchmarks
    Search {
        /// Search query
        query: Option<String>,

        /// Search by specific flags
        #[arg(long)]
        flags: Option<String>,
    },

    /// Recommend flags based on similar GPU setups
    Recommend {
        /// GPU family to search (default: auto-detect)
        #[arg(long)]
        gpu: Option<String>,

        /// Minimum TPS to include in recommendations
        #[arg(long, default_value = "0", value_parser = nonnegative_f64)]
        min_tps: f64,

        /// Number of results to show
        #[arg(long, default_value = "10", value_parser = recommendation_limit)]
        limit: i64,

        /// Output format: table, json
        #[arg(long, default_value = "table", value_parser = ["table", "json"])]
        format: String,
    },

    /// Show or set configuration
    Config {
        /// Configuration key
        #[arg(long)]
        key: Option<String>,

        /// Configuration value (to set)
        #[arg(long, requires = "key")]
        value: Option<String>,
    },

    /// Show version
    Version,

    /// Delete benchmark entries: "all" deletes all entries matching
    /// your hardware key, "last" deletes only the most recent entry.
    Delete {
        /// "all" or "last"
        #[arg(value_parser = ["all", "last"])]
        mode: String,
    },
}

fn positive_usize(input: &str) -> std::result::Result<usize, String> {
    input
        .parse::<usize>()
        .ok()
        .filter(|value| (1..=1000).contains(value))
        .ok_or_else(|| "Value must be an integer between 1 and 1000".to_string())
}

fn positive_id(input: &str) -> std::result::Result<i64, String> {
    input
        .parse::<i64>()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| "Benchmark ID must be a positive integer".to_string())
}

fn recommendation_limit(input: &str) -> std::result::Result<i64, String> {
    input
        .parse::<i64>()
        .ok()
        .filter(|value| (1..=200).contains(value))
        .ok_or_else(|| "Limit must be between 1 and 200".to_string())
}

fn nonnegative_f64(input: &str) -> std::result::Result<f64, String> {
    input
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite() && *value >= 0.0)
        .ok_or_else(|| "Value must be finite and nonnegative".to_string())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Ensure database and config are initialized on first run
    ensure_initialized()?;

    match cli.command {
        Some(Commands::Bench {
            context_lengths,
            repeats,
            prefill_context,
            tui,
            passthrough,
        }) => {
            run_bench(
                passthrough,
                context_lengths,
                repeats,
                prefill_context,
                tui,
                cli.offline,
            )
            .await?;
        }
        Some(Commands::List {
            model,
            gpu,
            quant,
            date_range,
        }) => {
            run_list(model, gpu, quant, date_range)?;
        }
        Some(Commands::Compare {
            id1,
            id2,
            model,
            gpu,
            similar,
            remote,
        }) => {
            if remote {
                if cli.offline {
                    anyhow::bail!("--offline is set: remote comparison requires network access. Run without --offline or omit --remote for local comparison.");
                }
                run_compare_remote(id1, id2, model, gpu).await?;
            } else {
                run_compare(id1, id2, model, gpu, similar)?;
            }
        }
        Some(Commands::Export {
            format,
            file,
            model,
            gpu,
            quant,
            date_range,
        }) => {
            run_export(format, file, model, gpu, quant, date_range)?;
        }
        Some(Commands::Leaderboard {
            period,
            model,
            gpu,
            sort,
            top,
            group_by,
        }) => {
            run_leaderboard(period, model, gpu, sort, top, group_by)?;
        }
        Some(Commands::Search { query, flags }) => {
            if cli.offline {
                anyhow::bail!("--offline is set: searching the community server requires network access. Run without --offline to search online.");
            }
            run_search(query, flags).await?;
        }
        Some(Commands::Recommend {
            gpu,
            min_tps,
            limit,
            format,
        }) => {
            run_recommend(gpu, min_tps, limit, format).await?;
        }
        Some(Commands::Config { key, value }) => {
            run_config(key, value)?;
        }
        Some(Commands::Delete { mode }) => match mode.as_str() {
            "all" | "last" => {
                run_delete(&mode).await?;
            }
            _ => {
                anyhow::bail!("Invalid delete mode: '{}'. Use 'all' or 'last'.", mode);
            }
        },
        Some(Commands::Version) => {
            println!("autarky {}", env!("CARGO_PKG_VERSION"));
        }
        None => {
            // No command - show help
            Cli::parse_from(["autarky", "--help"]);
        }
    }

    Ok(())
}

async fn run_bench(
    passthrough: Vec<String>,
    context_lengths: Option<String>,
    repeats: Option<usize>,
    prefill_context: Option<u8>,
    tui: bool,
    offline: bool,
) -> anyhow::Result<()> {
    let mut cancellation = Cancellation::new()?;
    let mut config = config::Config::load()?;
    // A legacy community URL must not prevent a local measurement.
    let mut benchmark_config = config.clone();
    benchmark_config.server_url = None;
    benchmark_config.validate()?;
    let repeats = repeats.or(config.default_repeats).unwrap_or(3);
    anyhow::ensure!(
        (1..=1000).contains(&repeats),
        "Measurement repeats must be between 1 and 1000"
    );
    let prefill_context = prefill_context
        .map(usize::from)
        .or(config.default_prefill_context)
        .unwrap_or(0);
    anyhow::ensure!(prefill_context <= 1, "Prefill context must be 0 or 1");
    let context_lengths = context_lengths.or_else(|| config.default_context_lengths.clone());
    // Extract server path and flags from passthrough
    let (server_path, flags) = if passthrough.is_empty() {
        anyhow::bail!(
            "No command provided. Pass your llama-server command after --passthrough.\n\
             Example: autarky bench --passthrough ./llama-server --model model.gguf"
        );
    } else {
        let mut flags = config.extra_flags.clone().unwrap_or_default();
        flags.extend_from_slice(&passthrough[1..]);
        (passthrough[0].clone(), flags)
    };

    // Parse GGUF metadata if --model flag is present
    let model_path = parse_model_flag(&flags);

    // Get model info for context clamping and display
    let model_info = model_path
        .as_ref()
        .and_then(|path| autarky::gguf::parse_gguf(path).ok());
    let model_ctx_tokens = model_info
        .as_ref()
        .and_then(|info| info.context_length)
        .map(|length| length as usize)
        .filter(|&tokens| tokens > 0);
    let model_label = model_info
        .as_ref()
        .map(|info| {
            let mut label = info.model_name.clone().unwrap_or_else(|| "unknown".into());
            if info.is_moe {
                let cpu_layers = info
                    .block_count
                    .and_then(|count| extract_cpu_moe_layers(&flags, count));
                if let Some(n) = cpu_layers {
                    label.push_str(&format!(" (moe - {} CPU layers)", n));
                } else {
                    label.push_str(" (moe)");
                }
            } else if info.expert_count == Some(0) {
                label.push_str(" (dense)");
            }
            label
        })
        .or_else(|| {
            model_path.as_ref().and_then(|p| {
                std::path::Path::new(p)
                    .file_name()?
                    .to_str()
                    .map(|s| s.to_string())
            })
        })
        .unwrap_or_else(|| "unknown".to_string());

    let lengths = resolve_context_lengths(context_lengths.as_deref(), model_ctx_tokens)?;

    // Validate all benchmark inputs before invoking any subprocess.
    let mut dashboard = None;
    let llama_server_version = drive_with_controls(
        autarky::launcher::get_server_version(&server_path),
        &mut dashboard,
        &mut cancellation,
    )
    .await?
    .context("Benchmark cancelled before version discovery")?;

    // Detect hardware
    let hardware = drive_with_controls(
        hardware::detect_hardware_for_server(&server_path, &flags),
        &mut dashboard,
        &mut cancellation,
    )
    .await?
    .context("Benchmark cancelled before hardware discovery")?;

    // All fallible measurement-phase work has one exit: stop, restore, save.
    let total = lengths.len();
    let mut results = Vec::new();
    let mut server = None;
    let mut current_context = lengths[0];
    let measurement: Result<()> = async {
        if tui {
            dashboard = Some(autarky::tui::Dashboard::new()?);
            let dash = dashboard.as_mut().unwrap();
            dash.set_total(total);
            dash.set_model(terminal_text(&model_label));
            dash.set_hardware(hardware.clone());
            dash.state.status_message = "Loading model...".into();
            dash.draw()?;
        } else {
            write_stdout("⏳ Loading model...")?;
        }
        server = drive_with_controls(
            launcher::launch_server(&server_path, &flags),
            &mut dashboard,
            &mut cancellation,
        )
        .await?;
        let server = server
            .as_mut()
            .context("Benchmark cancelled during startup")?;
        drive_with_controls(server.wait_ready(), &mut dashboard, &mut cancellation)
            .await?
            .context("Benchmark cancelled during startup")?;
        if !tui {
            write_stdout(" ✓\n")?;
        }

        for (i, &ctx_len) in lengths.iter().enumerate() {
            current_context = ctx_len;
            if let Some(dash) = dashboard.as_mut() {
                dash.state.current_ctx = ctx_len;
                dash.state.status_message = format!("Benchmarking context length {ctx_len}...");
                dash.state
                    .start_time
                    .get_or_insert_with(std::time::Instant::now);
                dash.draw()?;
            } else {
                write_stdout(&format!(
                    "⏳ [{}/{}] Benchmarking context length {}k...",
                    i + 1,
                    total,
                    ctx_len
                ))?;
            }
            let outcome = drive_with_controls(
                server.benchmark(ctx_len, repeats, prefill_context, model_ctx_tokens),
                &mut dashboard,
                &mut cancellation,
            )
            .await;
            let metric = match outcome {
                Ok(Some(metric)) => match metric.validate() {
                    Ok(()) => metric,
                    Err(error) => BenchmarkResult::error(ctx_len, error.to_string()),
                },
                Ok(None) => BenchmarkResult::cancelled(ctx_len),
                Err(error) => BenchmarkResult::error(ctx_len, error.to_string()),
            };
            // Retain the measurement BEFORE any output, including TUI redraws.
            results.push(metric);
            let metric = results.last().unwrap();
            if let Some(dash) = dashboard.as_mut() {
                dash.state.completed += 1;
                dash.state.results.push(metric.clone());
                dash.draw()?;
            } else if metric.status == "success" {
                write_stdout(&format!(
                    "\r   ✓ {}k Context Complete: {:.1} TPS, {:.1} Prompt/s\n",
                    ctx_len, metric.tokens_per_second, metric.prompt_per_second
                ))?;
            } else {
                write_stdout(&format!(
                    "\r   ✗ {}k: {}\n",
                    ctx_len,
                    terminal_text(&metric.status)
                ))?;
            }
            anyhow::ensure!(
                metric.status == "success",
                "Benchmark {} at {}k",
                metric.status,
                ctx_len
            );
        }
        Ok(())
    }
    .await;

    // Always stop and restore the terminal before the persistent report.
    let shutdown = match server.as_mut() {
        Some(server) => server.kill().await,
        None => Ok(()),
    };
    let terminal_cleanup = dashboard.as_mut().map(|dash| dash.cleanup()).transpose();
    drop(dashboard);
    let mut failure = measurement
        .err()
        .or_else(|| shutdown.err())
        .or_else(|| terminal_cleanup.err());
    // Include an interrupted context, but never replace an already valid point
    // with an output error or create duplicate context measurements.
    if let Some(error) = failure.as_ref() {
        if !results
            .iter()
            .any(|metric| metric.context_length == current_context)
        {
            results.push(if cancellation.cancelled {
                BenchmarkResult::cancelled(current_context)
            } else {
                BenchmarkResult::error(current_context, error.to_string())
            });
        }
    }
    if let Err(error) = cancellation.check().await {
        failure.get_or_insert(error);
    }
    let failure_text = failure.as_ref().map(ToString::to_string);
    let outcome = if cancellation.cancelled {
        db::RunOutcome::Cancelled
    } else if let Some(reason) = failure_text.as_deref() {
        db::RunOutcome::Failed(reason)
    } else {
        db::RunOutcome::Completed
    };
    let bench_id = db::store_benchmark_run(
        &hardware,
        &model_path,
        model_info.as_ref(),
        &db::RunMeasurements {
            results: &results,
            outcome,
        },
        server
            .as_ref()
            .map_or(flags.as_slice(), |server| &server.full_flags),
        &llama_server_version,
    )?;
    let report =
        autarky::report::render_report(&hardware, &model_label, &results, &llama_server_version);
    let output = write_stdout(&format!("{report}Saved benchmark #{bench_id}\n"));
    if let Some(error) = failure {
        return Err(error.context(format!("Partial results saved as benchmark #{bench_id}")));
    }
    output?;
    let summary = monitor::summarize_results(&results)?;

    // Auto-submit to community server (best-effort, respects --offline)
    let sharing_endpoint = if !offline && config.server_url.is_some() && summary.success_count > 0 {
        match config.get_server_url() {
            Ok(url) => Some(url),
            Err(error) => {
                writeln!(
                    std::io::stderr(),
                    "Community sharing skipped: {} (local results are saved)",
                    terminal_text(&error.to_string())
                )?;
                None
            }
        }
    } else {
        None
    };
    if let Some(server_url) = sharing_endpoint {
        cancellation.check().await?;
        write_stdout("  📡 Submitting to community server... ")?;
        let submit_result = async {
            let db_path = db::get_db_path()?;
            submit_benchmark(
                &db_path,
                bench_id,
                &server_url,
                &config.get_or_create_api_key()?,
            )
            .await
        };
        let submit_result = drive_with_controls(submit_result, &mut None, &mut cancellation).await;
        cancellation.check().await?;
        match submit_result {
            Ok(Some(id)) => write_stdout(&format!("✓ (ID #{id})\n"))?,
            Ok(None) => anyhow::bail!("Benchmark cancelled; local results are saved"),
            Err(e) => write_stdout(&format!(
                "✗ {} (local results are saved)\n",
                terminal_text(&e.to_string())
            ))?,
        };

        // After submission, check community for similar benchmarks we can compare against
        if let Some((name, quant)) = model_info.as_ref().and_then(|info| {
            known_identity(info.model_name.as_deref(), info.quantization.as_deref())
        }) {
            cancellation.check().await?;
            write_stdout("  🔍 Checking community benchmarks... ")?;
            let comparison = drive_with_controls(
                check_community_comparison(
                    &server_url,
                    &hardware,
                    name,
                    quant,
                    &model_info
                        .as_ref()
                        .map(ModelDescriptors::from_info)
                        .unwrap_or_default(),
                    &results,
                    config.api_key.as_deref(),
                ),
                &mut None,
                &mut cancellation,
            )
            .await;
            cancellation.check().await?;
            match comparison {
                Ok(Some(Some((entry, ctx_k, metric_label, quant_label)))) => {
                    write_stdout(&format!("\n  ⚡ {quant_label}\n"))?;
                    let answer = prompt_yes_no(
                        &format!("    {} context at {}. Want to see it?", ctx_k, metric_label),
                        &mut cancellation,
                    )
                    .await?;
                    if answer {
                        show_benchmark_details(&entry)?;
                    }
                }
                Ok(Some(None)) => {
                    write_stdout("none found\n")?;
                }
                Ok(None) => anyhow::bail!("Benchmark cancelled; local results are saved"),
                Err(e) => {
                    write_stdout(&format!("✗ {}\n", terminal_text(&e.to_string())))?;
                }
            }
        }
    }

    cancellation.check().await?;
    anyhow::ensure!(
        summary.success_count > 0,
        "No successful benchmark results; saved as benchmark #{bench_id}"
    );
    Ok(())
}

fn write_stdout(text: &str) -> Result<()> {
    let mut output = std::io::stdout().lock();
    output.write_all(text.as_bytes())?;
    output.flush()?;
    Ok(())
}

/// One owned signal stream spans startup, measurements, and community work.
struct Cancellation {
    #[cfg(unix)]
    signal: tokio::signal::unix::Signal,
    #[cfg(windows)]
    signal: tokio::signal::windows::CtrlC,
    cancelled: bool,
}

impl Cancellation {
    fn new() -> Result<Self> {
        #[cfg(unix)]
        let signal = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
        #[cfg(windows)]
        let signal = tokio::signal::windows::ctrl_c()?;
        Ok(Self {
            signal,
            cancelled: false,
        })
    }

    async fn wait(&mut self) -> Result<()> {
        if !self.cancelled {
            self.signal.recv().await.context("Ctrl+C listener closed")?;
            self.cancelled = true;
        }
        Ok(())
    }

    async fn check(&mut self) -> Result<()> {
        tokio::select! {
            biased;
            signal = self.wait() => {
                signal?;
                anyhow::bail!("Benchmark cancelled; any saved local results are retained");
            }
            _ = std::future::ready(()) => Ok(()),
        }
    }
}

fn dashboard_quit() -> Result<bool> {
    use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
    while event::poll(std::time::Duration::ZERO)? {
        if let Event::Key(key) = event::read()? {
            if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat)
                && (key.code == KeyCode::Char('q')
                    || matches!(key.code, KeyCode::Char('c' | 'C'))
                        && key.modifiers.contains(KeyModifiers::CONTROL))
            {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Keep benchmark/startup work active while input and redraws stay responsive.
async fn drive_with_controls<F, T>(
    future: F,
    dashboard: &mut Option<autarky::tui::Dashboard>,
    cancellation: &mut Cancellation,
) -> Result<Option<T>>
where
    F: std::future::Future<Output = Result<T>>,
{
    tokio::pin!(future);
    let mut redraw = tokio::time::interval(std::time::Duration::from_millis(50));
    loop {
        tokio::select! {
            biased;
            signal = cancellation.wait() => {
                signal?;
                return Ok(None);
            }
            result = &mut future => return result.map(Some),
            _ = redraw.tick(), if dashboard.is_some() => {
                let dash = dashboard.as_mut().expect("dashboard is present");
                if dashboard_quit()? {
                    cancellation.cancelled = true;
                    return Ok(None);
                }
                dash.draw()?;
            }
        }
    }
}

fn resolve_context_lengths(
    input: Option<&str>,
    model_context: Option<usize>,
) -> Result<Vec<usize>> {
    let model_max_k = model_context
        .filter(|&tokens| tokens > 0)
        .map(|tokens| tokens / 1000);
    let supported: Vec<usize> = launcher::SUPPORTED_CONTEXT_LENGTHS
        .iter()
        .copied()
        .filter(|&k| model_max_k.is_none_or(|max_k| k <= max_k))
        .collect();
    launcher::validate_context_lengths(&supported)?;
    let lengths = if let Some(input) = input {
        let parsed: Vec<usize> = input
            .split(',')
            .map(|part| part.trim().parse())
            .collect::<Result<Vec<_>, _>>()?;
        let max_supported = *supported.last().unwrap();
        let clamped: Vec<usize> = parsed
            .iter()
            .map(|&k| {
                if model_max_k.is_some() {
                    k.min(max_supported)
                } else {
                    k
                }
            })
            .collect();
        if clamped != parsed {
            write_stdout(&format!("⚠️  Some context lengths adjusted to fit supported prompts (max {max_supported}k).\n"))?;
        }
        let mut deduped = clamped;
        deduped.sort_unstable();
        deduped.dedup();
        deduped
    } else {
        supported
    };
    launcher::validate_context_lengths(&lengths)?;
    Ok(lengths)
}

async fn submit_benchmark(
    db_path: &str,
    bench_id: i64,
    server_url: &str,
    api_key: &str,
) -> Result<i64> {
    let payload = {
        let conn = rusqlite::Connection::open(db_path)?;
        let mut stmt = conn.prepare(
            "SELECT br.flags_used, br.context_lengths_tested, br.raw_metrics, br.oom_boundary,
                    hf.gpu_family, hf.vram_gb, hf.system_ram_gb, hf.os_platform,
                    mf.model_name, mf.quantization, mf.architecture_type, mf.expert_count, mf.parameters_billions,
                    br.llama_server_version
             FROM benchmark_runs br
             LEFT JOIN hardware_fingerprints hf ON br.hardware_id = hf.id
             LEFT JOIN model_fingerprints mf ON br.model_id = mf.id
             WHERE br.id = ?1"
        )?;
        let (mut payload, flags_json, ctx_json, raw_json) = stmt.query_row([bench_id], |row| {
            let flags_json: String = row.get(0)?;
            let ctx_json: String = row.get(1)?;
            let raw_json: String = row.get(2)?;
            let model_name: Option<String> = row.get(8)?;
            let model_name = model_name.filter(|name| !name.trim().is_empty());
            Ok((
                serde_json::json!({
                    "hardware": {
                        "gpu_family": row.get::<_, String>(4)?,
                        "vram_gb": row.get::<_, f64>(5)?,
                        "system_ram_gb": row.get::<_, u64>(6)?,
                        "os_platform": row.get::<_, String>(7)?,
                    },
                    "model": model_name.map(|name| -> rusqlite::Result<serde_json::Value> {
                        Ok(serde_json::json!({
                            "model_name": name,
                            "quantization": row.get::<_, Option<String>>(9)?,
                            "architecture_type": row.get::<_, Option<String>>(10)?,
                            "expert_count": row.get::<_, Option<i64>>(11)?,
                            "parameters_billions": row.get::<_, Option<f64>>(12)?,
                        }))
                    }).transpose()?,
                    "oom_boundary": row.get::<_, Option<i64>>(3)?,
                    "llama_server_version": row.get::<_, Option<String>>(13)?,
                }),
                flags_json,
                ctx_json,
                raw_json,
            ))
        })?;
        let flags: Vec<String> =
            serde_json::from_str(&flags_json).context("Invalid saved flags JSON")?;
        let contexts: Vec<usize> =
            serde_json::from_str(&ctx_json).context("Invalid saved contexts JSON")?;
        let mut metrics: Vec<BenchmarkResult> =
            serde_json::from_str(&raw_json).context("Invalid saved metrics JSON")?;
        let summary = monitor::summarize_results(&metrics)?;
        anyhow::ensure!(
            summary.success_count > 0,
            "No successful measurements to submit"
        );
        anyhow::ensure!(
            contexts
                == metrics
                    .iter()
                    .map(|metric| metric.context_length)
                    .collect::<Vec<_>>(),
            "Saved contexts do not match saved metrics"
        );
        // Error bodies and prompt text stay private, including in older rows.
        // Deserialize into the measurement schema to omit unknown/private fields.
        for metric in &mut metrics {
            metric.error = None;
        }
        payload["flags_used"] = serde_json::to_value(db::sanitize_flags(&flags))?;
        payload["context_lengths_tested"] = serde_json::to_value(contexts)?;
        payload["raw_metrics"] = serde_json::to_value(metrics)?;
        payload
    };

    let client = community::client()?;
    let resp = client
        .post(community::endpoint(server_url, "api/v1/benchmark")?)
        .header("X-API-Key", community::key_header(api_key)?)
        .json(&payload)
        .send()
        .await?;
    let body = community::response_json(resp).await?;
    body["id"]
        .as_i64()
        .filter(|id| *id > 0)
        .context("Community server did not return a valid submission ID")
}

/// Check if --cpu-moe / -cmoe or --n-cpu-moe / -ncmoe was passed to
/// llama-server. Returns Some(n) where n is the number of MoE expert
/// layers offloaded to CPU, or None if neither flag was set.
///
/// For --cpu-moe / -cmoe (no argument): returns the model's total
/// block_count (all layers).
///
/// For --n-cpu-moe N / -ncmoe N: returns N.
fn extract_cpu_moe_layers(flags: &[String], block_count: u32) -> Option<u32> {
    let parsed = autarky::arguments::parse(flags).ok()?;
    parsed
        .iter()
        .filter_map(|arg| match arg.key.as_str() {
            "-cmoe" | "--cpu-moe" => Some(block_count),
            "-ncmoe" | "--n-cpu-moe" => arg.values.first().and_then(|value| value.parse().ok()),
            _ => None,
        })
        .max()
}

fn date_range_arg(input: &str) -> std::result::Result<String, String> {
    parse_date_range(Some(input))
        .map(|_| input.to_string())
        .map_err(|error| error.to_string())
}

/// Date filters must be valid before querying or opening an export destination.
fn parse_date_range(input: Option<&str>) -> Result<Option<(String, String)>> {
    fn valid_date(value: &str) -> bool {
        let bytes = value.as_bytes();
        if bytes.len() != 10
            || bytes[4] != b'-'
            || bytes[7] != b'-'
            || bytes
                .iter()
                .enumerate()
                .any(|(i, b)| i != 4 && i != 7 && !b.is_ascii_digit())
        {
            return false;
        }
        let year = value[..4].parse::<u32>().unwrap();
        let month = value[5..7].parse::<usize>().unwrap();
        let day = value[8..].parse::<u32>().unwrap();
        let leap =
            year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
        let days = [
            31,
            if leap { 29 } else { 28 },
            31,
            30,
            31,
            30,
            31,
            31,
            30,
            31,
            30,
            31,
        ];
        year > 0 && (1..=12).contains(&month) && day > 0 && day <= days[month - 1]
    }
    input
        .map(|input| {
            let parts: Vec<_> = input.split(',').map(str::trim).collect();
            anyhow::ensure!(
                parts.len() == 2 && parts.iter().all(|part| valid_date(part)),
                "Date range must contain exactly two valid YYYY-MM-DD dates"
            );
            anyhow::ensure!(
                parts[0] <= parts[1],
                "Date range start must not follow its end"
            );
            Ok((parts[0].to_string(), parts[1].to_string()))
        })
        .transpose()
}

#[derive(Debug, serde::Serialize)]
struct HistoryEntry {
    id: i64,
    timestamp: String,
    model: Option<String>,
    quantization: Option<String>,
    #[serde(flatten)]
    descriptors: ModelDescriptors,
    gpu: String,
    vram_gb: f64,
    system_ram_gb: u64,
    os_platform: String,
    oom_boundary: Option<i64>,
    oom: bool,
    status: String,
    failure_reason: Option<String>,
    raw_metrics: serde_json::Value,
    context_lengths_tested: Vec<usize>,
    #[serde(flatten)]
    summary: monitor::MetricSummary,
}

const HISTORY_SELECT: &str = "SELECT br.id, br.timestamp, mf.model_name, mf.quantization,
    hf.gpu_family, hf.vram_gb, hf.system_ram_gb, hf.os_platform, br.oom_boundary, br.oom, br.raw_metrics,
    br.status, br.failure_reason, mf.architecture, mf.architecture_type, mf.expert_count,
    mf.parameters_billions, mf.block_count, mf.context_length
    FROM benchmark_runs br JOIN hardware_fingerprints hf ON br.hardware_id = hf.id
    LEFT JOIN model_fingerprints mf ON br.model_id = mf.id WHERE 1=1";

fn history_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<HistoryEntry> {
    Ok(HistoryEntry {
        id: row.get(0)?,
        timestamp: row.get(1)?,
        model: row.get(2)?,
        quantization: row.get(3)?,
        descriptors: ModelDescriptors {
            architecture: row.get(13)?,
            architecture_type: row.get(14)?,
            expert_count: row.get(15)?,
            parameters_billions: row.get(16)?,
            block_count: row.get(17)?,
            context_length: row.get(18)?,
        },
        gpu: row.get(4)?,
        vram_gb: row.get(5)?,
        system_ram_gb: row.get(6)?,
        os_platform: row.get(7)?,
        oom_boundary: row.get(8)?,
        oom: row.get(9)?,
        status: row.get(11)?,
        failure_reason: row.get(12)?,
        // JSON and metric errors are checked outside the SQLite row mapper.
        raw_metrics: serde_json::Value::String(row.get(10)?),
        context_lengths_tested: Vec::new(),
        summary: monitor::MetricSummary::default(),
    })
}

fn query_history(
    conn: &rusqlite::Connection,
    query: &str,
    params: &[rusqlite::types::Value],
) -> Result<Vec<HistoryEntry>> {
    let mut statement = conn.prepare(query)?;
    let rows = statement.query_map(rusqlite::params_from_iter(params), history_row)?;
    let mut entries = Vec::new();
    for row in rows {
        let mut entry = row?;
        entry.raw_metrics = serde_json::from_str(
            entry
                .raw_metrics
                .as_str()
                .context("Missing saved metrics")?,
        )
        .with_context(|| format!("Invalid metrics JSON in benchmark {}", entry.id))?;
        let metrics = monitor::parse_metrics(&entry.raw_metrics)?;
        entry.summary = monitor::summarize_results(&metrics)?;
        entry.context_lengths_tested = metrics.iter().map(|metric| metric.context_length).collect();
        entries.push(entry);
    }
    Ok(entries)
}

fn history_filters(
    query: &mut String,
    params: &mut Vec<rusqlite::types::Value>,
    model: Option<&str>,
    gpu: Option<&str>,
    quant: Option<&str>,
) {
    for (column, value) in [
        ("mf.model_name", model),
        ("hf.gpu_family", gpu),
        ("mf.quantization", quant),
    ] {
        if let Some(value) = value {
            query.push_str(&format!(" AND {column} LIKE ?"));
            params.push(format!("%{value}%").into());
        }
    }
}

fn filtered_history(
    model: Option<&str>,
    gpu: Option<&str>,
    quant: Option<&str>,
    dates: Option<&str>,
) -> Result<Vec<HistoryEntry>> {
    let dates = parse_date_range(dates)?;
    let conn = rusqlite::Connection::open(db::get_db_path()?)?;
    let mut query = HISTORY_SELECT.to_string();
    let mut params = Vec::new();
    history_filters(&mut query, &mut params, model, gpu, quant);
    if let Some((start, end)) = dates {
        query.push_str(" AND DATE(br.timestamp) >= ? AND DATE(br.timestamp) <= ?");
        params.extend([start.into(), end.into()]);
    }
    query.push_str(" ORDER BY br.timestamp DESC, br.id DESC");
    query_history(&conn, &query, &params)
}

fn cell(text: &str, width: usize) -> String {
    autarky::utils::pad_display(&autarky::utils::truncate_display(text, width), width)
}

fn run_list(
    model: Option<String>,
    gpu: Option<String>,
    quant: Option<String>,
    date_range: Option<String>,
) -> Result<()> {
    let entries = filtered_history(
        model.as_deref(),
        gpu.as_deref(),
        quant.as_deref(),
        date_range.as_deref(),
    )?;
    println!("\nBenchmark History:");
    println!(
        "ID     Timestamp            Model                Quant      GPU                  OOM@ Status"
    );
    for entry in &entries {
        println!(
            "{:<6} {} {} {} {} {} {}",
            entry.id,
            cell(&entry.timestamp, 20),
            cell(entry.model.as_deref().unwrap_or("unknown"), 20),
            cell(entry.quantization.as_deref().unwrap_or("unknown"), 10),
            cell(&format!("{} ({:.1}GB)", entry.gpu, entry.vram_gb), 20),
            entry
                .oom_boundary
                .map_or_else(|| "-".to_string(), |value| format!("{value}k")),
            terminal_text(&entry.status)
        );
    }
    println!("Total: {} benchmark(s)", entries.len());
    Ok(())
}

fn run_export(
    format: String,
    file: Option<String>,
    model: Option<String>,
    gpu: Option<String>,
    quant: Option<String>,
    date_range: Option<String>,
) -> Result<()> {
    let entries = filtered_history(
        model.as_deref(),
        gpu.as_deref(),
        quant.as_deref(),
        date_range.as_deref(),
    )?;
    let output = if format == "csv" {
        let mut writer = csv::Writer::from_writer(Vec::new());
        writer.write_record([
            "id",
            "timestamp",
            "model",
            "quantization",
            "gpu",
            "vram_gb",
            "oom_boundary",
            "oom",
            "avg_tps",
            "avg_prompt_per_sec",
            "context_lengths",
            "status",
            "failure_reason",
        ])?;
        for entry in &entries {
            writer.write_record([
                entry.id.to_string(),
                entry.timestamp.clone(),
                entry.model.clone().unwrap_or_default(),
                entry.quantization.clone().unwrap_or_default(),
                entry.gpu.clone(),
                entry.vram_gb.to_string(),
                entry
                    .oom_boundary
                    .map_or_else(String::new, |value| value.to_string()),
                entry.oom.to_string(),
                entry.summary.avg_tps.to_string(),
                entry.summary.avg_prompt.to_string(),
                serde_json::to_string(&entry.context_lengths_tested)?,
                entry.status.clone(),
                entry.failure_reason.clone().unwrap_or_default(),
            ])?;
        }
        writer.flush()?;
        String::from_utf8(writer.into_inner()?)?
    } else {
        serde_json::to_string_pretty(&entries)?
    };
    if let Some(path) = file {
        std::fs::write(&path, output)?;
        eprintln!(
            "Exported {} benchmark(s) to {}",
            entries.len(),
            terminal_text(&path)
        );
    } else {
        println!("{output}");
    }
    Ok(())
}

fn load_benchmark(conn: &rusqlite::Connection, id: i64) -> Result<HistoryEntry> {
    query_history(
        conn,
        &format!("{HISTORY_SELECT} AND br.id = ?"),
        &[id.into()],
    )?
    .pop()
    .with_context(|| format!("Benchmark {id} not found"))
}

fn load_previous(
    conn: &rusqlite::Connection,
    current_id: i64,
    model: Option<String>,
    gpu: Option<String>,
) -> Result<HistoryEntry> {
    let mut query = format!("{HISTORY_SELECT} AND br.id < ?");
    let mut params = vec![current_id.into()];
    history_filters(
        &mut query,
        &mut params,
        model.as_deref(),
        gpu.as_deref(),
        None,
    );
    query.push_str(" ORDER BY br.id DESC LIMIT 1");
    query_history(conn, &query, &params)?
        .pop()
        .context("No previous matching benchmark")
}

fn matching_workload(left: &HistoryEntry, right: &HistoryEntry) -> bool {
    let identity = known_identity(left.model.as_deref(), left.quantization.as_deref());
    identity.is_some()
        && identity == known_identity(right.model.as_deref(), right.quantization.as_deref())
        && left.descriptors.compatible(&right.descriptors)
}

/// Required name/quantization are only a minimum. An available descriptor must
/// not contradict its counterpart; absence never becomes a fabricated zero.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
struct ModelDescriptors {
    architecture: Option<String>,
    architecture_type: Option<String>,
    expert_count: Option<i64>,
    parameters_billions: Option<f64>,
    block_count: Option<u32>,
    context_length: Option<u32>,
}

impl ModelDescriptors {
    fn from_info(info: &autarky::gguf::GgufInfo) -> Self {
        Self {
            architecture: info.architecture.clone(),
            architecture_type: info
                .expert_count
                .map(|count| if count > 0 { "moe" } else { "dense" }.into()),
            expert_count: info.expert_count.map(i64::from),
            parameters_billions: info.parameters_billions,
            block_count: info.block_count,
            context_length: info.context_length,
        }
    }

    fn compatible(&self, other: &Self) -> bool {
        fn agrees<T: PartialEq>(left: &Option<T>, right: &Option<T>) -> bool {
            left.as_ref()
                .zip(right.as_ref())
                .is_none_or(|(left, right)| left == right)
        }
        agrees(&self.architecture, &other.architecture)
            && agrees(&self.architecture_type, &other.architecture_type)
            && agrees(&self.expert_count, &other.expert_count)
            && agrees(&self.parameters_billions, &other.parameters_billions)
            && agrees(&self.block_count, &other.block_count)
            && agrees(&self.context_length, &other.context_length)
    }
}

fn known_identity<'a>(
    name: Option<&'a str>,
    quantization: Option<&'a str>,
) -> Option<(&'a str, &'a str)> {
    name.filter(|name| !name.trim().is_empty())
        .zip(quantization.filter(|quant| !quant.trim().is_empty()))
}

fn display_comparison(left: &HistoryEntry, right: &HistoryEntry) -> Result<()> {
    println!("\nBenchmark Comparison");
    for (number, entry) in [(1, left), (2, right)] {
        println!(
            "Benchmark {number}: ID {} ({})",
            entry.id,
            terminal_text(&entry.timestamp)
        );
        println!(
            "  Model: {} ({})",
            terminal_text(entry.model.as_deref().unwrap_or("unknown")),
            terminal_text(entry.quantization.as_deref().unwrap_or("unknown"))
        );
        println!("  GPU: {}", terminal_text(&entry.gpu));
    }
    if !matching_workload(left, right) {
        println!("Different or unknown workloads: no relative performance claim.");
        println!(
            "Avg Tokens/s: {:.1} / {:.1}",
            left.summary.avg_tps, right.summary.avg_tps
        );
        println!(
            "Avg Prompt/s: {:.1} / {:.1}",
            left.summary.avg_prompt, right.summary.avg_prompt
        );
        return Ok(());
    }
    let Some(comparison) = monitor::paired_metrics(&left.raw_metrics, &right.raw_metrics)? else {
        println!("No matching successful contexts; comparison unavailable.");
        return Ok(());
    };
    println!("Matched successful contexts (K): {:?}", comparison.contexts);
    for (label, left, right, delta) in [
        (
            "Avg Tokens/s",
            comparison.left.avg_tps,
            comparison.right.avg_tps,
            comparison.tps_delta,
        ),
        (
            "Avg Prompt/s",
            comparison.left.avg_prompt,
            comparison.right.avg_prompt,
            comparison.prompt_delta,
        ),
    ] {
        let delta = delta.map_or_else(|| "N/A".to_string(), |value| format!("{value:+.1}%"));
        println!("{label}: {left:.1} / {right:.1} (benchmark 1 vs 2: {delta})");
    }
    Ok(())
}

fn run_compare(
    id1: Option<i64>,
    id2: Option<i64>,
    model: Option<String>,
    gpu: Option<String>,
    similar: bool,
) -> Result<()> {
    let conn = rusqlite::Connection::open(db::get_db_path()?)?;
    let mut query = HISTORY_SELECT.to_string();
    let mut params = Vec::new();
    history_filters(
        &mut query,
        &mut params,
        model.as_deref(),
        gpu.as_deref(),
        None,
    );
    query.push_str(" ORDER BY br.timestamp DESC, br.id DESC LIMIT 2");
    let mut matches = if id1.is_none() {
        query_history(&conn, &query, &params)?
    } else {
        Vec::new()
    }
    .into_iter();
    let left = if let Some(id) = id1 {
        load_benchmark(&conn, id)?
    } else {
        matches.next().context("No matching benchmark")?
    };
    let right = if similar {
        anyhow::ensure!(
            known_identity(left.model.as_deref(), left.quantization.as_deref()).is_some(),
            "Similar comparison requires known model and quantization"
        );
        let candidates = query_history(&conn, &format!("{HISTORY_SELECT} AND br.id != ? AND mf.model_name = ? AND mf.quantization = ? AND hf.gpu_family = ? AND hf.vram_gb = ? AND hf.system_ram_gb = ? AND hf.os_platform = ? ORDER BY br.timestamp DESC, br.id DESC"),
            &[left.id.into(), left.model.clone().unwrap().into(), left.quantization.clone().unwrap().into(), left.gpu.clone().into(), left.vram_gb.into(), i64::try_from(left.system_ram_gb)?.into(), left.os_platform.clone().into()])?;
        let mut eligible = None;
        for candidate in candidates {
            if matching_workload(&left, &candidate)
                && monitor::paired_metrics(&left.raw_metrics, &candidate.raw_metrics)?.is_some()
            {
                eligible = Some(candidate);
                break;
            }
        }
        eligible.context("No similar hardware benchmark with the same model, quantization and successful contexts")?
    } else if let Some(id) = id2 {
        load_benchmark(&conn, id)?
    } else if id1.is_some() {
        load_previous(&conn, left.id, model, gpu)?
    } else {
        matches
            .next()
            .context("Need at least 2 matching benchmarks to compare")?
    };
    display_comparison(&left, &right)
}

#[derive(Debug, serde::Deserialize)]
struct CommunityEntry {
    id: i64,
    submitted_at: String,
    model_name: Option<String>,
    quantization: Option<String>,
    #[serde(flatten)]
    descriptors: ModelDescriptors,
    gpu_family: String,
    vram_gb: f64,
    system_ram_gb: u64,
    os_platform: String,
    raw_metrics: serde_json::Value,
}

impl CommunityEntry {
    fn into_history(self) -> Result<HistoryEntry> {
        anyhow::ensure!(
            self.id > 0 && self.vram_gb.is_finite() && self.vram_gb >= 0.0,
            "Invalid community benchmark identity"
        );
        let metrics = monitor::parse_metrics(&self.raw_metrics)?;
        let summary = monitor::summarize_results(&metrics)?;
        let status = if metrics.is_empty() {
            "unknown"
        } else if metrics.iter().any(|metric| metric.status == "cancelled") {
            "cancelled"
        } else if metrics.iter().any(|metric| metric.status != "success") {
            "failed"
        } else {
            "completed"
        };
        Ok(HistoryEntry {
            id: self.id,
            timestamp: self.submitted_at,
            model: self.model_name,
            quantization: self.quantization,
            descriptors: self.descriptors,
            gpu: self.gpu_family,
            vram_gb: self.vram_gb,
            system_ram_gb: self.system_ram_gb,
            os_platform: self.os_platform,
            oom_boundary: metrics
                .iter()
                .filter(|metric| metric.status == "oom")
                .map(|metric| metric.context_length as i64)
                .min(),
            oom: metrics.iter().any(|metric| metric.status == "oom"),
            context_lengths_tested: metrics.iter().map(|metric| metric.context_length).collect(),
            raw_metrics: self.raw_metrics,
            summary,
            status: status.into(),
            failure_reason: None,
        })
    }
}

async fn run_compare_remote(
    id1: Option<i64>,
    id2: Option<i64>,
    _model: Option<String>,
    _gpu: Option<String>,
) -> Result<()> {
    let (id1, id2) = id1
        .zip(id2)
        .context("Remote compare requires both --id1 and --id2")?;
    let config = config::Config::load()?;
    let server_url = config.get_server_url()?;
    let response = community::client()?
        .post(community::endpoint(&server_url, "api/v1/compare")?)
        .json(&serde_json::json!({"id1": id1, "id2": id2}))
        .send()
        .await
        .context("Failed to connect to community server")?;
    let data = community::response_json(response).await?;
    let left: CommunityEntry = serde_json::from_value(data["benchmark_1"].clone())
        .context("Invalid first remote benchmark")?;
    let right: CommunityEntry = serde_json::from_value(data["benchmark_2"].clone())
        .context("Invalid second remote benchmark")?;
    anyhow::ensure!(
        left.id == id1 && right.id == id2,
        "Remote comparison returned different benchmark IDs"
    );
    println!("Remote comparison (server: {})", terminal_text(&server_url));
    display_comparison(&left.into_history()?, &right.into_history()?)
}

fn run_leaderboard(
    period: String,
    model: Option<String>,
    gpu: Option<String>,
    sort: String,
    top: usize,
    group_by: String,
) -> Result<()> {
    let conn = rusqlite::Connection::open(db::get_db_path()?)?;
    let mut query = HISTORY_SELECT.to_string();
    let mut params = Vec::new();
    history_filters(
        &mut query,
        &mut params,
        model.as_deref(),
        gpu.as_deref(),
        None,
    );
    if period != "all" {
        let offset = match period.as_str() {
            "24h" => "-1 day",
            "7d" => "-7 days",
            "30d" => "-30 days",
            _ => anyhow::bail!("Invalid leaderboard period"),
        };
        query.push_str(" AND br.timestamp >= datetime('now', ?)");
        params.push(offset.to_string().into());
    }
    query.push_str(" ORDER BY br.timestamp DESC, br.id DESC");
    let mut entries = query_history(&conn, &query, &params)?;
    entries.retain(|entry| entry.summary.success_count > 0);
    entries.sort_by(|a, b| {
        let order = match sort.as_str() {
            "prompt" => b.summary.avg_prompt.total_cmp(&a.summary.avg_prompt),
            "context" | "ctx" => b.summary.max_context.cmp(&a.summary.max_context),
            _ => b.summary.avg_tps.total_cmp(&a.summary.avg_tps),
        };
        order.then_with(|| b.id.cmp(&a.id))
    });
    let mut seen = std::collections::HashSet::new();
    entries.retain(|entry| {
        let key = match group_by.as_str() {
            "model" => format!("{:?}/{:?}", entry.model, entry.quantization),
            "gpu" => entry.gpu.clone(),
            "hardware" => format!(
                "{:?}/{:?}/{}/{}",
                entry.gpu, entry.vram_gb, entry.system_ram_gb, entry.os_platform
            ),
            _ => return true,
        };
        seen.insert(key)
    });
    entries.truncate(top);
    println!("\nLeaderboard (period: {period}, sorted by: {sort})");
    println!("Rank   Model                Quant      GPU                  TPS        Prompt/s     Max Ctx");
    for (i, entry) in entries.iter().enumerate() {
        println!(
            "{:<6} {} {} {} {:<10.1} {:<12.1} {}k",
            i + 1,
            cell(entry.model.as_deref().unwrap_or("unknown"), 20),
            cell(entry.quantization.as_deref().unwrap_or("unknown"), 10),
            cell(&entry.gpu, 20),
            entry.summary.avg_tps,
            entry.summary.avg_prompt,
            entry.summary.max_context
        );
    }
    println!("Total: {} benchmark(s) shown", entries.len());
    Ok(())
}

async fn run_search(query: Option<String>, flags: Option<String>) -> Result<()> {
    let config = config::Config::load()?;
    let server_url = config.get_server_url()?;
    let mut params = vec![("limit", "20".to_string())];
    if let Some(query) = query {
        params.push(("model", query));
    }
    if let Some(flags) = flags {
        params.push(("flags", flags));
    }
    let response = community::client()?
        .get(community::endpoint(&server_url, "api/v1/benchmarks")?)
        .query(&params)
        .send()
        .await
        .context("Failed to connect to community server")?;
    let data = community::response_json(response).await?;
    let results = data["results"]
        .as_array()
        .context("Community response is missing results array")?;
    anyhow::ensure!(
        data["count"].as_u64() == Some(results.len() as u64),
        "Community response count does not match its results"
    );
    // Validate every row before printing any apparent success.
    let entries = results
        .iter()
        .map(|entry| -> Result<_> {
            serde_json::from_value::<CommunityEntry>(entry.clone())
                .context("Invalid community benchmark schema")?
                .into_history()
        })
        .collect::<Result<Vec<_>>>()?;
    println!("Found {} community benchmark(s):", entries.len());
    println!("ID     GPU                  Model                TPS        Prompt/s   Submitted");
    for entry in entries {
        println!(
            "{:<6} {} {} {:<10.1} {:<10.1} {}",
            entry.id,
            cell(&entry.gpu, 20),
            cell(entry.model.as_deref().unwrap_or("unknown"), 20),
            entry.summary.avg_tps,
            entry.summary.avg_prompt,
            cell(&entry.timestamp, 10)
        );
    }
    Ok(())
}

async fn run_recommend(
    gpu: Option<String>,
    min_tps: f64,
    limit: i64,
    format: String,
) -> anyhow::Result<()> {
    // Resolve GPU: use provided or auto-detect
    let gpu_family = match gpu {
        Some(g) => g,
        None => {
            // Auto-detect (we're in async context, can await)
            let hw = hardware::detect_hardware().await?;
            hw.gpu_family
        }
    };

    if gpu_family.is_empty() {
        anyhow::bail!("Could not detect GPU. Please specify --gpu flag.");
    }

    // Query recommendations
    let results = db::get_recommendations(&gpu_family, min_tps, limit)?;

    // Machine-readable output must stay valid even when there are no rows.
    if format == "json" {
        println!("{}", serde_json::to_string_pretty(&results)?);
        return Ok(());
    }

    if results.is_empty() {
        println!(
            "No benchmark results found for GPU '{}' with TPS >= {}.",
            terminal_text(&gpu_family),
            min_tps
        );
        println!("Run some benchmarks first with: autarky bench ...");
        return Ok(());
    }

    // Output
    match format.as_str() {
        "json" => {
            println!("{}", serde_json::to_string_pretty(&results)?);
        }
        _ => {
            // Table format
            println!(
                "\nFlag Recommendations for GPU: {}",
                terminal_text(&gpu_family)
            );
            println!("{}", "=".repeat(70));
            println!(
                "{:<5} {:<25} {:<35} {:<10} {:<10} OOM",
                "ID", "Model", "Flags", "Max TPS", "Avg TPS"
            );
            println!("{}", "-".repeat(70));

            for bench in &results {
                // Use parsed flags for readable display; filter out autarky-injected noise
                let parsed: std::collections::BTreeMap<String, String> =
                    serde_json::from_str(&bench.parsed_flags)
                        .context("Invalid saved parsed flags JSON")?;
                let interesting: Vec<(String, String)> = parsed
                    .into_iter()
                    .filter(|(k, _)| k != "--host" && k != "--port")
                    .collect();
                let flags_display = if interesting.is_empty() {
                    "(none)".to_string()
                } else {
                    interesting
                        .iter()
                        .map(|(k, v)| {
                            if v.is_empty() {
                                k.clone()
                            } else {
                                format!("{} {}", k, v)
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                };

                let oom_display = bench
                    .oom_boundary
                    .map_or("".to_string(), |b| format!("OOM@{}", b));

                println!(
                    "{:<5} {:<25} {:<35} {:<10.1} {:<10.1} {}",
                    bench.id,
                    autarky::utils::pad_display(
                        &autarky::utils::truncate_display(
                            bench.model_name.as_deref().unwrap_or("unknown"),
                            24
                        ),
                        25
                    ),
                    terminal_text(&flags_display),
                    bench.max_tps,
                    bench.avg_tps,
                    oom_display
                );
            }
            println!("{}", "-".repeat(70));
            println!(
                "\nTip: These records are not a matched-model speed comparison. Test flags with your own model: autarky bench --passthrough ./llama-server --model <path> [flags...]"
            );
        }
    }

    Ok(())
}

async fn run_delete(mode: &str) -> anyhow::Result<()> {
    match mode {
        "all" => {
            let hardware = hardware::detect_hardware().await?;
            let deleted = db::delete_all_for_hardware(&hardware)?;
            if deleted > 0 {
                println!("deleted all entries");
            } else {
                println!("No entries found for your hardware setup.");
            }
        }
        "last" => {
            let deleted = db::delete_last_entry()?;
            if deleted > 0 {
                println!("deleted last entry");
            } else {
                println!("No entries to delete.");
            }
        }
        _ => unreachable!(), // validated before calling
    }
    Ok(())
}

/// Ensure database and config are initialized (idempotent, called on every run)
fn ensure_initialized() -> anyhow::Result<()> {
    let config_dir = dirs::config_dir()
        .context("Cannot locate the config directory")?
        .join("autarky");
    let data_dir = dirs::data_dir()
        .context("Cannot locate the data directory")?
        .join("autarky");
    std::fs::create_dir_all(&config_dir)?;
    std::fs::create_dir_all(&data_dir)?;
    db::init_db(&data_dir)?;

    let mut config = config::Config::load()?;
    config.get_or_create_api_key()?;
    Ok(())
}

fn run_config(key: Option<String>, value: Option<String>) -> anyhow::Result<()> {
    match (key, value) {
        (Some(k), Some(v)) => {
            let mut config = config::Config::load()?;
            config.set(&k, &v)?;
            config.save()?;
            println!(
                "Set {} = {}",
                terminal_text(&k),
                terminal_text(&config.get(&k)?)
            );
        }
        (Some(k), None) => {
            let config = config::Config::load()?;
            println!("{}: {}", terminal_text(&k), terminal_text(&config.get(&k)?));
        }
        (None, None) => {
            let config = config::Config::load()?;
            println!(
                "Config file: {}",
                terminal_text(&config::Config::config_path().display().to_string())
            );
            println!("{}", serde_json::to_string_pretty(&config.masked())?);
        }
        (_, Some(_)) => {
            anyhow::bail!("Cannot set config value without specifying a key");
        }
    }
    Ok(())
}

fn parse_model_flag(flags: &[String]) -> Option<String> {
    let parsed = autarky::arguments::parse(flags).ok()?;
    autarky::arguments::last_value(&parsed, &["--model", "-m"])
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

// Community benchmark comparison (check after auto-submit).

/// Prompt the user for a yes/no answer and return true for yes.
async fn prompt_yes_no(prompt: &str, cancellation: &mut Cancellation) -> Result<bool> {
    if !std::io::stdin().is_terminal() {
        return Ok(false);
    }
    write_stdout(&format!("{} ", terminal_text(prompt)))?;
    let mut input = String::new();
    loop {
        tokio::select! {
            biased;
            signal = cancellation.wait() => {
                signal?;
                anyhow::bail!("Benchmark cancelled; local results are saved");
            }
            _ = tokio::time::sleep(std::time::Duration::from_millis(50)) => {
                use crossterm::event::{self, Event, KeyCode};
                while event::poll(std::time::Duration::ZERO)? {
                    if let Event::Key(key) = event::read()? {
                        match key.code {
                            KeyCode::Enter => return Ok(matches!(input.trim().to_lowercase().as_str(), "y" | "yes")),
                            KeyCode::Char(ch) => input.push(ch),
                            KeyCode::Backspace => { input.pop(); }
                            _ => {}
                        }
                    }
                }
            }
        }
    }
}

/// Query the community server for benchmarks matching our GPU + model,
/// compare per-context-length metrics, and return the best match.
///
/// Returns `(entry, ctx_label, metric_label, quant_label)` where:
/// - `entry`: the remote benchmark JSON object
/// - `ctx_label`: e.g. "4k"
/// - `metric_label`: e.g. "132.5 tokens/s"
/// - `quant_label`: the full description line for the prompt
async fn check_community_comparison(
    server_url: &str,
    hardware: &hardware::HardwareInfo,
    model_name: &str,
    our_quant: &str,
    descriptors: &ModelDescriptors,
    our_results: &[autarky::monitor::BenchmarkResult],
    api_key: Option<&str>,
) -> Result<Option<(serde_json::Value, String, String, String)>> {
    monitor::summarize_results(our_results)?;
    let client = community::client()?;

    // Query server for benchmarks matching our GPU and model
    let url = community::endpoint(server_url, "api/v1/benchmarks")?;
    let mut req = client.get(url).query(&[
        ("gpu", hardware.gpu_family.as_str()),
        ("model", model_name),
        ("limit", "50"),
    ]);

    if let Some(key) = api_key {
        req = req.header("X-API-Key", community::key_header(key)?);
    }

    let resp = req
        .send()
        .await
        .context("Failed to connect to community server")?;

    let data = community::response_json(resp).await?;
    let results = data["results"]
        .as_array()
        .context("Community response is missing results array")?;
    anyhow::ensure!(
        data["count"].as_u64() == Some(results.len() as u64),
        "Community response count does not match its results"
    );

    if results.is_empty() {
        return Ok(None);
    }

    // Build a lookup of our context_length → TPS/prompt for fast comparison
    let mut our_map: std::collections::HashMap<u64, (f64, f64)> = std::collections::HashMap::new();
    for r in our_results {
        if r.status == "success" {
            our_map.insert(
                r.context_length as u64,
                (r.tokens_per_second, r.prompt_per_second),
            );
        }
    }

    if our_map.is_empty() {
        return Ok(None);
    }

    // Retain the exact context and metric that proved the improvement.
    let mut best_entry: Option<(f64, u64, f64, &str, &serde_json::Value)> = None;

    for entry in results {
        let remote = serde_json::from_value::<CommunityEntry>(entry.clone())
            .context("Invalid community benchmark schema")?
            .into_history()?;
        if remote.model.as_deref() != Some(model_name)
            || remote.quantization.as_deref() != Some(our_quant)
            || !descriptors.compatible(&remote.descriptors)
            || remote.gpu != hardware.gpu_family
            || remote.vram_gb != hardware.vram_gb
            || remote.system_ram_gb != hardware.system_ram_gb
            || remote.os_platform != hardware.os_platform
        {
            continue;
        }

        for metric in monitor::parse_metrics(&remote.raw_metrics)? {
            let ctx = metric.context_length as u64;
            if metric.status != "success" {
                continue;
            }

            // Match our context lengths
            if let Some(&(our_tps, our_prompt)) = our_map.get(&ctx) {
                let remote_tps = metric.tokens_per_second;
                let remote_prompt = metric.prompt_per_second;

                for (ours, remote, label) in [
                    (our_tps, remote_tps, "tokens/s"),
                    (our_prompt, remote_prompt, "prompt/s"),
                ] {
                    if ours > 0.0 && remote > ours {
                        let improvement = (remote - ours) / ours;
                        if (improvement * 100.0).is_finite()
                            && best_entry.as_ref().is_none_or(|best| improvement > best.0)
                        {
                            best_entry = Some((improvement, ctx, remote, label, entry));
                        }
                    }
                }
            }
        }
    }

    match best_entry {
        Some((improvement, best_ctx, best_value, best_metric, entry)) => {
            let remote_quant = entry["quantization"].as_str().unwrap_or("?");
            let remote_model = entry["model_name"].as_str().unwrap_or("?");
            let remote_gpu = entry["gpu_family"].as_str().unwrap_or("?");

            let ctx_label = format!("{}k", best_ctx);
            let metric_label = format!("{best_value:.1} {best_metric}");
            let quant_label = terminal_text(&format!(
                "Found a benchmark using {} and {} {} {} context at {} ({:.1}% faster)",
                remote_gpu,
                remote_model,
                remote_quant,
                ctx_label,
                metric_label,
                improvement * 100.0
            ));

            Ok(Some((entry.clone(), ctx_label, metric_label, quant_label)))
        }
        None => Ok(None),
    }
}

/// Show detailed benchmark results and flags from a remote entry.
fn show_benchmark_details(entry: &serde_json::Value) -> Result<()> {
    let history = serde_json::from_value::<CommunityEntry>(entry.clone())
        .context("Invalid community benchmark schema")?
        .into_history()?;
    let mut output = Vec::new();
    writeln!(output, "\n  Benchmark details:")?;
    writeln!(
        output,
        "    Model:     {} ({})",
        terminal_text(history.model.as_deref().unwrap_or("unknown")),
        terminal_text(history.quantization.as_deref().unwrap_or("unknown"))
    )?;
    writeln!(
        output,
        "    GPU:       {} ({:.1} GB)",
        terminal_text(&history.gpu),
        history.vram_gb
    )?;
    writeln!(
        output,
        "    Submitted: {}",
        terminal_text(&history.timestamp)
    )?;
    writeln!(output, "\n    Context  Tokens/s  Prompt/s  Status")?;
    writeln!(output, "    ───────  ────────  ────────  ──────")?;
    for metric in monitor::parse_metrics(&history.raw_metrics)? {
        if metric.status == "success" {
            writeln!(
                output,
                "    {:>5}k  {:>8.1}  {:>8.1}  ✓",
                metric.context_length, metric.tokens_per_second, metric.prompt_per_second
            )?;
        } else {
            writeln!(
                output,
                "    {:>5}k  {:>8}  {:>8}  {}",
                metric.context_length,
                "-",
                "-",
                terminal_text(&metric.status)
            )?;
        }
    }

    // Show flags/passthrough
    if let Some(flags) = entry["flags_used"].as_array() {
        if !flags.is_empty() {
            let flag_strs: Vec<String> =
                serde_json::from_value(serde_json::Value::Array(flags.clone()))
                    .context("Invalid community flags")?;
            let display = db::sanitize_flags(&flag_strs);
            writeln!(
                output,
                "\n    Flags: --passthrough {}",
                terminal_text(&display.join(" "))
            )?;
        }
    }
    writeln!(output)?;
    write_stdout(&String::from_utf8(output)?)
}

#[cfg(test)]
#[path = "main_cli_tests.rs"]
mod cli_tests;

#[cfg(test)]
mod submission_tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[test]
    fn context_clamping_selects_existing_embedded_prompts() {
        assert_eq!(
            resolve_context_lengths(Some("256"), Some(131072)).unwrap(),
            vec![128]
        );
        assert_eq!(
            resolve_context_lengths(Some("4,2,8"), Some(2048)).unwrap(),
            vec![2]
        );
        assert_eq!(
            resolve_context_lengths(None, Some(6000)).unwrap(),
            vec![1, 2, 4]
        );
    }

    #[test]
    fn unknown_context_metadata_keeps_the_automatic_benchmark_range() {
        assert_eq!(
            resolve_context_lengths(None, Some(0)).unwrap(),
            launcher::SUPPORTED_CONTEXT_LENGTHS
        );
        assert_eq!(
            resolve_context_lengths(None, None).unwrap(),
            launcher::SUPPORTED_CONTEXT_LENGTHS
        );
    }

    #[test]
    fn invalid_context_requests_fail_before_launch() {
        for value in ["0", "3", "512", "", "1,no"] {
            assert!(
                resolve_context_lengths(Some(value), None).is_err(),
                "{}",
                value
            );
        }
        assert!(resolve_context_lengths(None, Some(500)).is_err());
    }

    pub(super) fn benchmark_db() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        db::init_db(dir.path()).unwrap();
        let conn = rusqlite::Connection::open(dir.path().join("benchmarks.db")).unwrap();
        conn.execute("INSERT INTO hardware_fingerprints (id, gpu_family, vram_gb, system_ram_gb, os_platform) VALUES (1, 'Test GPU', 16, 32, 'linux')", []).unwrap();
        conn.execute("INSERT INTO model_fingerprints (id, model_name, quantization, architecture_type, expert_count, parameters_billions) VALUES (1, 'Test Model', 'Q4_K_M', 'dense', 0, 1.1)", []).unwrap();
        let flags = serde_json::json!([
            "--model=/private/model.gguf",
            "--api-key=private-value",
            "--threads",
            "8"
        ]);
        let metrics = serde_json::json!([autarky::monitor::BenchmarkResult::new(1, 42.0, 100.0)]);
        conn.execute("INSERT INTO benchmark_runs (id, hardware_id, model_id, flags_used, context_lengths_tested, raw_metrics, llama_server_version) VALUES (1, 1, 1, ?1, '[1]', ?2, 'test-build')", [flags.to_string(), metrics.to_string()]).unwrap();
        dir
    }

    pub(super) async fn submission_server(
        status: &str,
        body: &str,
    ) -> (String, tokio::task::JoinHandle<serde_json::Value>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let response = format!("HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", status, body.len(), body);
        let task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let (header_end, body_length) = loop {
                let mut chunk = [0; 4096];
                let count = stream.read(&mut chunk).await.unwrap();
                assert!(count > 0, "Submission connection closed before headers");
                request.extend_from_slice(&chunk[..count]);
                if let Some(end) = request.windows(4).position(|s| s == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&request[..end]).to_ascii_lowercase();
                    assert!(headers.starts_with("post /api/v1/benchmark http/1.1"));
                    assert!(headers.contains("x-api-key: test-submitter"));
                    let length: usize = headers
                        .lines()
                        .find_map(|line| line.strip_prefix("content-length: "))
                        .unwrap()
                        .parse()
                        .unwrap();
                    break (end + 4, length);
                }
            };
            while request.len() < header_end + body_length {
                let mut chunk = [0; 4096];
                let count = stream.read(&mut chunk).await.unwrap();
                assert!(count > 0, "Submission connection closed before payload");
                request.extend_from_slice(&chunk[..count]);
            }
            let payload =
                serde_json::from_slice(&request[header_end..header_end + body_length]).unwrap();
            stream.write_all(response.as_bytes()).await.unwrap();
            payload
        });
        (format!("http://{}", address), task)
    }

    pub(super) async fn submit_to_fixture(dir: &tempfile::TempDir, url: &str) -> Result<i64> {
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            submit_benchmark(
                dir.path().join("benchmarks.db").to_str().unwrap(),
                1,
                url,
                "test-submitter",
            ),
        )
        .await
        .expect("Submission must complete on the existing async runtime")
    }

    #[tokio::test]
    async fn submission_uses_existing_runtime_and_sanitizes_saved_flags() {
        let dir = benchmark_db();
        let (url, server) = submission_server("200 OK", r#"{"id":7}"#).await;
        assert_eq!(submit_to_fixture(&dir, &url).await.unwrap(), 7);
        let payload = server.await.unwrap();
        assert_eq!(payload["hardware"]["gpu_family"], "Test GPU");
        assert_eq!(payload["model"]["quantization"], "Q4_K_M");
        assert_eq!(payload["raw_metrics"][0]["tokens_per_second"], 42.0);
        assert_eq!(
            payload["flags_used"],
            serde_json::json!(["--model=/model.gguf", "--threads", "8"])
        );
        assert_eq!(payload["llama_server_version"], "test-build");
    }

    #[tokio::test]
    async fn failed_upload_returns_an_error_and_keeps_the_local_record() {
        let dir = benchmark_db();
        let (url, server) =
            submission_server("429 Too Many Requests", r#"{"error":"rate limited"}"#).await;
        let error = submit_to_fixture(&dir, &url).await.unwrap_err();
        assert!(error.to_string().contains("429"));
        server.await.unwrap();
        let conn = rusqlite::Connection::open(dir.path().join("benchmarks.db")).unwrap();
        assert_eq!(
            conn.query_row("SELECT count(*) FROM benchmark_runs", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            1
        );
    }

    #[tokio::test]
    async fn success_without_a_submission_id_is_not_reported_as_an_upload() {
        let dir = benchmark_db();
        let (url, server) = submission_server("200 OK", r#"{"status":"ok"}"#).await;
        assert!(submit_to_fixture(&dir, &url)
            .await
            .unwrap_err()
            .to_string()
            .contains("submission ID"));
        server.await.unwrap();
    }

    #[test]
    fn model_path_detection_supports_aliases_and_inline_values() {
        for input in [
            vec!["--model", "/private/model.gguf"],
            vec!["-m", "/private/model.gguf"],
            vec!["--model=/private/model.gguf"],
            vec!["-m=/private/model.gguf"],
        ] {
            let flags = input.into_iter().map(str::to_string).collect::<Vec<_>>();
            assert_eq!(
                parse_model_flag(&flags).as_deref(),
                Some("/private/model.gguf")
            );
        }
    }

    #[test]
    fn last_model_argument_wins_across_aliases() {
        let flags = ["--model", "first.gguf", "-m=last.gguf"]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        assert_eq!(parse_model_flag(&flags).as_deref(), Some("last.gguf"));
    }
}
