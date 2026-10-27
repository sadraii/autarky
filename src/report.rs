//! Structured terminal reports with display-cell alignment.
use super::hardware::HardwareInfo;
use super::monitor::BenchmarkResult;
use crate::utils::{display_width, pad_display, strip_ansi};
use anyhow::Result;
use std::io::Write;

const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

pub fn print_report(
    hardware: &HardwareInfo,
    model_name: &str,
    results: &[BenchmarkResult],
    llama_server_version: &str,
) -> Result<()> {
    let rendered = render_report(hardware, model_name, results, llama_server_version);
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    output.write_all(rendered.as_bytes())?;
    output.flush()?;
    Ok(())
}

pub fn render_report(
    hardware: &HardwareInfo,
    model_name: &str,
    results: &[BenchmarkResult],
    llama_server_version: &str,
) -> String {
    let lines = build_content_lines(hardware, model_name, results, llama_server_version);
    let width = lines
        .iter()
        .map(|line| display_width(line))
        .max()
        .unwrap_or(0);
    let border = "─".repeat(width + 2);
    let mut report = format!("╭{}╮\n", border);
    for line in lines {
        report.push_str(&format!("│ {} │\n", pad_display(&line, width)));
    }
    report.push_str(&format!("╰{}╯\n", border));
    report
}

fn valid_result(result: &BenchmarkResult) -> bool {
    result.status == "success"
        && result.tokens_per_second.is_finite()
        && result.tokens_per_second > 0.0
        && result.prompt_per_second.is_finite()
        && result.prompt_per_second > 0.0
}

fn build_content_lines(
    hardware: &HardwareInfo,
    model_name: &str,
    results: &[BenchmarkResult],
    llama_server_version: &str,
) -> Vec<String> {
    let mut lines = vec![
        format!("autarky bench — {}", strip_ansi(model_name)),
        format!("llama-server {}", strip_ansi(llama_server_version)),
        String::new(),
        format!(
            "Setup: {} {:.0}GB • {} GB • {}",
            strip_ansi(&hardware.gpu_family),
            hardware.vram_gb,
            hardware.system_ram_gb,
            strip_ansi(&hardware.os_platform)
        ),
        String::new(),
        "  Context   Tokens/s   Prompt/s   Status".into(),
        "  ───────   ────────   ────────   ──────".into(),
    ];
    let best = results
        .iter()
        .enumerate()
        .filter(|(_, r)| valid_result(r))
        .max_by(|(_, left), (_, right)| left.tokens_per_second.total_cmp(&right.tokens_per_second))
        .map(|(index, _)| index);
    for (index, result) in results.iter().enumerate() {
        let context = format!("{:>6}k", result.context_length);
        if valid_result(result) {
            let rate = format!("{:>8.1}", result.tokens_per_second);
            let rate = if best == Some(index) {
                format!("{BOLD}{rate}{RESET}")
            } else {
                rate
            };
            lines.push(format!(
                "  {}   {}   {:>8.1}   ✓",
                context, rate, result.prompt_per_second
            ));
        } else {
            let status = match result.status.as_str() {
                "oom" => "OOM",
                "cancelled" => "Cancelled",
                "success" => "Invalid timing",
                _ => "Error",
            };
            lines.push(format!(
                "  {}   {:>8}   {:>8}   {}",
                context, "-", "-", status
            ));
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decimal_best_is_not_integer_truncated_and_nonfinite_rates_are_invalid() {
        let hardware = HardwareInfo {
            gpu_family: "GPU 模型".into(),
            vram_gb: 0.0,
            system_ram_gb: 1,
            os_platform: "linux".into(),
            devices: vec![],
        };
        let results = [
            BenchmarkResult::new(1, 12.9, 20.0),
            BenchmarkResult::new(2, 12.1, 20.0),
            BenchmarkResult::new(4, f64::INFINITY, 20.0),
        ];
        let report = render_report(&hardware, "模型🙂", &results, "1");
        assert!(report.contains("\x1b[1m    12.9\x1b[0m"));
        assert!(!report.contains("\x1b[1m    12.1"));
        assert!(report.contains("Invalid timing"));
        let widths = report.lines().map(display_width).collect::<Vec<_>>();
        assert!(widths.iter().all(|width| *width == widths[0]));
    }
}
