//! Monitor module
//! Parse benchmark results from HTTP responses

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub context_length: usize,
    pub tokens_per_second: f64,
    pub prompt_per_second: f64,
    pub status: String,
    pub error: Option<String>,
}

impl BenchmarkResult {
    pub fn new(context_length: usize, tokens_per_second: f64, prompt_per_second: f64) -> Self {
        Self {
            context_length,
            tokens_per_second,
            prompt_per_second,
            status: "success".to_string(),
            error: None,
        }
    }

    pub fn oom(context_length: usize) -> Self {
        Self {
            context_length,
            tokens_per_second: 0.0,
            prompt_per_second: 0.0,
            status: "oom".to_string(),
            error: Some("Out of memory".to_string()),
        }
    }

    pub fn error(context_length: usize, message: impl Into<String>) -> Self {
        Self {
            context_length,
            tokens_per_second: 0.0,
            prompt_per_second: 0.0,
            status: "error".to_string(),
            error: Some(message.into()),
        }
    }

    pub fn cancelled(context_length: usize) -> Self {
        Self {
            status: "cancelled".to_string(),
            ..Self::error(context_length, "Benchmark cancelled")
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.context_length == 0 {
            bail!("A benchmark context must be positive");
        }
        if !self.tokens_per_second.is_finite()
            || !self.prompt_per_second.is_finite()
            || self.tokens_per_second < 0.0
            || self.prompt_per_second < 0.0
        {
            bail!("Benchmark rates must be finite and nonnegative");
        }
        match self.status.as_str() {
            "success" if self.tokens_per_second > 0.0 && self.error.is_none() => {}
            "oom" | "error" | "cancelled"
                if self.tokens_per_second == 0.0 && self.prompt_per_second == 0.0 => {}
            _ => bail!("Benchmark status and measurements are inconsistent"),
        }
        if self
            .error
            .as_ref()
            .is_some_and(|message| message.len() > 4096)
        {
            bail!("Benchmark error description is too long");
        }
        Ok(())
    }
}

#[derive(Debug, Default, Clone, Serialize, PartialEq)]
pub struct MetricSummary {
    pub max_tps: f64,
    pub avg_tps: f64,
    pub avg_prompt: f64,
    pub max_context: usize,
    pub success_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct PairedComparison {
    pub contexts: Vec<usize>,
    pub left: MetricSummary,
    pub right: MetricSummary,
    pub tps_delta: Option<f64>,
    pub prompt_delta: Option<f64>,
}

pub fn parse_metrics(raw: &Value) -> Result<Vec<BenchmarkResult>> {
    let results: Vec<BenchmarkResult> =
        serde_json::from_value(raw.clone()).context("Invalid benchmark metrics")?;
    validate_results(&results)?;
    Ok(results)
}

fn validate_results(results: &[BenchmarkResult]) -> Result<()> {
    let mut contexts = BTreeSet::new();
    for result in results {
        result.validate()?;
        if !contexts.insert(result.context_length) {
            bail!("Benchmark metrics contain a repeated context");
        }
    }
    Ok(())
}

pub fn summarize_results(results: &[BenchmarkResult]) -> Result<MetricSummary> {
    validate_results(results)?;
    let mut summary = MetricSummary::default();
    for result in results.iter().filter(|result| result.status == "success") {
        summary.success_count += 1;
        let count = summary.success_count as f64;
        // Incremental means cannot overflow when every input is finite/nonnegative.
        summary.avg_tps += (result.tokens_per_second - summary.avg_tps) / count;
        summary.avg_prompt += (result.prompt_per_second - summary.avg_prompt) / count;
        summary.max_tps = summary.max_tps.max(result.tokens_per_second);
        summary.max_context = summary.max_context.max(result.context_length);
    }
    Ok(summary)
}

pub fn summarize_metrics(raw: &Value) -> Result<MetricSummary> {
    summarize_results(&parse_metrics(raw)?)
}

fn percentage_delta(value: f64, baseline: f64) -> Option<f64> {
    if baseline <= 0.0 {
        return None;
    }
    let delta = ((value - baseline) / baseline) * 100.0;
    delta.is_finite().then_some(delta)
}

/// The left configuration is compared with the right baseline, on shared contexts only.
pub fn paired_metrics(left: &Value, right: &Value) -> Result<Option<PairedComparison>> {
    let left = parse_metrics(left)?;
    let right = parse_metrics(right)?;
    let successful_right: BTreeSet<_> = right
        .iter()
        .filter(|result| result.status == "success")
        .map(|result| result.context_length)
        .collect();
    let contexts: BTreeSet<_> = left
        .iter()
        .filter(|result| {
            result.status == "success" && successful_right.contains(&result.context_length)
        })
        .map(|result| result.context_length)
        .collect();
    if contexts.is_empty() {
        return Ok(None);
    }
    let left: Vec<_> = left
        .into_iter()
        .filter(|result| contexts.contains(&result.context_length))
        .collect();
    let right: Vec<_> = right
        .into_iter()
        .filter(|result| contexts.contains(&result.context_length))
        .collect();
    let left = summarize_results(&left)?;
    let right = summarize_results(&right)?;
    Ok(Some(PairedComparison {
        contexts: contexts.into_iter().collect(),
        tps_delta: percentage_delta(left.avg_tps, right.avg_tps),
        prompt_delta: percentage_delta(left.avg_prompt, right.avg_prompt),
        left,
        right,
    }))
}

#[cfg(test)]
#[path = "monitor_tests.rs"]
mod tests;
