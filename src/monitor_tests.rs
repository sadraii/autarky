use super::*;

fn raw(results: Vec<BenchmarkResult>) -> Value {
    serde_json::to_value(results).unwrap()
}

#[test]
fn failed_contexts_never_expand_measured_capacity_or_depress_averages() {
    let metrics = vec![
        BenchmarkResult::new(1, 10.9, 100.0),
        BenchmarkResult::new(2, 10.1, 300.0),
        BenchmarkResult::oom(4),
        BenchmarkResult::error(8, "fixture failure"),
    ];
    let summary = summarize_results(&metrics).unwrap();
    assert_eq!(summary.success_count, 2);
    assert_eq!(summary.max_context, 2);
    assert_eq!(summary.max_tps, 10.9);
    assert_eq!(summary.avg_tps, 10.5);
    assert_eq!(summary.avg_prompt, 200.0);
    assert_eq!(summarize_metrics(&raw(metrics)).unwrap(), summary);
}

#[test]
fn absent_samples_do_not_establish_an_equal_performance_result() {
    assert_eq!(summarize_results(&[]).unwrap().success_count, 0);
    assert_eq!(
        summarize_results(&[BenchmarkResult::oom(1)])
            .unwrap()
            .max_context,
        0
    );
    assert!(paired_metrics(
        &raw(vec![BenchmarkResult::oom(1)]),
        &raw(vec![BenchmarkResult::new(1, 10.0, 20.0)])
    )
    .unwrap()
    .is_none());
    assert!(paired_metrics(
        &raw(vec![BenchmarkResult::new(1, 1.0, 2.0)]),
        &raw(vec![BenchmarkResult::new(2, 1.0, 2.0)])
    )
    .unwrap()
    .is_none());
}

#[test]
fn paired_comparisons_use_only_matching_successful_workloads() {
    let left = raw(vec![
        BenchmarkResult::new(1, 20.0, 200.0),
        BenchmarkResult::new(2, 1.0, 10.0),
        BenchmarkResult::oom(4),
    ]);
    let right = raw(vec![
        BenchmarkResult::new(1, 10.0, 100.0),
        BenchmarkResult::new(4, 9999.0, 9999.0),
    ]);
    let comparison = paired_metrics(&left, &right).unwrap().unwrap();
    assert_eq!(comparison.contexts, vec![1]);
    assert_eq!(comparison.left.avg_tps, 20.0);
    assert_eq!(comparison.right.avg_tps, 10.0);
    assert_eq!(comparison.tps_delta, Some(100.0));
    assert_eq!(comparison.prompt_delta, Some(100.0));
}

#[test]
fn huge_finite_samples_do_not_overflow_the_summary() {
    let summary = summarize_results(&[
        BenchmarkResult::new(1, 1e308, 1e308),
        BenchmarkResult::new(2, 1e308, 1e308),
    ])
    .unwrap();
    assert_eq!(summary.avg_tps, 1e308);
    assert_eq!(summary.avg_prompt, 1e308);
    assert_eq!(percentage_delta(1e308, 1e-308), None);
    assert_eq!(percentage_delta(10.0, 0.0), None);
}

#[test]
fn zero_prompt_baselines_are_explicitly_unavailable_not_equal() {
    let comparison = paired_metrics(
        &raw(vec![BenchmarkResult::new(1, 20.0, 10.0)]),
        &raw(vec![BenchmarkResult::new(1, 10.0, 0.0)]),
    )
    .unwrap()
    .unwrap();
    assert_eq!(comparison.tps_delta, Some(100.0));
    assert_eq!(comparison.prompt_delta, None);
}

#[test]
fn invalid_or_duplicate_measurements_fail_loudly() {
    for invalid in [
        BenchmarkResult::new(0, 1.0, 2.0),
        BenchmarkResult::new(1, 0.0, 2.0),
        BenchmarkResult::new(1, -1.0, 2.0),
        BenchmarkResult::new(1, f64::NAN, 2.0),
        BenchmarkResult::new(1, f64::INFINITY, 2.0),
        BenchmarkResult::new(1, 1.0, -2.0),
    ] {
        assert!(summarize_results(&[invalid]).is_err());
    }
    assert!(summarize_results(&[
        BenchmarkResult::new(1, 1.0, 2.0),
        BenchmarkResult::new(1, 3.0, 4.0)
    ])
    .is_err());
    assert!(summarize_metrics(&serde_json::json!({})).is_err());
    assert!(summarize_metrics(&serde_json::json!([{"context_length":1,"tokens_per_second":1.0,"prompt_per_second":2.0,"status":"invalid"}])).is_err());
}

#[test]
fn explicit_error_and_cancelled_states_remain_valid_without_fake_rates() {
    for result in [
        BenchmarkResult::oom(1),
        BenchmarkResult::error(2, "fixture failure"),
        BenchmarkResult::cancelled(4),
    ] {
        result.validate().unwrap();
        assert_eq!(result.tokens_per_second, 0.0);
        assert_eq!(result.prompt_per_second, 0.0);
    }
}
