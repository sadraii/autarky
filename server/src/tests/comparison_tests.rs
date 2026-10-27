use super::*;

fn assert_unavailable(result: &Value, state: &str) {
    for metric in ["tps", "prompt"] {
        assert_eq!(result[metric]["state"], state);
        assert!(result[metric]["percent"].is_null());
        assert!(result[metric]["baseline_average"].is_null());
        assert!(result[metric]["candidate_average"].is_null());
    }
    assert_eq!(result["delta"], json!({}));
}

#[actix_web::test]
async fn unknown_or_partial_model_identity_keeps_observations_without_relative_claims() {
    let fixture = Fixture::new();
    let app = app!(fixture);
    let known = payload()["model"].clone();
    let mut partial = known.clone();
    partial["quantization"] = Value::Null;
    let mut omitted_quant = known.clone();
    omitted_quant
        .as_object_mut()
        .unwrap()
        .remove("quantization");
    for (a_model, b_model) in [
        (Value::Null, Value::Null),
        (Value::Null, known.clone()),
        (known.clone(), Value::Null),
        (partial.clone(), partial.clone()),
        (partial.clone(), known.clone()),
        (known.clone(), partial),
        (omitted_quant, known),
    ] {
        let mut a = payload();
        let mut b = payload();
        a["model"] = a_model.clone();
        b["model"] = b_model.clone();
        b["raw_metrics"][0]["tokens_per_second"] = json!(1.0);
        let a_id = post!(app, "key", &a);
        let b_id = post!(app, "key", &b);
        for result in [comparison!(app, a_id, b_id), comparison!(app, b_id, a_id)] {
            assert_unavailable(&result, "unknown_model_identity");
            assert_eq!(result["matched_contexts"], json!([1024, 2048]));
        }
        let result = comparison!(app, a_id, b_id);
        for (field, submitted) in [("benchmark_1", &a), ("benchmark_2", &b)] {
            assert_eq!(result[field]["raw_metrics"], submitted["raw_metrics"]);
            for descriptor in ["model_name", "quantization", "architecture_type"] {
                assert_eq!(result[field][descriptor], submitted["model"][descriptor]);
            }
            assert_eq!(
                result[field]["gpu_family"],
                submitted["hardware"]["gpu_family"]
            );
        }
    }
}

#[actix_web::test]
async fn different_known_names_or_quantizations_do_not_produce_deltas() {
    let fixture = Fixture::new();
    let app = app!(fixture);
    let a = payload();
    let a_id = post!(app, "key", &a);
    for (field, value) in [("model_name", "other-model"), ("quantization", "Q8")] {
        let mut b = payload();
        b["model"][field] = json!(value);
        let b_id = post!(app, "key", &b);
        for result in [comparison!(app, a_id, b_id), comparison!(app, b_id, a_id)] {
            assert_unavailable(&result, "incompatible_model");
            assert_eq!(result["benchmark_1"]["raw_metrics"], a["raw_metrics"]);
            assert_eq!(result["benchmark_2"]["raw_metrics"], b["raw_metrics"]);
        }
        let result = comparison!(app, a_id, b_id);
        assert_eq!(result["benchmark_2"][field], value);
    }
}

#[actix_web::test]
async fn literal_unknown_name_and_third_party_quantization_compare_across_hardware() {
    let fixture = Fixture::new();
    let app = app!(fixture);
    let mut a = payload();
    a["model"]["model_name"] = json!("unknown");
    a["model"]["quantization"] = json!("vendor-custom-quant");
    // Unknown optional numeric metadata does not make a known workload unknown.
    a["model"]["expert_count"] = Value::Null;
    a["model"]["parameters_billions"] = Value::Null;
    let a_id = post!(app, "key", &a);
    let mut b = a.clone();
    b["hardware"]["gpu_family"] = json!("different GPU");
    b["hardware"]["vram_gb"] = json!(24.0);
    for point in b["raw_metrics"].as_array_mut().unwrap() {
        point["tokens_per_second"] = json!(point["tokens_per_second"].as_f64().unwrap() / 2.0);
        point["prompt_per_second"] = json!(point["prompt_per_second"].as_f64().unwrap() / 2.0);
    }
    let b_id = post!(app, "key", &b);
    let result = comparison!(app, a_id, b_id);
    for metric in ["tps", "prompt"] {
        assert_eq!(result[metric]["state"], "available");
        assert_eq!(result[metric]["percent"], 100.0);
    }
    assert_eq!(result["delta"]["tps_percent"], "+100.0%");
    assert_eq!(result["delta"]["prompt_percent"], "+100.0%");
    assert_eq!(result["benchmark_1"]["model_name"], "unknown");
    assert_eq!(result["benchmark_1"]["quantization"], "vendor-custom-quant");
    assert!(result["benchmark_1"]["expert_count"].is_null());
    assert_eq!(result["benchmark_2"]["gpu_family"], "different GPU");
}

#[actix_web::test]
async fn legacy_empty_identity_fields_remain_raw_but_cannot_establish_a_workload() {
    let fixture = Fixture::new();
    let app = app!(fixture);
    for (index, (name, quant)) in [("", "Q4"), (" \t ", "Q4"), ("model", ""), ("model", " ")]
        .into_iter()
        .enumerate()
    {
        let mut body = payload();
        body["model"]["model_name"] = json!(format!("legacy-{index}"));
        let a_id = post!(app, "key", &body);
        let b_id = post!(app, "key", &body);
        fixture.state.db.call(move |conn, _| {
            conn.execute("UPDATE model_fingerprints SET model_name=?,quantization=? WHERE id=(SELECT model_id FROM benchmark_submissions WHERE id=?)", params![name,quant,a_id])?;
            Ok(())
        }).await.unwrap();
        let result = comparison!(app, a_id, b_id);
        assert_unavailable(&result, "unknown_model_identity");
        assert_eq!(result["benchmark_1"]["model_name"], name);
        assert_eq!(result["benchmark_1"]["quantization"], quant);
        assert_eq!(result["benchmark_1"]["raw_metrics"], body["raw_metrics"]);
    }
}
