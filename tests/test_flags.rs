//! Uploaded flags must retain tuning values, but never secrets or private paths.

use autarky::db::{parse_flags, sanitize_flags};

#[test]
fn multi_value_options_and_option_looking_values_keep_native_boundaries() {
    let flags = args(&[
        "--spec-replace",
        "--host=target",
        "--api-key=draft",
        "--alias",
        "--api-key=display-name",
        "--lora-scaled",
        "/private/a.gguf:0.5,/private/b.gguf:-1",
    ]);
    let clean = sanitize_flags(&flags);
    assert_eq!(
        clean,
        args(&[
            "--spec-replace",
            "--host=target",
            "--api-key=draft",
            "--alias",
            "--api-key=display-name",
            "--lora-scaled",
            "/a.gguf:0.5,/b.gguf:-1"
        ])
    );
    assert_eq!(
        parse_flags(&clean)["--spec-replace"],
        serde_json::json!(["--host=target", "--api-key=draft"])
    );
    assert_eq!(sanitize_flags(&clean), clean);
}

#[test]
fn required_values_are_not_reparsed_and_legacy_secrets_stay_removed() {
    let legacy = args(&["--flash-attn", "--api-key", "--threads=owned-test-secret"]);
    assert!(sanitize_flags(&legacy)
        .iter()
        .all(|value| !value.contains("owned-test-secret")));

    let flags = args(&["--alias", "--gpu-layers=0", "--gpu-layers", "1"]);
    assert_eq!(
        parse_flags(&flags),
        serde_json::json!({"--alias":"--gpu-layers=0", "--gpu-layers":"1"})
    );
    assert_eq!(sanitize_flags(&flags), flags);
    for flags in [
        args(&[
            "--alias",
            "--lora",
            "--api-key",
            "owned-test-secret",
            "--model",
            "/private/model.gguf",
        ]),
        args(&[
            "--flash-attn",
            "--api-key",
            "owned-test-secret",
            "--threads",
            "8",
        ]),
    ] {
        let clean = sanitize_flags(&flags);
        assert!(!clean
            .iter()
            .any(|value| value.contains("owned-test-secret") || value.contains("/private/")));
        assert_eq!(sanitize_flags(&clean), clean);
    }
}

fn args(values: &[&str]) -> Vec<String> {
    values.iter().map(|value| value.to_string()).collect()
}

#[test]
fn secrets_are_removed_in_both_argument_forms() {
    for key in [
        "--api-key",
        "--api-key-file",
        "--ssl-key-file",
        "--ssl-cert-file",
        "--hf-token",
        "-hft",
    ] {
        for input in [
            args(&[key, "-private-value", "--threads", "8"]),
            args(&[&format!("{key}=private-value"), "--threads", "8"]),
        ] {
            assert_eq!(sanitize_flags(&input), args(&["--threads", "8"]), "{key}");
        }
    }
}

#[test]
fn private_paths_are_removed_for_aliases_and_inline_values() {
    for key in [
        "--model",
        "-m",
        "--lora",
        "--model-draft",
        "-md",
        "--mmproj",
        "-mm",
        "--hf-repo",
        "-hf",
        "-hfr",
        "--hf-file",
        "-hff",
        "--log-file",
        "--hf-file-v",
        "-hffv",
        "--hf-repo-v",
        "-hfv",
        "-hfrv",
        "--grammar-file",
        "--json-schema-file",
        "-jf",
        "--tts-speaker-file",
    ] {
        assert_eq!(
            sanitize_flags(&args(&[key, "/home/private/models/model.gguf"])),
            args(&[key, "/model.gguf"])
        );
        assert_eq!(
            sanitize_flags(&args(&[&format!("{key}=/home/private/models/model.gguf")])),
            args(&[&format!("{key}=/model.gguf")])
        );
    }
    assert_eq!(
        sanitize_flags(&args(&["-m", r"C:\Users\private\model.gguf"])),
        args(&["-m", "/model.gguf"])
    );
}

#[test]
fn url_credentials_and_access_tokens_are_not_shared() {
    assert_eq!(sanitize_flags(&args(&[
        "--model-url=https://private:password@example.test/models/model.gguf?token=private#fragment",
        "--lora", "/private/a.gguf,/private/b.gguf",
    ])), args(&["--model-url=/model.gguf", "--lora", "/a.gguf,/b.gguf"]));
}

#[test]
fn commas_in_url_queries_cannot_bypass_redaction() {
    for url in [
        "https://private:password@example.test/model.gguf?token=private,second-secret#fragment",
        "file:///private/model.gguf?token=private,second-secret#fragment",
    ] {
        assert_eq!(
            sanitize_flags(&args(&["--model-url", url])),
            args(&["--model-url", "/model.gguf"])
        );
    }
}

#[test]
fn underscore_aliases_use_the_same_privacy_policy() {
    for key in [
        "--api_key",
        "--api-key_file",
        "--ssl_key-file",
        "--hf_token",
    ] {
        assert_eq!(
            sanitize_flags(&args(&[key, "private-value", "--threads", "8"])),
            args(&["--threads", "8"])
        );
        assert!(sanitize_flags(&args(&[&format!("{key}=private-value")])).is_empty());
    }
    let sanitized = sanitize_flags(&args(&[
        "--model_draft=/private/model_name.gguf",
        "--reuse_port",
        "--gpu_layers",
        "-1",
    ]));
    assert_eq!(
        sanitized,
        args(&["--model_draft=/model_name.gguf", "--gpu_layers", "-1"])
    );
    assert_eq!(
        parse_flags(&sanitized),
        serde_json::json!({"--model-draft": "/model_name.gguf", "--gpu-layers": "-1"})
    );
}

#[test]
fn legacy_url_paths_lose_query_data_and_url_filenames_remain_stable() {
    for key in ["--mmproj-url", "-mmu", "--model_url", "-mu"] {
        assert_eq!(
            sanitize_flags(&args(&[
                key,
                "/model.gguf?token=private,second-secret#fragment"
            ])),
            args(&[key, "/model.gguf"])
        );
    }
    let sanitized = sanitize_flags(&args(&[
        "--model-url",
        "https://example.test/model,variant.gguf?token=private",
    ]));
    assert_eq!(sanitized, args(&["--model-url", "/model,variant.gguf"]));
    assert_eq!(sanitize_flags(&sanitized), sanitized);
}

#[test]
fn internal_flags_are_removed_without_losing_tuning_values() {
    let input = args(&[
        "--host=0.0.0.0",
        "--port",
        "8080",
        "--reuse-port",
        "--threads=8",
        "--gpu-layers",
        "-1",
        "--flash-attn",
    ]);
    let sanitized = sanitize_flags(&input);
    assert_eq!(
        sanitized,
        args(&["--threads=8", "--gpu-layers", "-1", "--flash-attn"])
    );
    assert_eq!(
        parse_flags(&sanitized),
        serde_json::json!({"--threads": "8", "--gpu-layers": "-1", "--flash-attn": ""})
    );
    assert_eq!(
        sanitize_flags(&sanitized),
        sanitized,
        "Upload-time sanitization must be safe for already-sanitized records"
    );
}
