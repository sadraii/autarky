//! Generated metadata fixtures must not depend on a developer's model files.

use autarky::gguf::{parse_gguf, GgufInfo};
use tempfile::tempdir;

type Entry = (String, u32, Vec<u8>);

fn string(value: &str) -> Vec<u8> {
    let mut bytes = (value.len() as u64).to_le_bytes().to_vec();
    bytes.extend(value.as_bytes());
    bytes
}

fn entry(key: &str, kind: u32, value: Vec<u8>) -> Entry {
    (key.to_string(), kind, value)
}

fn fixture(entries: &[Entry]) -> Vec<u8> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    bytes.extend((entries.len() as u64).to_le_bytes());
    for (key, kind, value) in entries {
        bytes.extend(string(key));
        bytes.extend(kind.to_le_bytes());
        bytes.extend(value);
    }
    bytes
}

fn parse(bytes: &[u8]) -> anyhow::Result<GgufInfo> {
    let dir = tempdir()?;
    let path = dir.path().join("fixture.gguf");
    std::fs::write(&path, bytes)?;
    parse_gguf(path.to_str().unwrap())
}

fn array(kind: u32, count: u64, payload: &[u8]) -> Vec<u8> {
    let mut bytes = kind.to_le_bytes().to_vec();
    bytes.extend(count.to_le_bytes());
    bytes.extend(payload);
    bytes
}

#[test]
fn metadata_preserves_names_and_integer_parameters() {
    let info = parse(&fixture(&[
        entry("general.name", 8, string("org/My_Model_Name")),
        entry("general.architecture", 8, string("llama")),
        entry(
            "general.parameter_count",
            10,
            7_000_000_000u64.to_le_bytes().to_vec(),
        ),
        entry(
            "llama.context_length",
            10,
            131_072u64.to_le_bytes().to_vec(),
        ),
        entry("llama.expert_count", 2, 8u16.to_le_bytes().to_vec()),
        entry("llama.block_count", 1, vec![32]),
    ]))
    .unwrap();
    assert_eq!(info.model_name.as_deref(), Some("org/My_Model_Name"));
    assert_eq!(info.architecture.as_deref(), Some("llama"));
    assert_eq!(info.parameters_billions, Some(7.0));
    assert_eq!(info.context_length, Some(131_072));
    assert_eq!(info.expert_count, Some(8));
    assert_eq!(info.block_count, Some(32));
    assert!(info.is_moe);
}

#[test]
fn every_installed_llama_file_type_is_mapped_not_a_tensor_type() {
    // Complete active llama_ftype table, including its sentinel, from llama.h.
    let types = [
        (0, "F32"),
        (1, "F16"),
        (2, "Q4_0"),
        (3, "Q4_1"),
        (7, "Q8_0"),
        (8, "Q5_0"),
        (9, "Q5_1"),
        (10, "Q2_K"),
        (11, "Q3_K_S"),
        (12, "Q3_K_M"),
        (13, "Q3_K_L"),
        (14, "Q4_K_S"),
        (15, "Q4_K_M"),
        (16, "Q5_K_S"),
        (17, "Q5_K_M"),
        (18, "Q6_K"),
        (19, "IQ2_XXS"),
        (20, "IQ2_XS"),
        (21, "Q2_K_S"),
        (22, "IQ3_XS"),
        (23, "IQ3_XXS"),
        (24, "IQ1_S"),
        (25, "IQ4_NL"),
        (26, "IQ3_S"),
        (27, "IQ3_M"),
        (28, "IQ2_S"),
        (29, "IQ2_M"),
        (30, "IQ4_XS"),
        (31, "IQ1_M"),
        (32, "BF16"),
        (36, "TQ1_0"),
        (37, "TQ2_0"),
        (38, "MXFP4_MOE"),
        (39, "NVFP4"),
        (40, "Q1_0"),
    ];
    for (kind, name) in types {
        let info = parse(&fixture(&[entry(
            "general.file_type",
            4,
            u32::to_le_bytes(kind).to_vec(),
        )]))
        .unwrap();
        assert_eq!(
            info.quantization.as_deref(),
            Some(name),
            "llama_ftype={kind}"
        );
        assert_eq!(info.file_type, Some(kind));
    }
    for removed_or_unknown in [4u32, 5, 6, 33, 34, 35, 41, 1024, u32::MAX] {
        let info = parse(&fixture(&[entry(
            "general.file_type",
            4,
            removed_or_unknown.to_le_bytes().to_vec(),
        )]))
        .unwrap();
        assert_eq!(info.quantization, None);
        assert_eq!(info.file_type, Some(removed_or_unknown));
    }
}

#[test]
fn every_scalar_and_fixed_array_type_keeps_the_next_key_aligned() {
    let scalars = [
        (0, vec![12]),
        (1, vec![12]),
        (2, 12u16.to_le_bytes().to_vec()),
        (3, 12i16.to_le_bytes().to_vec()),
        (4, 12u32.to_le_bytes().to_vec()),
        (5, 12i32.to_le_bytes().to_vec()),
        (6, 12f32.to_le_bytes().to_vec()),
        (7, vec![1]),
        (10, 12u64.to_le_bytes().to_vec()),
        (11, 12i64.to_le_bytes().to_vec()),
        (12, 12f64.to_le_bytes().to_vec()),
    ];
    for (kind, payload) in scalars {
        let mut elements = payload.clone();
        elements.extend(&payload);
        let info = parse(&fixture(&[
            entry("unused.scalar", kind, payload),
            entry("unused.array", 9, array(kind, 2, &elements)),
            entry("llama.context_length", 4, 8192u32.to_le_bytes().to_vec()),
        ]))
        .unwrap();
        assert_eq!(info.context_length, Some(8192), "type {kind}");
    }
}

#[test]
fn boolean_arrays_use_one_byte_and_strings_and_nested_arrays_are_skipped() {
    let mut strings = string("first");
    strings.extend(string("second"));
    let info = parse(&fixture(&[
        entry("unused.bools", 9, array(7, 3, &[1, 0, 1])),
        entry("unused.strings", 9, array(8, 2, &strings)),
        entry("unused.nested", 9, array(9, 1, &array(7, 2, &[0, 1]))),
        entry("general.name", 8, string("still_aligned")),
    ]))
    .unwrap();
    assert_eq!(info.model_name.as_deref(), Some("still_aligned"));
}

#[test]
fn truncation_at_every_metadata_byte_returns_an_error() {
    let bytes = fixture(&[
        entry("unused.strings", 9, array(8, 1, &string("token"))),
        entry("unused.bools", 9, array(7, 2, &[1, 0])),
        entry("general.name", 8, string("full_name")),
        entry("llama.context_length", 10, 4096u64.to_le_bytes().to_vec()),
    ]);
    assert!(parse(&bytes).is_ok());
    for length in 0..bytes.len() {
        assert!(
            parse(&bytes[..length]).is_err(),
            "accepted truncation at {length}"
        );
    }
}

#[test]
fn supported_versions_and_big_endian_metadata_are_explicit() {
    let mut bytes = fixture(&[]);
    for version in [2u32, 3] {
        bytes[4..8].copy_from_slice(&version.to_le_bytes());
        assert!(parse(&bytes).is_ok());
    }
    for version in [0u32, 1, 4, u32::MAX] {
        bytes[4..8].copy_from_slice(&version.to_le_bytes());
        assert!(parse(&bytes).is_err());
    }
    let mut big = b"GGUF".to_vec();
    big.extend(3u32.to_be_bytes());
    big.extend(0u64.to_be_bytes());
    big.extend(1u64.to_be_bytes());
    let key = "llama.context_length";
    big.extend((key.len() as u64).to_be_bytes());
    big.extend(key.as_bytes());
    big.extend(10u32.to_be_bytes());
    big.extend(32_768u64.to_be_bytes());
    assert_eq!(parse(&big).unwrap().context_length, Some(32_768));
}

#[test]
fn untrusted_lengths_counts_and_types_are_rejected_without_allocation() {
    let mut excessive_count = fixture(&[]);
    excessive_count[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
    let mut excessive_key = fixture(&[]);
    excessive_key[16..24].copy_from_slice(&1u64.to_le_bytes());
    excessive_key.extend(u64::MAX.to_le_bytes());
    let cases = [
        excessive_count,
        excessive_key,
        fixture(&[entry("general.name", 8, u64::MAX.to_le_bytes().to_vec())]),
        fixture(&[entry("unused", 8, u64::MAX.to_le_bytes().to_vec())]),
        fixture(&[entry("unused", 9, array(10, u64::MAX, &[]))]),
        fixture(&[entry("unused", 9, array(8, u64::MAX, &[]))]),
        fixture(&[entry("unused", 9, array(99, 0, &[]))]),
        fixture(&[entry("unused", 99, vec![])]),
        fixture(&[entry("unused", 7, vec![2])]),
    ];
    for (index, bytes) in cases.iter().enumerate() {
        assert!(parse(bytes).is_err(), "case {index}");
    }
    let mut nested = array(7, 0, &[]);
    for _ in 0..20 {
        nested = array(9, 1, &nested);
    }
    assert!(parse(&fixture(&[entry("unused", 9, nested)])).is_err());
}

#[test]
fn integer_counts_are_checked_and_all_integer_parameter_types_work() {
    for (kind, payload) in [
        (0, vec![42]),
        (1, vec![42]),
        (2, 42u16.to_le_bytes().to_vec()),
        (3, 42i16.to_le_bytes().to_vec()),
        (4, 42u32.to_le_bytes().to_vec()),
        (5, 42i32.to_le_bytes().to_vec()),
        (10, 42u64.to_le_bytes().to_vec()),
        (11, 42i64.to_le_bytes().to_vec()),
    ] {
        let info = parse(&fixture(&[
            entry("general.parameter_count", kind, payload.clone()),
            entry("llama.context_length", kind, payload),
        ]))
        .unwrap();
        assert_eq!(info.parameters_billions, Some(42.0 / 1_000_000_000.0));
        assert_eq!(info.context_length, Some(42));
    }
    for key in [
        "llama.context_length",
        "llama.expert_count",
        "llama.block_count",
        "general.file_type",
    ] {
        assert!(parse(&fixture(&[entry(
            key,
            10,
            (u64::from(u32::MAX) + 1).to_le_bytes().to_vec()
        )]))
        .is_err());
        assert!(parse(&fixture(&[entry(key, 11, (-1i64).to_le_bytes().to_vec())])).is_err());
    }
    for value in [f64::NAN, f64::INFINITY, -1.0] {
        assert!(parse(&fixture(&[entry(
            "general.parameter_count",
            12,
            value.to_le_bytes().to_vec()
        )]))
        .is_err());
    }
    assert!(parse(&fixture(&[entry(
        "general.parameter_count",
        11,
        (-1i64).to_le_bytes().to_vec()
    )]))
    .is_err());
    assert_eq!(
        parse(&fixture(&[entry(
            "general.parameter_count",
            12,
            1e9f64.to_le_bytes().to_vec()
        )]))
        .unwrap()
        .parameters_billions,
        Some(1.0)
    );
}

#[test]
fn invalid_utf8_or_wrong_authoritative_metadata_types_are_errors() {
    let mut bad_string = 1u64.to_le_bytes().to_vec();
    bad_string.push(0xff);
    assert!(parse(&fixture(&[entry("general.name", 8, bad_string)])).is_err());
    for key in ["general.name", "general.architecture"] {
        assert!(parse(&fixture(&[entry(key, 4, 1u32.to_le_bytes().to_vec())])).is_err());
    }
    assert!(parse(&fixture(&[entry(
        "llama.context_length",
        8,
        string("4096")
    )]))
    .is_err());
}

#[test]
fn tiny_tensor_payload_and_filename_do_not_invent_metadata_identity() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("model-Q5_K_M.gguf");
    let mut bytes = fixture(&[entry("general.name", 8, string("small_fixture"))]);
    bytes[8..16].copy_from_slice(&1u64.to_le_bytes());
    bytes.extend([0xff; 32]);
    std::fs::write(&path, &bytes).unwrap();
    let info = parse_gguf(path.to_str().unwrap()).unwrap();
    assert_eq!(info.model_name.as_deref(), Some("small_fixture"));
    assert_eq!(info.quantization, None);
    assert_eq!(info.file_type, None);
}

#[test]
fn oversized_array_in_a_tiny_truncated_file_is_rejected() {
    let bytes = fixture(&[entry("unused", 9, array(0, 257 * 1024 * 1024, &[]))]);
    assert!(parse(&bytes).is_err());
}

#[test]
fn absent_numeric_metadata_and_explicit_zero_have_distinct_snapshots() {
    let absent = parse(&fixture(&[])).unwrap();
    let zero = parse(&fixture(&[
        entry("general.parameter_count", 10, 0u64.to_le_bytes().to_vec()),
        entry("llama.expert_count", 4, 0u32.to_le_bytes().to_vec()),
        entry("llama.block_count", 4, 0u32.to_le_bytes().to_vec()),
        entry("llama.context_length", 4, 0u32.to_le_bytes().to_vec()),
    ]))
    .unwrap();
    assert_eq!(absent.expert_count, None);
    assert_eq!(absent.block_count, None);
    assert_eq!(absent.parameters_billions, None);
    assert_eq!(absent.context_length, None);
    assert_eq!(zero.expert_count, Some(0));
    assert_eq!(zero.block_count, Some(0));
    assert_eq!(zero.parameters_billions, Some(0.0));
    assert_eq!(zero.context_length, Some(0));
    assert!(!absent.is_moe);
    assert!(!zero.is_moe);
    let absent = serde_json::to_value(absent).unwrap();
    let zero = serde_json::to_value(zero).unwrap();
    assert_eq!(absent.as_object().unwrap().len(), 9);
    assert_eq!(zero.as_object().unwrap().len(), 9);
    for key in [
        "expert_count",
        "block_count",
        "parameters_billions",
        "context_length",
    ] {
        assert!(absent[key].is_null(), "{key} must remain unknown");
        assert_eq!(zero[key].as_f64(), Some(0.0), "{key} must retain zero");
    }
}

#[test]
fn invalid_or_missing_files_return_errors_and_directory_names_do_not_guess_quant() {
    let dir = tempdir().unwrap();
    assert!(parse_gguf(dir.path().join("missing.gguf").to_str().unwrap()).is_err());
    assert!(parse(b"not GGUF").is_err());
    let quant_dir = dir.path().join("Q8_0");
    std::fs::create_dir(&quant_dir).unwrap();
    let path = quant_dir.join("unknown.gguf");
    std::fs::write(&path, fixture(&[])).unwrap();
    assert_eq!(
        parse_gguf(path.to_str().unwrap()).unwrap().quantization,
        None
    );
}
