use super::*;

#[test]
fn actual_file_load_rejects_unknown_fields_without_echoing_secrets() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("config.toml");
    for unknown in ["unknown_setting = 123", "default_repeat = 7"] {
        std::fs::write(
            &path,
            format!("default_repeats = 7\napi_key = 'fixture-private-key'\n{unknown}\n"),
        )
        .unwrap();
        let error = Config::load_from(&path).unwrap_err();
        for output in [format!("{error:#}"), format!("{error:?}")] {
            assert!(!output.contains("fixture-private-key"));
            assert!(!output.contains(unknown));
        }
    }
}

#[test]
fn identity_survives_actual_replacement_and_repeated_directory_sync_failure() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("config.toml");
    let mut config = Config::default();
    config.set("default_repeats", "7").unwrap();
    config.save_to(&path).unwrap();
    let old_bytes = std::fs::read(&path).unwrap();
    let error = config
        .ensure_api_key_at_with_directory_sync(&path, |parent| {
            assert_eq!(parent, directory.path());
            let visible = Config::load_from(&path).unwrap();
            assert!(visible.api_key.is_some(), "replacement must precede sync");
            Err(std::io::Error::from_raw_os_error(5))
        })
        .unwrap_err();
    let visible = Config::load_from(&path).unwrap();
    let key = visible.api_key.unwrap();
    assert!(matches!(
        error.downcast_ref::<ConfigSaveError>(),
        Some(ConfigSaveError::DurabilityUnconfirmed(_))
    ));
    assert_eq!(config.api_key.as_ref(), Some(&key));
    assert_ne!(std::fs::read(&path).unwrap(), old_bytes);
    let replaced_bytes = std::fs::read(&path).unwrap();
    for output in [
        format!("{error:#}"),
        format!("{error:?}"),
        format!("{config:?}"),
    ] {
        assert!(!output.contains(&key));
    }
    for _ in 0..2 {
        let mut synced = false;
        let error = config
            .ensure_api_key_at_with_directory_sync(&path, |_| {
                synced = true;
                Err(std::io::Error::from_raw_os_error(5))
            })
            .unwrap_err();
        assert!(matches!(
            error.downcast_ref::<ConfigSaveError>(),
            Some(ConfigSaveError::DurabilityUnconfirmed(_))
        ));
        assert!(synced, "retry must not silently claim durability");
        assert_eq!(config.api_key.as_ref(), Some(&key));
        assert_eq!(std::fs::read(&path).unwrap(), replaced_bytes);
    }
    let mut synced = false;
    assert_eq!(
        config
            .ensure_api_key_at_with_directory_sync(&path, |parent| {
                synced = true;
                sync_config_directory(parent)
            })
            .unwrap(),
        key
    );
    assert!(synced);
    assert_eq!(std::fs::read(&path).unwrap(), replaced_bytes);
    let mut reloaded = Config::load_from(&path).unwrap();
    assert_eq!(reloaded.api_key.as_ref(), Some(&key));
    assert!(reloaded
        .ensure_api_key_at_with_directory_sync(&path, |_| {
            Err(std::io::Error::from_raw_os_error(5))
        })
        .is_err());
    assert_eq!(reloaded.api_key.as_ref(), Some(&key));
    assert_eq!(reloaded.ensure_api_key_at(&path).unwrap(), key);
    assert_eq!(std::fs::read(&path).unwrap(), replaced_bytes);
}

#[test]
fn failed_replacement_keeps_the_old_file_and_unset_identity() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("config.toml");
    std::fs::create_dir(&path).unwrap();
    let sentinel = path.join("old-file");
    std::fs::write(&sentinel, "unchanged").unwrap();
    let mut config = Config::default();
    let mut synced = false;
    let error = config
        .ensure_api_key_at_with_directory_sync(&path, |_| {
            synced = true;
            Ok(())
        })
        .unwrap_err();
    assert!(matches!(
        error.downcast_ref::<ConfigSaveError>(),
        Some(ConfigSaveError::BeforeReplacement(_))
    ));
    assert!(
        !synced,
        "directory sync follows successful replacement only"
    );
    assert!(config.api_key.is_none());
    assert_eq!(std::fs::read_to_string(&sentinel).unwrap(), "unchanged");
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
}

#[test]
fn ordinary_save_reports_visible_replacement_on_directory_sync_failure() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("config.toml");
    let mut config = Config::default();
    config.set("api_key", "fixture-private-key").unwrap();
    config.save_to(&path).unwrap();
    config.set("default_repeats", "7").unwrap();
    let error = config
        .save_to_with_directory_sync(&path, |_| Err(std::io::Error::from_raw_os_error(5)))
        .unwrap_err();
    assert!(matches!(error, ConfigSaveError::DurabilityUnconfirmed(_)));
    let visible = Config::load_from(&path).unwrap();
    assert_eq!(visible.api_key, config.api_key);
    assert_eq!(visible.default_repeats, Some(7));
    assert!(!format!("{error:?}").contains("fixture-private-key"));
    config.save_to(&path).unwrap();
}

#[test]
fn settings_validate_without_silent_fallbacks() {
    let mut config = Config::default();
    assert!(config.get_server_url().is_err());
    for (key, invalid) in [
        ("default_repeats", "0"),
        ("default_repeats", "1001"),
        ("default_repeats", "x"),
        ("default_prefill_context", "2"),
        ("default_context_lengths", "1,bad"),
        ("default_context_lengths", "3"),
        ("default_context_lengths", ""),
        ("extra_flags", "--threads 8"),
        ("server_url", "http://public.example"),
        ("models_dir", "unused"),
        ("api_key", ""),
    ] {
        let before = serde_json::to_value(&config).unwrap();
        assert!(config.set(key, invalid).is_err(), "{key}");
        assert_eq!(serde_json::to_value(&config).unwrap(), before);
    }
    for (key, value) in [
        ("default_repeats", "5"),
        ("default_prefill_context", "1"),
        ("default_context_lengths", "4,1,2"),
        ("extra_flags", r#"["--threads","8"]"#),
        ("server_url", "https://community.example/prefix/"),
    ] {
        config.set(key, value).unwrap();
        assert_eq!(config.get(key).unwrap(), value);
    }
    assert_eq!(
        config.get_server_url().unwrap(),
        "https://community.example/prefix/"
    );
    assert!(config.get("unknown").is_err());
}

#[test]
fn diagnostic_views_never_contain_stored_secrets() {
    let mut config = Config::default();
    config.set("api_key", "fixture-private-key").unwrap();
    config.set("extra_flags", r#"["--api-key=fixture-child-key","-m","/private/location/model.gguf","--threads","8"]"#).unwrap();
    for output in [
        format!("{config:?}"),
        config.masked().to_string(),
        config.get("api_key").unwrap(),
        config.get("extra_flags").unwrap(),
    ] {
        assert!(!output.contains("fixture-private-key"));
        assert!(!output.contains("fixture-child-key"));
        assert!(!output.contains("/private/location"));
    }
    assert!(config.get("extra_flags").unwrap().contains("--threads"));
    assert_eq!(config.get("api_key").unwrap(), "<redacted>");
}

#[test]
fn atomic_save_preserves_the_old_file_on_validation_failure() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("config.toml");
    let mut config = Config::default();
    config.set("default_repeats", "7").unwrap();
    config.save_to(&path).unwrap();
    let before = std::fs::read(&path).unwrap();
    config.default_prefill_context = Some(99);
    assert!(config.save_to(&path).is_err());
    let error = config.ensure_api_key_at(&path).unwrap_err();
    assert!(matches!(
        error.downcast_ref::<ConfigSaveError>(),
        Some(ConfigSaveError::BeforeReplacement(_))
    ));
    assert!(config.api_key.is_none());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(Config::load_from(&path).unwrap().get_default_repeats(), 7);
    assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
}

#[test]
fn identity_generation_reports_persistence_failure_and_is_stable_on_success() {
    let directory = tempfile::tempdir().unwrap();
    let blocker = directory.path().join("not-a-directory");
    std::fs::write(&blocker, "fixture").unwrap();
    let mut config = Config::default();
    assert!(config
        .ensure_api_key_at(&blocker.join("config.toml"))
        .is_err());
    assert!(config.api_key.is_none());
    let path = directory.path().join("private").join("config.toml");
    let key = config.ensure_api_key_at(&path).unwrap();
    assert_eq!(config.ensure_api_key_at(&path).unwrap(), key);
    assert_eq!(
        Config::load_from(&path).unwrap().api_key.as_ref(),
        Some(&key)
    );
}

#[test]
fn configuration_parse_errors_do_not_echo_a_secret_source_line() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("config.toml");
    std::fs::write(&path, "api_key = ['fixture-private-key']").unwrap();
    let error = Config::load_from(&path).unwrap_err();
    assert!(!format!("{error:#}").contains("fixture-private-key"));
}

#[test]
fn legacy_http_configuration_does_not_prevent_local_benchmarks() {
    let config = Config {
        server_url: Some("http://old-public.example:8080".into()),
        ..Config::default()
    };
    config.validate().unwrap();
    assert!(config.get_server_url().is_err());
}

#[cfg(unix)]
#[test]
fn persisted_identity_and_existing_legacy_files_have_private_permissions() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("private").join("config.toml");
    let mut config = Config::default();
    config.ensure_api_key_at(&path).unwrap();
    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert_eq!(
        std::fs::metadata(path.parent().unwrap())
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o700
    );
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
    std::fs::set_permissions(
        path.parent().unwrap(),
        std::fs::Permissions::from_mode(0o777),
    )
    .unwrap();
    Config::load_from(&path).unwrap();
    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
    assert_eq!(
        std::fs::metadata(path.parent().unwrap())
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o700
    );
}
