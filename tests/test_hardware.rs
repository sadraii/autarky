//! Hardware detection tests

use autarky::hardware::detect_hardware;

#[tokio::test]
async fn test_detect_hardware_returns_valid_info() {
    let info = detect_hardware().await;
    assert!(info.is_ok());

    let info = info.unwrap();
    assert!(
        !info.gpu_family.is_empty(),
        "GPU family should not be empty"
    );
    assert!(
        !info.os_platform.is_empty(),
        "OS platform should not be empty"
    );
}

#[tokio::test]
async fn test_hardware_detection_is_deterministic() {
    let info1 = detect_hardware().await.unwrap();
    let info2 = detect_hardware().await.unwrap();

    assert_eq!(info1.gpu_family, info2.gpu_family);
}

#[tokio::test]
async fn cpu_selection_is_valid_without_vram() {
    let info = autarky::hardware::detect_hardware_for_flags(&["--device=none".into()])
        .await
        .unwrap();
    assert_eq!(info.vram_gb, 0.0);
    assert_eq!(info.devices.len(), 1);
    assert_eq!(info.devices[0].id, "CPU");
}

#[tokio::test]
async fn unresolved_explicit_device_is_an_error_not_an_invented_gpu() {
    let info = autarky::hardware::detect_hardware_for_server(
        "/does/not/exist",
        &["--device=CUDA2".into()],
    )
    .await;
    assert!(info.is_err());
    let unknown = autarky::hardware::detect_hardware_for_server("/does/not/exist", &[])
        .await
        .unwrap();
    assert!(unknown.gpu_family.starts_with("Unknown"));
    assert!(unknown.devices.is_empty());
}
