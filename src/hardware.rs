//! Hardware identity from the selected llama.cpp backend, not a physical GPU guess.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HardwareDevice {
    /// Backend-local identifier. Never include this field in public upload payloads.
    pub id: String,
    pub name: String,
    pub memory_gb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub gpu_family: String,
    pub vram_gb: f64,
    pub system_ram_gb: u64,
    pub os_platform: String,
    #[serde(default)]
    pub devices: Vec<HardwareDevice>,
}

pub async fn detect_hardware() -> Result<HardwareInfo> {
    detect_hardware_for_flags(&[]).await
}

/// Uses llama-server from PATH. Use detect_hardware_for_server when launching
/// another binary, so device numbering uses exactly the same backend build.
pub async fn detect_hardware_for_flags(flags: &[String]) -> Result<HardwareInfo> {
    detect_hardware_for_server("llama-server", flags).await
}

pub async fn detect_hardware_for_server(
    server_path: &str,
    flags: &[String],
) -> Result<HardwareInfo> {
    let selection = Selection::parse(flags)?;
    if selection.cpu {
        return Ok(hardware(vec![cpu_device()]));
    }
    let mut command = tokio::process::Command::new(server_path);
    command.arg("--list-devices");
    // Native enumeration inherits CUDA/HIP/ROCR/Vulkan visible-device mappings.
    // Physical nvidia-smi indices cannot be substituted for these local IDs.
    let output =
        crate::utils::bounded_command_output(&mut command, Duration::from_secs(5), 256 * 1024)
            .await;
    let devices = match output {
        Ok(output) if output.status.success() => {
            parse_native_devices(&String::from_utf8_lossy(&output.stdout))?
        }
        _ => {
            anyhow::ensure!(
                !selection.explicit(),
                "Cannot resolve selected devices: llama-server --list-devices is unavailable"
            );
            return Ok(HardwareInfo {
                gpu_family: "Unknown (native backend unavailable)".into(),
                vram_gb: 0.0,
                system_ram_gb: system_ram(),
                os_platform: std::env::consts::OS.into(),
                devices: vec![],
            });
        }
    };
    if devices.is_empty() {
        anyhow::ensure!(
            !selection.explicit(),
            "The native backend reports no selected GPU devices"
        );
        return Ok(hardware(vec![cpu_device()]));
    }
    Ok(hardware(selection.select(devices)?))
}

fn system_ram() -> u64 {
    sys_info::mem_info()
        .map(|info| info.total / 1024 / 1024)
        .unwrap_or(0)
}

fn cpu_device() -> HardwareDevice {
    let name = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                line.strip_prefix("model name")
                    .and_then(|rest| rest.split_once(':'))
                    .map(|(_, name)| name.trim().to_string())
            })
        })
        .unwrap_or_else(|| std::env::consts::ARCH.to_string());
    HardwareDevice {
        id: "CPU".into(),
        name: format!("CPU: {}", name),
        memory_gb: 0.0,
    }
}

fn hardware(devices: Vec<HardwareDevice>) -> HardwareInfo {
    HardwareInfo {
        gpu_family: devices
            .iter()
            .map(|device| device.name.as_str())
            .collect::<Vec<_>>()
            .join(" + "),
        vram_gb: devices.iter().map(|device| device.memory_gb).sum(),
        system_ram_gb: system_ram(),
        os_platform: std::env::consts::OS.into(),
        devices,
    }
}

fn parse_native_devices(text: &str) -> Result<Vec<HardwareDevice>> {
    let (_, lines) = text
        .split_once("Available devices:")
        .context("Native backend did not report a device list")?;
    let mut devices = Vec::new();
    for line in lines.lines().filter(|line| !line.trim().is_empty()) {
        let (id, description) = line
            .trim()
            .split_once(':')
            .context("Malformed native device identity")?;
        let (name, memory) = description
            .rsplit_once(" (")
            .context("Missing native device memory")?;
        let memory = memory
            .split_once(" MiB,")
            .context("Unsupported native device memory units")?
            .0
            .parse::<f64>()?;
        anyhow::ensure!(
            memory.is_finite() && memory >= 0.0,
            "Invalid native device memory"
        );
        anyhow::ensure!(
            !id.is_empty()
                && !name.trim().is_empty()
                && !devices
                    .iter()
                    .any(|device: &HardwareDevice| device.id == id),
            "Invalid or duplicate native device ID"
        );
        devices.push(HardwareDevice {
            id: id.to_string(),
            name: clean_gpu_name(name.trim()),
            memory_gb: memory / 1024.0,
        });
    }
    Ok(devices)
}

#[derive(Default)]
struct Selection {
    cpu: bool,
    devices: Option<String>,
    main: Option<i32>,
    split: Option<Vec<f64>>,
    mode: String,
}
impl Selection {
    fn parse(flags: &[String]) -> Result<Self> {
        Self::parse_with_env(flags, |name| std::env::var(name).ok())
    }
    fn parse_with_env(flags: &[String], env: impl Fn(&str) -> Option<String>) -> Result<Self> {
        let parsed = crate::arguments::parse(flags)?;
        let value = |aliases: &[&str], variable: &str| -> Result<Option<String>> {
            Ok(crate::arguments::last_value(&parsed, aliases)
                .map(str::to_owned)
                .or_else(|| env(variable)))
        };
        let devices = value(&["-dev", "--device"], "LLAMA_ARG_DEVICE")?;
        let layers = value(
            &["-ngl", "--gpu-layers", "--n-gpu-layers"],
            "LLAMA_ARG_N_GPU_LAYERS",
        )?;
        let main = value(&["-mg", "--main-gpu"], "LLAMA_ARG_MAIN_GPU")?
            .map(|value| value.parse())
            .transpose()
            .context("Invalid main GPU index")?;
        let split = value(&["-ts", "--tensor-split"], "LLAMA_ARG_TENSOR_SPLIT")?
            .map(|value| {
                value
                    .split([',', '/'])
                    .map(str::parse::<f64>)
                    .collect::<std::result::Result<Vec<_>, _>>()
            })
            .transpose()?;
        if let Some(split) = &split {
            anyhow::ensure!(
                !split.is_empty() && split.iter().all(|value| value.is_finite() && *value >= 0.0),
                "Invalid tensor split"
            );
        }
        let mode = value(&["-sm", "--split-mode"], "LLAMA_ARG_SPLIT_MODE")?
            .unwrap_or_else(|| "layer".into());
        anyhow::ensure!(
            matches!(mode.as_str(), "none" | "layer" | "row" | "tensor"),
            "Invalid split mode"
        );
        Ok(Self {
            cpu: devices.as_deref() == Some("none")
                || layers
                    .as_deref()
                    .is_some_and(|value| value.parse::<i32>().ok() == Some(0))
                || (mode == "none" && main.is_some_and(|index| index < 0)),
            devices,
            main,
            split,
            mode,
        })
    }
    fn explicit(&self) -> bool {
        self.devices.is_some() || self.main.is_some() || self.split.is_some()
    }
    fn select(&self, available: Vec<HardwareDevice>) -> Result<Vec<HardwareDevice>> {
        let devices = if let Some(ids) = &self.devices {
            let mut selected = Vec::new();
            for id in ids.split(',') {
                let device = available
                    .iter()
                    .find(|device| device.id == id)
                    .with_context(|| format!("Selected backend device {} is unavailable", id))?;
                anyhow::ensure!(
                    !selected
                        .iter()
                        .any(|device: &HardwareDevice| device.id == id),
                    "Duplicate selected backend device"
                );
                selected.push(device.clone());
            }
            selected
        } else {
            // Multiple backend implementations can expose the same physical GPU.
            // Do not invent which backend the native loader will prefer.
            let backend = |id: &str| {
                id.trim_end_matches(|c: char| c.is_ascii_digit())
                    .to_string()
            };
            anyhow::ensure!(
                available
                    .iter()
                    .all(|device| backend(&device.id) == backend(&available[0].id)),
                "Multiple native backends are available; select --device explicitly"
            );
            available
        };
        let main: usize = self
            .main
            .unwrap_or(0)
            .try_into()
            .context("Negative main GPU requires split mode none (CPU)")?;
        anyhow::ensure!(
            main < devices.len(),
            "Main GPU index is outside the selected device list"
        );
        if self.mode == "none" {
            return Ok(vec![devices[main].clone()]);
        }
        let Some(split) = &self.split else {
            return Ok(devices);
        };
        anyhow::ensure!(
            split.len() <= devices.len(),
            "Tensor split exceeds the selected device count"
        );
        if split.iter().all(|value| *value == 0.0) {
            return Ok(devices);
        }
        Ok(devices
            .into_iter()
            .enumerate()
            .filter_map(|(index, device)| {
                (split.get(index).is_some_and(|value| *value > 0.0)
                    || (self.mode == "row" && index == main))
                    .then_some(device)
            })
            .collect())
    }
}

fn clean_gpu_name(name: &str) -> String {
    let prefixes = [
        "NVIDIA GeForce ",
        "NVIDIA ",
        "AMD Radeon ",
        "AMD ",
        "Intel Arc ",
        "Intel ",
        "Tesla ",
        "Quadro ",
    ];
    let cleaned = prefixes
        .iter()
        .find_map(|prefix| name.strip_prefix(prefix))
        .unwrap_or(name);
    cleaned
        .strip_suffix(" GPU")
        .or_else(|| cleaned.strip_suffix(" Graphics"))
        .unwrap_or(cleaned)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    const DEVICES: &str = "Available devices:\n  CUDA0: NVIDIA GeForce First GPU (8192 MiB, 4096 MiB free)\n  CUDA1: NVIDIA GeForce 第二 GPU (0 MiB, 0 MiB free)\n";
    fn selected(flags: &[&str]) -> Result<Vec<HardwareDevice>> {
        Selection::parse_with_env(
            &flags.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            |_| None,
        )?
        .select(parse_native_devices(DEVICES)?)
    }
    #[test]
    fn selection_uses_backend_ids_and_main_gpu_in_explicit_order() {
        let devices = selected(&["-dev", "CUDA1,CUDA0", "--split-mode=none", "-mg", "1"]).unwrap();
        assert_eq!(devices[0].id, "CUDA0");
        assert!(selected(&["--device=CUDA9"]).is_err());
        assert!(selected(&["--main-gpu=2"]).is_err());
    }
    #[test]
    fn every_native_selection_alias_matches_without_changing_values() {
        for device in ["-dev", "--device"] {
            for mode in ["-sm", "--split-mode", "--split_mode"] {
                for main in ["-mg", "--main-gpu", "--main_gpu"] {
                    for inline in [true, false] {
                        let flags = [(device, "CUDA1,CUDA0"), (mode, "none"), (main, "1")]
                            .into_iter()
                            .flat_map(|(key, value)| {
                                if inline {
                                    vec![format!("{key}={value}")]
                                } else {
                                    vec![key.into(), value.into()]
                                }
                            })
                            .collect::<Vec<String>>();
                        let choice = Selection::parse_with_env(&flags, |_| None).unwrap();
                        assert_eq!(choice.devices.as_deref(), Some("CUDA1,CUDA0"));
                        assert_eq!(
                            choice
                                .select(parse_native_devices(DEVICES).unwrap())
                                .unwrap()[0]
                                .id,
                            "CUDA0"
                        );
                    }
                }
            }
        }
        for key in ["-ts", "--tensor-split", "--tensor_split"] {
            for flags in [vec![key.into(), "0,1".into()], vec![format!("{key}=0,1")]] {
                let choice = Selection::parse_with_env(&flags, |_| None).unwrap();
                assert_eq!(
                    choice
                        .select(parse_native_devices(DEVICES).unwrap())
                        .unwrap()[0]
                        .id,
                    "CUDA1"
                );
            }
        }
        let choice =
            Selection::parse_with_env(&["--device=DEVICE_with_under_score".into()], |_| None)
                .unwrap();
        assert_eq!(choice.devices.as_deref(), Some("DEVICE_with_under_score"));
    }

    #[test]
    fn tensor_split_uses_all_selected_devices_and_keeps_zero_vram() {
        let devices = selected(&["--tensor-split=0,1"]).unwrap();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].name, "第二");
        assert_eq!(devices[0].memory_gb, 0.0);
        assert_eq!(selected(&["--tensor-split=1,1"]).unwrap().len(), 2);
        assert!(selected(&["--tensor-split=NaN,1"]).is_err());
    }
    #[test]
    fn cpu_flags_and_environment_are_explicit() {
        for option in ["--alias", "--api_key", "--prompt", "--lora"] {
            let flags = ["--gpu_layers", "1", option, "--gpu-layers=0"].map(str::to_string);
            assert!(!Selection::parse_with_env(&flags, |_| None).unwrap().cpu);
        }
        for key in [
            "-ngl",
            "--gpu-layers",
            "--gpu_layers",
            "--n-gpu-layers",
            "--n_gpu_layers",
        ] {
            for flags in [vec![key.into(), "0".into()], vec![format!("{key}=0")]] {
                assert!(
                    Selection::parse_with_env(&flags, |_| None).unwrap().cpu,
                    "{flags:?} must select CPU"
                );
            }
        }
        let selection = Selection::parse_with_env(&[], |key| {
            (key == "LLAMA_ARG_DEVICE").then(|| "none".into())
        })
        .unwrap();
        assert!(selection.cpu);
        let selection = Selection::parse_with_env(&["-ngl=0".into()], |_| None).unwrap();
        assert!(selection.cpu);
        let selection = Selection::parse_with_env(
            &["--split-mode=none".into(), "--main-gpu=-1".into()],
            |_| None,
        )
        .unwrap();
        assert!(selection.cpu);
        let info = hardware(vec![cpu_device()]);
        assert_eq!(info.vram_gb, 0.0);
        assert!(info.gpu_family.starts_with("CPU:"));
    }
    #[test]
    fn native_visibility_ids_are_not_replaced_with_physical_indices() {
        let devices = parse_native_devices(
            "Available devices:\n CUDA0: Visible physical GPU 7 (16384 MiB, 12000 MiB free)\n",
        )
        .unwrap();
        let selected = Selection::parse_with_env(&["--device=CUDA0".into()], |_| None)
            .unwrap()
            .select(devices)
            .unwrap();
        assert_eq!(selected[0].id, "CUDA0");
        assert!(selected[0].name.contains("GPU 7"));
    }
    #[test]
    fn old_hardware_json_keeps_default_device_list() {
        let info: HardwareInfo = serde_json::from_str(
            r#"{"gpu_family":"old","vram_gb":0,"system_ram_gb":1,"os_platform":"linux"}"#,
        )
        .unwrap();
        assert!(info.devices.is_empty());
    }
}
