//! Config module
//! Handles loading/saving configuration from ~/.config/autarky/config.toml

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize, Default, Clone)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Default context lengths for bench command (comma-separated)
    pub default_context_lengths: Option<String>,
    /// Default number of measurement repeats per context length
    pub default_repeats: Option<usize>,
    /// Run one unmeasured warmup before cold-cache measurements (0 or 1)
    pub default_prefill_context: Option<usize>,
    /// Additional flags to pass to llama-server
    pub extra_flags: Option<Vec<String>>,
    /// Community server URL
    pub server_url: Option<String>,
    /// Private identifier for anonymous community submission attribution
    pub api_key: Option<String>,
}

/// A failed save can leave the replacement visible without confirmed durability.
#[derive(Debug, thiserror::Error)]
pub enum ConfigSaveError {
    #[error("Config replacement did not occur: {0}")]
    BeforeReplacement(#[from] anyhow::Error),
    #[error(
        "config.toml replacement is visible, but directory sync failed; durability is unconfirmed"
    )]
    DurabilityUnconfirmed(#[source] std::io::Error),
}

impl std::fmt::Debug for Config {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.masked().fmt(formatter)
    }
}

fn private_config_directory(path: &Path) -> Result<&Path> {
    let parent = path.parent().context("Config path has no parent")?;
    std::fs::create_dir_all(parent).context("Cannot create the config directory")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700))
            .context("Cannot protect the config directory")?;
    }
    Ok(parent)
}

fn sync_config_directory(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    std::fs::File::open(path)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

impl Config {
    /// Get config file path
    pub fn config_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("autarky")
            .join("config.toml")
    }

    /// Load config from file, returning default if not exists
    pub fn load() -> Result<Self> {
        Self::load_from(&Self::config_path())
    }

    fn load_from(path: &Path) -> Result<Self> {
        private_config_directory(path)?;
        let content = match std::fs::read_to_string(path) {
            Ok(content) => content,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self::default())
            }
            Err(error) => return Err(error).context("Failed to read config.toml"),
        };
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
                .context("Cannot protect config.toml permissions")?;
        }
        // Do not include TOML parser excerpts: they can contain the private key.
        let config: Self = toml::from_str(&content).map_err(|_| {
            anyhow::anyhow!("Invalid config.toml; check its field names, types and syntax")
        })?;
        config.validate()?;
        Ok(config)
    }

    /// Save config to file
    pub fn save(&self) -> Result<()> {
        self.save_to(&Self::config_path())
    }

    fn save_to(&self, path: &Path) -> Result<()> {
        self.save_to_with_directory_sync(path, sync_config_directory)
            .map_err(Into::into)
    }

    fn save_to_with_directory_sync(
        &self,
        path: &Path,
        sync_directory: impl FnOnce(&Path) -> std::io::Result<()>,
    ) -> std::result::Result<(), ConfigSaveError> {
        self.validate()?;
        let parent = private_config_directory(path)?;
        let mut temporary = tempfile::NamedTempFile::new_in(parent)
            .context("Cannot create a private config file")?;
        let content = toml::to_string_pretty(self).context("Cannot serialize configuration")?;
        temporary
            .write_all(content.as_bytes())
            .context("Cannot write configuration")?;
        temporary
            .as_file()
            .sync_all()
            .context("Cannot sync configuration")?;
        temporary
            .persist(path)
            .map_err(|error| error.error)
            .context("Cannot atomically replace config.toml")?;
        sync_directory(parent).map_err(ConfigSaveError::DurabilityUnconfirmed)?;
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        if self
            .default_repeats
            .is_some_and(|value| !(1..=1000).contains(&value))
        {
            bail!("default_repeats must be between 1 and 1000");
        }
        if self.default_prefill_context.is_some_and(|value| value > 1) {
            bail!("default_prefill_context must be 0 or 1");
        }
        if let Some(contexts) = &self.default_context_lengths {
            for context in contexts.split(',') {
                let value: usize = context
                    .trim()
                    .parse()
                    .context("Invalid default_context_lengths")?;
                if !crate::launcher::SUPPORTED_CONTEXT_LENGTHS.contains(&value) {
                    bail!("default_context_lengths must use supported K indices: 1,2,4,8,16,32,64,128,256");
                }
            }
        }
        if self
            .extra_flags
            .as_ref()
            .is_some_and(|flags| flags.iter().any(|flag| flag.contains('\0')))
        {
            bail!("extra_flags must not contain a NUL byte");
        }
        if let Some(key) = &self.api_key {
            crate::community::key_header(key)?;
        }
        if let Some(base) = &self.server_url {
            let url =
                reqwest::Url::parse(base).map_err(|_| anyhow::anyhow!("Invalid server_url"))?;
            if !matches!(url.scheme(), "http" | "https")
                || url.host().is_none()
                || !url.username().is_empty()
                || url.password().is_some()
                || url.query().is_some()
                || url.fragment().is_some()
            {
                bail!("server_url must be an HTTP(S) origin or path prefix without credentials, query or fragment");
            }
        }
        Ok(())
    }

    /// Get default repeats count
    pub fn get_default_repeats(&self) -> usize {
        self.default_repeats.unwrap_or(3)
    }

    /// Get default prefill context setting
    pub fn get_default_prefill_context(&self) -> usize {
        self.default_prefill_context.unwrap_or(0)
    }

    /// Sharing has no default endpoint. A legacy HTTP endpoint does not stop local work.
    pub fn get_server_url(&self) -> Result<String> {
        let base = self
            .server_url
            .as_ref()
            .context("Community sharing is not configured; set server_url to an HTTPS endpoint")?;
        crate::community::endpoint(base, "")?;
        Ok(base.clone())
    }

    /// Get or generate API key
    pub fn get_or_create_api_key(&mut self) -> Result<String> {
        self.ensure_api_key_at(&Self::config_path())
    }

    fn ensure_api_key_at(&mut self, path: &Path) -> Result<String> {
        self.ensure_api_key_at_with_directory_sync(path, sync_config_directory)
    }

    fn ensure_api_key_at_with_directory_sync(
        &mut self,
        path: &Path,
        sync_directory: impl FnOnce(&Path) -> std::io::Result<()>,
    ) -> Result<String> {
        if let Some(ref key) = self.api_key {
            crate::community::key_header(key)?;
            // An earlier replacement may have succeeded before directory sync failed.
            // Confirm durability on every reuse, without another write or a hidden flag.
            let parent = path.parent().context("Config path has no parent")?;
            sync_directory(parent)
                .map_err(ConfigSaveError::DurabilityUnconfirmed)
                .context("Cannot confirm durability of the community identity")?;
            return Ok(key.clone());
        }
        let key = uuid::Uuid::new_v4().to_string();
        self.api_key = Some(key.clone());
        if let Err(error) = self.save_to_with_directory_sync(path, sync_directory) {
            if matches!(error, ConfigSaveError::BeforeReplacement(_)) {
                self.api_key = None;
            }
            return Err(error).context("Cannot persist the community identity");
        }
        Ok(key)
    }

    pub fn masked(&self) -> serde_json::Value {
        serde_json::json!({
            "default_context_lengths": self.default_context_lengths,
            "default_repeats": self.get_default_repeats(),
            "default_prefill_context": self.get_default_prefill_context(),
            "extra_flags": self.extra_flags.as_ref().map(|flags| crate::db::sanitize_flags(flags)),
            "server_url": self.server_url,
            "api_key": self.api_key.as_ref().map(|_| "<redacted>")
        })
    }

    pub fn get(&self, key: &str) -> Result<String> {
        let masked = self.masked();
        let value = masked.get(key).context("Unknown configuration key")?;
        Ok(match value {
            serde_json::Value::String(value) => value.clone(),
            other => other.to_string(),
        })
    }

    /// Validate before changing the object. Persist only after the caller accepts the change.
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        let mut candidate = self.clone();
        match key {
            "default_context_lengths" => {
                candidate.default_context_lengths = Some(value.to_string())
            }
            "default_repeats" => {
                candidate.default_repeats = Some(
                    value
                        .parse()
                        .context("default_repeats must be an integer")?,
                )
            }
            "default_prefill_context" => {
                candidate.default_prefill_context = Some(
                    value
                        .parse()
                        .context("default_prefill_context must be 0 or 1")?,
                )
            }
            "extra_flags" => {
                candidate.extra_flags =
                    Some(serde_json::from_str(value).map_err(|_| {
                        anyhow::anyhow!("extra_flags must be a JSON array of strings")
                    })?)
            }
            "server_url" => {
                crate::community::endpoint(value, "")?;
                candidate.server_url = Some(value.to_string());
            }
            "api_key" => candidate.api_key = Some(value.to_string()),
            _ => bail!("Unknown configuration key"),
        }
        candidate.validate()?;
        *self = candidate;
        Ok(())
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
