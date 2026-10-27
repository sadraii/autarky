//! Native option boundaries, shared by launch policy and metadata consumers.
//!
//! The registry is pinned to llama.cpp build 9013. Unknown options cannot be
//! assigned an arity safely: execution/selection fail without echoing their
//! potentially private text. Values are never canonicalized or parsed as keys.
use anyhow::{bail, Result};
use std::{collections::BTreeMap, ops::Range, sync::OnceLock};

pub fn canonical_name(key: &str) -> String {
    if key.starts_with("--") {
        key.replace('_', "-")
    } else {
        key.to_owned()
    }
}

fn arities() -> &'static BTreeMap<&'static str, usize> {
    static TABLE: OnceLock<BTreeMap<&str, usize>> = OnceLock::new();
    TABLE.get_or_init(|| {
        include_str!("native_options.tsv")
            .lines()
            .filter(|line| !line.starts_with('#') && !line.is_empty())
            .map(|line| {
                let (key, value) = line.split_once('\t').expect("option registry");
                (key, value.parse().expect("option arity"))
            })
            .collect()
    })
}

pub struct Argument<'a> {
    pub key: String,
    pub values: Vec<&'a str>,
    pub tokens: &'a [String],
    pub range: Range<usize>,
    pub inline: bool,
    pub known: bool,
    pub complete: bool,
}

/// Tolerant boundary walk for old stored records. Call `parse` for execution.
/// Unknown tokens consume no guessed values and are marked explicitly.
pub fn walk(flags: &[String]) -> Vec<Argument<'_>> {
    walk_with_legacy_flash(flags, false)
}

/// Old stored records used boolean flash-attn. This alternate schema is used
/// only to find secrets; its tokens must never be used for native execution.
pub(crate) fn legacy_storage_walk(flags: &[String]) -> Vec<Argument<'_>> {
    walk_with_legacy_flash(flags, true)
}

fn walk_with_legacy_flash(flags: &[String], legacy: bool) -> Vec<Argument<'_>> {
    let mut result = Vec::new();
    let mut index = 0;
    while index < flags.len() {
        let start = index;
        let (key, inline) = flags[index]
            .split_once('=')
            .map_or((flags[index].as_str(), None), |(key, value)| {
                (key, Some(value))
            });
        let key = canonical_name(key);
        let arity = if legacy && matches!(key.as_str(), "--flash-attn" | "-fa") {
            Some(0)
        } else {
            arities().get(key.as_str()).copied()
        };
        let mut values = Vec::new();
        if let Some(value) = inline {
            values.push(value);
        }
        index += 1;
        if let Some(arity) = arity {
            for _ in values.len()..arity {
                if let Some(value) = flags.get(index) {
                    values.push(value);
                    index += 1;
                }
            }
        }
        result.push(Argument {
            key,
            complete: arity.is_some_and(|n| n == values.len()),
            known: arity.is_some(),
            values,
            tokens: &flags[start..index],
            range: start..index,
            inline: inline.is_some(),
        });
    }
    result
}

pub fn parse(flags: &[String]) -> Result<Vec<Argument<'_>>> {
    let args = walk(flags);
    if args.iter().any(|arg| !arg.known) {
        bail!("Cannot determine native option boundaries: unsupported option or positional token (registry: llama.cpp 9013)");
    }
    if args.iter().any(|arg| !arg.complete) {
        bail!("Native option has missing or unexpected values");
    }
    Ok(args)
}

pub fn last_value<'a>(args: &[Argument<'a>], aliases: &[&str]) -> Option<&'a str> {
    args.iter()
        .rev()
        .find(|arg| aliases.contains(&arg.key.as_str()))
        .and_then(|arg| arg.values.first().copied())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn flags(values: &[&str]) -> Vec<String> {
        values.iter().map(|s| s.to_string()).collect()
    }
    #[test]
    fn boundaries_follow_arity_not_value_prefixes() {
        let input = flags(&[
            "--alias",
            "--host=private",
            "--control-vector-layer-range",
            "-1",
            "--api-key",
            "--gpu_layers=1",
            "--gpu-layers",
            "2",
        ]);
        let args = parse(&input).unwrap();
        assert_eq!(args.len(), 4);
        assert_eq!(args[0].values, ["--host=private"]);
        assert_eq!(args[1].values, ["-1", "--api-key"]);
        assert_eq!(last_value(&args, &["--gpu-layers"]), Some("2"));
        assert_eq!(
            args.iter()
                .flat_map(|arg| arg.tokens.iter())
                .collect::<Vec<_>>(),
            input.iter().collect::<Vec<_>>()
        );
        assert!(parse(&flags(&["--alias"])).is_err());
        assert!(parse(&flags(&["--future-option", "--gpu-layers=0"])).is_err());
        assert!(parse(&flags(&["--mmap=value"])).is_err());
    }
}
