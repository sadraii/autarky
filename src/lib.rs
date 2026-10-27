//! Autarky — Local LLM benchmark launcher for llama-server
//!
//! This crate provides a CLI for benchmarking local LLMs via llama-server.

pub mod arguments;
pub mod community;
pub mod config;
pub mod db;
pub mod gguf;
pub mod hardware;
pub mod launcher;
pub mod monitor;
mod process;
pub mod report;
pub mod tui;
pub mod utils;

#[cfg(all(test, unix))]
#[path = "../tests/support/process.rs"]
mod test_process;
