# Autarky

Benchmark local GGUF models with `llama-server`. Store per-context measurements in SQLite and compare runs.

## Privacy and network use

**A fresh installation is local-only.** No community endpoint is set. After you configure a real `server_url`, a successful benchmark automatically attempts to share its result and query comparable community results. Sharing is best-effort: a community service failure does not remove the saved local run. There is no separate submit command.

**Use `--offline` for runs that must not contact the community service**, including runs on an installation that already has a server configured:

```bash
autarky --offline bench --context-lengths 1,2 --repeats 3 --prefill-context 0 \
  --passthrough /path/to/llama-server --model /path/to/model.gguf --ctx-size 4096
```

Autarky still uses loopback HTTP to control its local benchmark child. Supply a local model and do not request model downloads in the forwarded server command for an offline benchmark.

Community submissions include GPU family, VRAM, system RAM, OS, model metadata when known, sanitized server flags, tested contexts, per-context performance/status data, OOM boundary, and `llama-server` version. The service uses an API key to identify submissions. Model weights, prompt text, and stable GPU device identifiers are not upload fields. Known local model-path flags retain only filenames. A retained filename or model display name can still reveal private information.

Known secret flags are removed; known path flags are reduced to filenames, and URL credentials, query strings, and fragments are removed. Autarky applies filtering before storage and again before submission, including submission of older saved flags. **This filter is not a general secret detector. Unknown text-valued flags can pass through. It does not make sharing NDA-safe.** Review forwarded flags and model names. Use `--offline` if disclosure is not acceptable.

## Install from source

Install Rust with Cargo, Git, and a C compiler for bundled SQLite and native dependencies. The project uses Rust 2021; CI selects Rust 1.99.1.

```bash
git clone https://github.com/sadraii/autarky.git
cd autarky
cargo install --path . --locked

# On a Unix shell, add this to your shell startup file if needed:
export PATH="$HOME/.cargo/bin:$PATH"
autarky --version
```

With a custom `CARGO_HOME`, add `$CARGO_HOME/bin` instead. On Windows, add `%USERPROFILE%\.cargo\bin` to your user `PATH`, or the `bin` directory under your custom `CARGO_HOME`.

Install `llama-server` separately and obtain a GGUF model you are permitted to use. Autarky does not bundle either in its source package or release archives. Use the complete server command after `--passthrough`; the first item is the executable. There is no server-path setting or discovery step.

### Release archives

The release workflow prepares **draft** GitHub releases. It does not guarantee that a published release is available. Check the releases in `sadraii/autarky` before using this alternative. Planned archive names contain a full Rust target triple:

| Platform | Target |
| --- | --- |
| Linux x86-64, GNU libc | `x86_64-unknown-linux-gnu` |
| macOS Apple Silicon | `aarch64-apple-darwin` |
| macOS Intel | `x86_64-apple-darwin` |

For example, with GitHub CLI and Python 3.9 or newer installed, download and verify a **published Linux x86-64** archive in a new directory:

```bash
(
set -eu
target=x86_64-unknown-linux-gnu
gh release download --repo sadraii/autarky \
  --pattern "autarky-${target}.tar.gz" \
  --pattern "autarky-${target}.tar.gz.sha256"
python3 - "autarky-${target}.tar.gz" <<'PY'
import hashlib, pathlib, sys
filename = sys.argv[1]
checksum = pathlib.Path(filename + '.sha256')
digest, checked_name = checksum.read_text().strip().split('  ', 1)
if checked_name != filename:
    raise SystemExit('Unexpected archive name')
if hashlib.sha256(pathlib.Path(filename).read_bytes()).hexdigest() != digest:
    raise SystemExit('Checksum mismatch')
print('Checksum verified')
PY
tar -xzf "autarky-${target}.tar.gz"
mkdir -p "$HOME/.local/bin"
install -m 755 autarky "$HOME/.local/bin/autarky"
export PATH="$HOME/.local/bin:$PATH"
autarky --version
)
```

The subshell stops before extraction or installation if download or verification fails. Add `$HOME/.local/bin` to your shell startup `PATH` to use the installed command in later sessions. Checksums detect corruption; they are not independent signatures. Native macOS/Windows operation has not been verified in the local repair session. No Windows release artifact is defined.

## Benchmark controls

Put Autarky options **before** `--passthrough`; all items after it belong to the server command.

Native option boundaries use the argument registry from llama.cpp build 9013. Autarky preserves the forwarded tokens, but rejects unknown option arities instead of guessing whether a token is an option or a value. A newer server option requires a registry update before it can be used.

```bash
autarky --offline bench --context-lengths 1,2,4 --repeats 3 --prefill-context 1 \
  --passthrough ./llama-server --model ./model.gguf --ctx-size 8192 --threads 8
```

- `--context-lengths` takes **K indices**, not raw token counts: `1,2,4,8,16,32,64,128,256`. For example, use `1,2,4`, not `1024,2048,4096`.
- These indices select the nine embedded prompt assets. They are not exact tokenizer-specific prompt token counts. Model context capacity and output headroom can restrict selection.
- `--repeats` sets the number of measured requests per context. It must be positive.
- `--prefill-context` accepts `0` or `1`. A value of `1` runs one unmeasured warmup. Measured repeats do not reuse the KV cache, so reported prompt rates measure cold-cache prefill.
- Explicit benchmark options override saved defaults. If no context list is set, Autarky selects supported prompt sizes within the model context when known.
- `--tui` enables the dashboard. Use `autarky bench --help` for current options and bounds.

Autarky starts its child on loopback, waits for readiness, performs the measurements, and stops the child. Prompt processing and generation use separate server timing data, not blended HTTP wall-clock throughput. A missing or invalid timing sample is not a successful measurement. Raw per-context results retain failure status; summaries use successful points.

## Commands

```bash
autarky list
autarky list --model tinyllama --date-range 2026-10-01,2026-10-25
autarky compare --id1 1 --id2 2
autarky export --format json --file results.json
autarky export --format csv --file results.csv
autarky leaderboard --period all --sort tps --top 10
autarky recommend --min-tps 10 --limit 10
autarky config
autarky config --key default_repeats --value 3
autarky config --key default_context_lengths --value 1,2,4
autarky config --key default_prefill_context --value 0
autarky config --key extra_flags --value '["--threads","8"]'
autarky config --key server_url
```

Configuration keys are `default_context_lengths`, `default_repeats`, `default_prefill_context`, `extra_flags`, `server_url`, and `api_key`. `extra_flags` is a JSON string array. Normal configuration output masks the API key. Configuration and the local database are initialized automatically. `init`, `setup`, `submit`, and `--server-path` are not supported.

To enable community features, set `server_url` to the actual HTTPS base URL of a service you trust, using `autarky config --key server_url --value "$AUTARKY_COMMUNITY_URL"` after setting that shell variable. Do not use a fictional hostname. Plain HTTP is permitted only for a loopback development server. Community search and remote comparison require a configured endpoint and are unavailable with `--offline`:

```bash
autarky search tinyllama
autarky compare --remote --id1 1 --id2 2
```

Local IDs and community submission IDs are different. Comparison is meaningful only for compatible model/quantization, hardware, and successful contexts. Community rankings are observations, not performance guarantees.

`recommend` lists saved flags across models that meet the GPU and TPS filters. It is not a matched-model speed comparison. Benchmark a proposed configuration with your own model and context sizes before drawing a conclusion.

**Deletion changes local history.** `autarky delete last` deletes the most recent local entry; `autarky delete all` deletes entries for the detected hardware identity. Neither command deletes community submissions. Back up the database before deletion. Use subcommand help for selectors and filters.

On Linux, the usual locations are `~/.config/autarky/config.toml` and `~/.local/share/autarky/benchmarks.db`. Platform directory APIs and XDG settings determine the actual paths. Configuration output shows its path; keep configuration and database backups private.

## Community server

The server is a **separate crate**, with its own `server/Cargo.lock` and build output:

```bash
cargo build --locked --release --manifest-path server/Cargo.toml
AUTARKY_SERVER_DB=./submissions.db BIND_ADDR=127.0.0.1 PORT=8080 \
  ./server/target/release/autarky-server
```

This starts a local development service, not a production deployment. The server speaks plain HTTP and binds to loopback by default. For remote clients, place it behind an HTTPS/TLS terminator; do not expose the HTTP listener publicly. The server uses direct socket peer addresses and does not trust forwarding headers. Clients behind one reverse proxy share its peer bucket: enforce per-client limits at the trusted proxy and size the backend bucket deliberately.

The OVH community service uses `https://40.160.103.121`. To enable sharing with this service, run `autarky config --key server_url --value https://40.160.103.121`. Fresh installations remain local-only until configured. The web root opens `/health`; there is no web dashboard. Deployment settings, certificate renewal, and recovery steps are in `deploy/ovh/README.txt`.

Server settings are read at startup. Invalid numeric values fail startup rather than disable limits:

| Environment variable | Default | Meaning / accepted range |
| --- | --- | --- |
| `AUTARKY_SERVER_DB` | Platform data directory, `autarky-server/submissions.db` | SQLite file |
| `BIND_ADDR` | `127.0.0.1` | Listener IP address, not a hostname |
| `PORT` | `8080` | `1..=65535` |
| `AUTARKY_RATE_MAX` | `10` | Requests per direct peer per window, `1..=100000` |
| `AUTARKY_RATE_WINDOW` | `60` | Window in seconds, `1..=86400` |
| `AUTARKY_GLOBAL_RATE_MAX` | `100` | Global requests per window, `1..=1000000` |
| `AUTARKY_MAX_SUBMISSIONS` | `100000` | Stored submission quota, `1..=10000000` |
| `AUTARKY_DB_MAX_MB` | `256` | SQLite size cap in MiB, `1..=16384`; allow extra disk space for journals and backups |

Pagination accepts `limit` from 1 to 50 and `offset` from 0 to 10000. JSON request bodies are limited to 64 KiB; serialized responses are limited to 2 MiB. These are application bounds, not a complete host resource budget.

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness, not a database readiness guarantee |
| GET | `/ready` | Bounded database readiness check |
| POST | `/api/v1/benchmark` | Submit with `X-API-Key` |
| GET | `/api/v1/benchmarks` | Bounded query and pagination |
| GET | `/api/v1/models/{name}` | Exact model-name query |
| GET | `/api/v1/hardware/{id}/benchmarks` | Hardware fingerprint query |
| POST | `/api/v1/compare` | Compare two submission IDs |

Public reads do not register API keys. Anonymous submission keys are not verified personal identities. Admission controls, storage quotas, typed validation, and a bounded database queue reduce abuse; they do not replace deployment monitoring, backups, or transport security.

A complete submission can be accepted even if its client later disconnects. A missing response does not prove that the transaction was cancelled. Confirm the stored result before retrying; the API does not promise exactly-once delivery across network failures.

## Development checks

Run checks separately for both crates; the root is not a workspace containing the server:

```bash
cargo test --locked --all-targets
cargo test --locked --doc
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --manifest-path server/Cargo.toml --all-targets
cargo fmt --manifest-path server/Cargo.toml --all -- --check
cargo clippy --locked --manifest-path server/Cargo.toml --all-targets -- -D warnings
cargo install cargo-audit --version 0.22.2 --locked
cargo audit --deny warnings --file Cargo.lock
cargo audit --deny warnings --file server/Cargo.lock
python3 scripts/test-readme-install.py
python3 scripts/check-package.py
```

The server has no library target or doctests. Both crates run all unit and integration test targets; the CLI also runs doctests. Dependency checks reject advisory warnings as well as vulnerabilities.

One Linux native-authentication test is ignored by the default test command because it requires a separately installed server and model. To execute it as part of the full suite, select local assets with absolute paths:

```bash
AUTARKY_NATIVE_SERVER=/absolute/path/to/llama-server \
AUTARKY_NATIVE_MODEL=/absolute/path/to/model.gguf \
  cargo test --locked --all-targets -- --include-ignored
```

That test checks quoted CSV keys, environment keys, appended keys, and raw key-file lines against the selected native server. CI does not supply these external assets; its default test run does not certify native authentication.

The package check needs Python 3.11 or newer and cached Cargo dependencies. It runs Cargo in offline mode, checks the file list before archiving, rejects model weights, checks all CLI source/tests and all nine prompt files, compares archived content, and asks Cargo to build the extracted package. It does not publish.

CI runs on pushes and pull requests. Tag releases depend on those checks, install explicit targets, test and build both binaries, and produce per-archive portable SHA-256 files. GitHub action revisions are pinned. The workflow creates a draft release only; it does not publish to crates.io or deploy a service. Workflow definitions are not evidence of a successful hosted run. See `PROGRESS.md` for verification limits and `PHASE1_MVP.md` for historical context.

## License

MIT.