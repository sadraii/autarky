-- Initial server schema from 5856f78:server/src/db.rs (init_db).
CREATE TABLE IF NOT EXISTS api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL,
    label TEXT DEFAULT '',
    created_at TEXT DEFAULT (datetime('now')),
    last_used_at TEXT
);

CREATE TABLE IF NOT EXISTS hardware_fingerprints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    gpu_family TEXT NOT NULL,
    vram_gb REAL NOT NULL,
    system_ram_gb INTEGER,
    os_platform TEXT
);

CREATE TABLE IF NOT EXISTS model_fingerprints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    quantization TEXT,
    architecture_type TEXT,
    expert_count INTEGER DEFAULT 0,
    parameters_billions REAL
);

CREATE TABLE IF NOT EXISTS benchmark_submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    api_key_id INTEGER REFERENCES api_keys(id),
    hardware_id INTEGER REFERENCES hardware_fingerprints(id),
    model_id INTEGER REFERENCES model_fingerprints(id),
    flags_used TEXT,
    context_lengths_tested TEXT,
    raw_metrics TEXT,
    oom_boundary INTEGER,
    oom INTEGER DEFAULT 0,
    submitted_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_submissions_hardware ON benchmark_submissions(hardware_id);
CREATE INDEX IF NOT EXISTS idx_submissions_model ON benchmark_submissions(model_id);
CREATE INDEX IF NOT EXISTS idx_submissions_submitted ON benchmark_submissions(submitted_at);
