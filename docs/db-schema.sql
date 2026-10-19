-- 由 storage/platform.db 导出（git 不跟踪 .db，只跟踪 schema）
CREATE TABLE annotation_reviews (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id      INTEGER NOT NULL,
    episode_index   INTEGER NOT NULL,
    reviewer        TEXT NOT NULL,
    task_text       TEXT,
    success         INTEGER,
    note            TEXT,
    resolved        INTEGER DEFAULT 0,
    resolution_note TEXT,
    created_at      TEXT NOT NULL,
    UNIQUE(dataset_id, episode_index, reviewer)
)
CREATE TABLE annotations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id    INTEGER NOT NULL,
    episode_index INTEGER NOT NULL,
    task_text     TEXT,
    success       INTEGER,
    note          TEXT,
    annotator     TEXT,
    updated_at    TEXT NOT NULL,
    UNIQUE(dataset_id, episode_index)
)
CREATE TABLE annotators (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE,
    role       TEXT DEFAULT '标注员',
    active     INTEGER DEFAULT 1,
    note       TEXT,
    created_at TEXT NOT NULL
)
CREATE TABLE artifacts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id     INTEGER,
    dataset_id INTEGER,
    kind       TEXT,
    path       TEXT NOT NULL,
    bytes      INTEGER,
    created_at TEXT NOT NULL
)
CREATE TABLE audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    level       TEXT NOT NULL DEFAULT 'INFO',
    module      TEXT,
    message     TEXT,
    ref_type    TEXT,
    ref_id      INTEGER,
    detail_json TEXT
, actor TEXT)
CREATE TABLE bad_cases (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT NOT NULL,
    dataset_id      INTEGER,
    episode_index   INTEGER,
    evaluation_id   INTEGER,
    eval_episode_ix INTEGER,
    reason          TEXT,
    detail          TEXT,
    status          TEXT DEFAULT 'open',
    created_at      TEXT NOT NULL,
    handled_at      TEXT
)
CREATE TABLE collect_sessions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name     TEXT NOT NULL,
    dataset_id       INTEGER,
    robot_type       TEXT,
    fps              REAL,
    task_text        TEXT,
    state_dim        INTEGER,
    joint_names_json TEXT,
    image_keys_json  TEXT,
    status           TEXT DEFAULT 'open',
    frames           INTEGER DEFAULT 0,
    episode_index    INTEGER,
    path             TEXT,
    source           TEXT,
    note             TEXT,
    result_json      TEXT,
    error            TEXT,
    created_at       TEXT NOT NULL,
    committed_at     TEXT
)
CREATE TABLE dataset_splits (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id    INTEGER NOT NULL,
    episode_index INTEGER NOT NULL,
    split         TEXT NOT NULL,
    salt          TEXT,
    mode          TEXT,
    created_at    TEXT NOT NULL,
    UNIQUE(dataset_id, episode_index)
)
CREATE TABLE dataset_versions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id        INTEGER NOT NULL,
    version           INTEGER NOT NULL,
    kind              TEXT NOT NULL,
    content_hash      TEXT,
    episodes          INTEGER,
    frames            INTEGER,
    bytes             INTEGER,
    path              TEXT,
    zip_path          TEXT,
    params_json       TEXT,
    parent_dataset_id INTEGER,
    parent_version    INTEGER,
    note              TEXT,
    created_at        TEXT NOT NULL,
    UNIQUE(dataset_id, version)
)
CREATE TABLE datasets (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT NOT NULL UNIQUE,
    path             TEXT NOT NULL,
    format           TEXT NOT NULL DEFAULT 'lerobot',
    codebase_version TEXT,
    robot_type       TEXT,
    fps              REAL,
    total_episodes   INTEGER DEFAULT 0,
    total_frames     INTEGER DEFAULT 0,
    total_tasks      INTEGER DEFAULT 0,
    total_videos     INTEGER DEFAULT 0,
    chunks_size      INTEGER,
    codec            TEXT,
    image_shape      TEXT,
    features_json    TEXT,
    size_bytes       INTEGER DEFAULT 0,
    status           TEXT DEFAULT 'ready',
    created_at       TEXT NOT NULL,
    refreshed_at     TEXT
, version INTEGER DEFAULT 1, content_hash TEXT, parent_dataset_id INTEGER, derived_by TEXT, server_path TEXT)
CREATE TABLE device_scans (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT NOT NULL,
    cameras_json   TEXT,
    ports_json     TEXT,
    realsense_json TEXT,
    summary        TEXT,
    operator       TEXT
)
CREATE TABLE episode_hashes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id    INTEGER NOT NULL,
    episode_index INTEGER NOT NULL,
    exact_sha     TEXT,
    content_sha   TEXT,
    sample_sha    TEXT,
    near_sha      TEXT,
    rows          INTEGER,
    bytes         INTEGER,
    computed_at   TEXT,
    UNIQUE(dataset_id, episode_index)
)
CREATE TABLE episodes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id    INTEGER NOT NULL,
    episode_index INTEGER NOT NULL,
    length        INTEGER,
    task_index    INTEGER,
    task_text     TEXT,
    duration_s    REAL,
    data_rows     INTEGER,
    video_frames  INTEGER,
    action_dim    INTEGER,
    state_dim     INTEGER,
    success_count INTEGER,
    done_count    INTEGER,
    nan_values    INTEGER,
    data_bytes    INTEGER,
    video_bytes   INTEGER,
    parquet_path  TEXT,
    video_path    TEXT,
    probe_json    TEXT,
    created_at    TEXT NOT NULL,
    UNIQUE(dataset_id, episode_index)
)
CREATE TABLE eval_cases (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    evaluation_id INTEGER NOT NULL,
    episode_ix    INTEGER,
    seed          INTEGER,
    sum_reward    REAL,
    max_reward    REAL,
    success       INTEGER
)
CREATE TABLE evaluations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id  INTEGER,
    name           TEXT,
    eval_dir       TEXT,
    step           INTEGER,
    num_episodes   INTEGER,
    pc_success     REAL,
    avg_sum_reward REAL,
    avg_max_reward REAL,
    eval_s         REAL,
    eval_ep_s      REAL,
    videos_json    TEXT,
    created_at     TEXT NOT NULL, split TEXT,
    UNIQUE(eval_dir)
)
CREATE TABLE experiment_metrics (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    step          INTEGER,
    loss          REAL,
    grad_norm     REAL,
    lr            REAL,
    epoch         REAL,
    episodes      INTEGER,
    samples       REAL,
    updt_s        REAL
)
CREATE TABLE experiments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    run_dir         TEXT,
    output_dir      TEXT,
    policy          TEXT,
    dataset_repo_id TEXT,
    dataset_id      INTEGER,
    dataset_version INTEGER,
    dataset_hash    TEXT,
    steps           INTEGER,
    batch_size      INTEGER,
    learning_rate   REAL,
    seed            INTEGER,
    save_freq       INTEGER,
    eval_freq       INTEGER,
    config_json     TEXT,
    log_path        TEXT,
    metrics_count   INTEGER DEFAULT 0,
    status          TEXT DEFAULT 'registered',
    reproduce_cmd   TEXT,
    note            TEXT,
    created_at      TEXT NOT NULL, split TEXT, split_salt TEXT,
    UNIQUE(run_dir, steps)
)
CREATE TABLE jobs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    kind          TEXT NOT NULL,
    target_type   TEXT,
    target_id     INTEGER,
    payload_json  TEXT,
    status        TEXT NOT NULL DEFAULT 'queued',
    progress      INTEGER DEFAULT 0,
    total         INTEGER DEFAULT 0,
    message       TEXT,
    result_json   TEXT,
    error         TEXT,
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT
)
CREATE TABLE models (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    backend       TEXT,
    kind          TEXT,
    dataset_id    INTEGER,
    split         TEXT,
    val_split     TEXT,
    split_salt    TEXT,
    uses_vision   INTEGER DEFAULT 0,
    path          TEXT,
    meta_path     TEXT,
    bytes         INTEGER,
    metrics_json  TEXT,
    job_id        INTEGER,
    created_at    TEXT NOT NULL
)
CREATE TABLE qc_findings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL,
    dataset_id    INTEGER NOT NULL,
    episode_index INTEGER,
    rule          TEXT NOT NULL,
    severity      TEXT NOT NULL,
    detail        TEXT,
    measured      TEXT,
    expected      TEXT
, category TEXT, check_version TEXT, source TEXT)
CREATE TABLE qc_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id  INTEGER NOT NULL,
    job_id      INTEGER,
    report_path TEXT,
    total       INTEGER,
    passed      INTEGER,
    failed      INTEGER,
    result      TEXT,
    summary_json TEXT,
    created_at  TEXT NOT NULL
)
CREATE TABLE review_batches (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id    INTEGER NOT NULL,
    salt          TEXT,
    ratio         REAL,
    requested     INTEGER,
    episodes_json TEXT,
    note          TEXT,
    created_at    TEXT NOT NULL
)
CREATE TABLE spatial_annotations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id    INTEGER NOT NULL,
    episode_index INTEGER NOT NULL,
    kind          TEXT NOT NULL,
    label         TEXT NOT NULL,
    data_json     TEXT NOT NULL,
    frame_index   INTEGER,
    annotator     TEXT,
    created_at    TEXT NOT NULL
)
CREATE TABLE sqlite_sequence(name,seq)
CREATE TABLE storage_policy (
    id           INTEGER PRIMARY KEY CHECK (id = 1),
    enabled      INTEGER DEFAULT 0,
    threshold_mb REAL DEFAULT 200,
    kinds_json   TEXT,
    interval_s   INTEGER DEFAULT 300,
    last_check   TEXT,
    last_action  TEXT,
    last_result  TEXT
)
CREATE TABLE task_aliases (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    alias      TEXT NOT NULL UNIQUE,
    canonical  TEXT NOT NULL,
    note       TEXT,
    created_at TEXT NOT NULL
)
CREATE TABLE vlm_checks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id    INTEGER NOT NULL,
    episode_index INTEGER NOT NULL,
    model         TEXT,
    endpoint      TEXT,
    mode          TEXT,
    prompt_hash   TEXT,
    content_sha   TEXT,
    verdict       TEXT,
    compliant     INTEGER,
    findings_json TEXT,
    latency_ms    INTEGER,
    images        INTEGER,
    created_at    TEXT NOT NULL,
    UNIQUE(dataset_id, episode_index, model, prompt_hash, content_sha)
)
CREATE TABLE work_sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    annotator  TEXT NOT NULL,
    dataset_id INTEGER,
    started_at TEXT NOT NULL,
    ended_at   TEXT,
    note       TEXT,
    created_at TEXT NOT NULL
)
CREATE INDEX idx_audit ON audit_log(level, module)
CREATE INDEX idx_badcases_status ON bad_cases(status)
CREATE INDEX idx_ehash_content ON episode_hashes(content_sha)
CREATE INDEX idx_episodes_dataset ON episodes(dataset_id)
CREATE INDEX idx_findings_run ON qc_findings(run_id)
CREATE INDEX idx_jobs_status ON jobs(status)
CREATE INDEX idx_metrics_exp ON experiment_metrics(experiment_id)
CREATE INDEX idx_reviews_dataset ON annotation_reviews(dataset_id)
CREATE INDEX idx_sessions_status ON collect_sessions(status)
CREATE INDEX idx_spatial ON spatial_annotations(dataset_id, episode_index)
CREATE INDEX idx_versions_dataset ON dataset_versions(dataset_id)
CREATE INDEX idx_vlm_checks_lookup ON vlm_checks(dataset_id, episode_index, content_sha)
CREATE INDEX idx_work_open ON work_sessions(annotator, ended_at)
