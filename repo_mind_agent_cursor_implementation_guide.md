## RepoMind Agent v0.1: Cursor AI End‑to‑End Implementation Guide

Follow these steps in Cursor AI to deliver the full 10‑day roadmap for RepoMind Agent v0.1.

---

### 📁 Workspace & Repo Setup

1. **New Cursor Project**
   - Name: `repo-mind-agent`
   - Python 3.9 environment, Git initialized.
2. **Add config files** in root:
   - `.pre-commit-config.yaml` (Black, whitespace, YAML checks)
   - `pyproject.toml` (metadata, deps)
   - `README.md` (purpose, install, usage)
3. **Install dependencies**:
   ```bash
   pip install gitpython click fastapi uvicorn sentence-transformers pgvector ruff radon pytest
   pre-commit install
   ```

---

### 🗂 Directory Structure

```
repo-mind-agent/
├── .pre-commit-config.yaml
├── pyproject.toml
├── README.md
├── scripts/
│   └── ingest_repo.py
├── repo_mind_agent/
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── tools/
│   │   ├── vector_search.py
│   │   ├── sql_query.py
│   │   └── static_analysis.py
│   └── ui/
│       └── trace_viewer.py
├── tests/
│   ├── test_ingest_repo.py
│   ├── test_sql_query.py
│   ├── test_vector_search.py
│   └── test_eval_harness.py
└── golden_qas.yaml
```

---

### 🛠 Day 1: Ingest & Tests

- **Script**: `scripts/ingest_repo.py` (clone a GitHub URL → `repo_tree.json`).
- **Tests**: `tests/test_ingest_repo.py` asserts JSON entries have `path`, `is_dir`, `size`.

```bash
pytest tests/test_ingest_repo.py
```

---

### 🛠 Day 2: SQLite ETL Loader

- **File**: `repo_mind_agent/tools/sql_query.py`
- **Tasks**:
  1. Define SQLite schema (`commits`, `files`, `issues`, `labels`).
  2. ETL: read `repo_tree.json` + Git history via `gitpython` → populate tables.
- **Test**: `tests/test_sql_query.py` verifies row counts and sample queries.

---

### 🛠 Day 3: Vector Search Tool

- **File**: `repo_mind_agent/tools/vector_search.py`
- **Implement**:
  1. Chunk code/docs (≤ 400 tokens) with simple newline splitter.
  2. Embed via `sentence-transformers` (`bge-large-en`).
  3. Upsert into `pgvector` table with `file_path`, `start_line`, `end_line`.
- **Test**: `tests/test_vector_search.py` checks embedding dim and top‑k retrieval.

---

### 🛠 Day 4: SQL Query Tool

- **File**: same `sql_query.py`, add `run_query(sql, params)` function.
- **Usage**: parameterised queries for commit history, file changes, issue lookups.
- **Examples** in docstring and `tests/test_sql_query.py`.

---

### 🛠 Day 5: ReAct Agent MVP

- **File**: `repo_mind_agent/orchestrator.py`
- **Structure**:
  1. **Think**: parse user question.
  2. **Act**: call vector search or SQL or static analysis tool.
  3. **Observe**: collect tool output.
  4. **Think** again, loop ≤ 3 steps.
  5. **Synthesize**: Markdown answer with inline citations (`[file.py:42]`).
- **Test**: manual REPL in Cursor; `python -c "from orchestrator import ask; print(ask('Where is main()?'))"`

---

### 🛠 Day 6: Static Analysis Integration

- **File**: `repo_mind_agent/tools/static_analysis.py`
- **Implement**:
  1. Wrap `ruff --select C,F` for lint/TODO counts.
  2. Wrap `radon cc` for cyclomatic complexity.
  3. Return JSON metrics per file.
- **Integration**: orchestrator calls this tool when complexity stats requested.
- **Test**: add `tests/test_static_analysis.py`.

---

### 🛠 Day 7: Streamlit Trace Viewer UI

- **File**: `repo_mind_agent/ui/trace_viewer.py`
- **Use**: Streamlit to display:
  - **Sidebar**: question input + submit button.
  - **Main**: agent steps (think/act/observe), raw I/O, final answer.
- **Run**:
  ```bash
  streamlit run repo_mind_agent/ui/trace_viewer.py
  ```

---

### 🛠 Day 8: Eval Harness

- **golden\_qas.yaml**: list of `{question, expected_answer}` samples.
- **File**: `tests/test_eval_harness.py`:
  - Load YAML, call `ask()` for each.
  - Compute Exact Match (EM) & groundedness rubric placeholder.
- **Schedule**: add nightly GitHub Action to run this test suite.

---

### 🛠 Day 9: Docker & Deployment

1. **Dockerfile** (multi‑stage):
   - Base: Python 3.9, install deps.
   - Copy src & expose port 8000.
   - Entrypoint: `uvicorn repo_mind_agent.orchestrator:app --host 0.0.0.0 --port 8000`
2. **Fly.io**:
   - `fly launch --name repo-mind-agent` → generates `fly.toml`.
   - `fly deploy` → staging & production.
3. **CI/CD**: Extend `.github/workflows/ci.yml` with `deploy` job on `main`:
   ```yaml
   jobs:
     deploy:
       runs-on: ubuntu-latest
       needs: test
       steps:
         - uses: actions/checkout@v3
         - uses: superfly/flyctl-actions@1.1
           with:
             args: "deploy --config fly.toml --remote-only"
   ```

---

### 🛠 Day 10: Polish & Release

- **README.md**: update with:
  - Local usage, Docker, Streamlit, REST API examples.
  - VS Code extension instructions (if any).
- **Loom Demo**: script in `docs/demo.md`:
  1. Clone & ingest script.
  2. Ask a sample question.
  3. Show Streamlit viewer.
  4. Deploy URL.
- **Tag v0.1**:
  ```bash
  git tag -a v0.1.0 -m "RepoMind Agent v0.1"
  git push origin v0.1.0
  ```

---

## CI/CD Workflow Snippet (`.github/workflows/ci.yml`)

```yaml
name: CI & Deploy
on:
  push:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * *'   # nightly eval
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: {python-version: 3.9}
      - run: pip install -r requirements.txt
      - run: pre-commit run --all-files
      - run: pytest
  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - uses: superfly/flyctl-actions@1.1
        with: {args: 'deploy --config fly.toml --remote-only'}
```

---

### 💡 Cursor Best Practices

- **Split panes**: code, terminal, REPL, editor.
- **Notebook cells**: prototype complex SQL or embedding logic inline.
- **Git lens**: commit early and review diffs visually.
- **Command palette**: jump to any file or run scripts rapidly.
- **Interactive REPL**: test tools directly without full orchestration.

By following this end‑to‑end guide, you’ll hit every milestone from cloning to production deploy in Cursor AI—delivering a polished RepoMind Agent v0.1 that aligns perfectly with the PRD’s vision and Conexio AI’s target skill set. 🚀

