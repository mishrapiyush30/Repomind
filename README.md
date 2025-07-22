# RepoMind Agent v0.1

An intelligent agent that provides instant understanding of codebases through semantic search, SQL queries, and static analysis.

## 🎯 Purpose & Vision

Give any engineer an "instant map" of an unfamiliar codebase. RepoMind answers natural-language queries about a GitHub repo—citing exact lines, commit hashes, and complexity stats—so onboarding, code-review, security audit, and refactor planning are 10× faster.

## 🚀 Quick Start

### Local Installation

```bash
# Clone and setup
git clone <your-repo-url>
cd repo-mind-agent

# Install dependencies
pip install -e .

# Install pre-commit hooks
pre-commit install

# Ingest a repository
python scripts/ingest_repo.py https://github.com/username/repo-name

# Start the Streamlit UI
streamlit run repo_mind_agent/ui/trace_viewer.py

# Or use the CLI
repomind ask "Where is the main function defined?"
```

### Docker

```bash
# Build and run
docker build -t repo-mind-agent .
docker run -p 8000:8000 repo-mind-agent

# Access the API
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Where is the main function defined?"}'
```

## 🏗 Architecture

```
User → FastAPI (/ask)
            │
            ▼
      ReAct Orchestrator
   ┌─────┬────────┬────────┐
   ▼     ▼        ▼
VectorDB  SQL DB  Static‑Analyzer
(pgvector) (SQLite) (ruff/radon)
            │
            ▼
         Synthesiser
            │
            ▼
        Markdown + JSON trace
```

## 🛠 Key Features

- **Vector Code Search**: Semantic search across code and documentation
- **Commit/Issue SQL Query**: Historical analysis and metadata lookup
- **Static Analysis**: Complexity metrics and code quality insights
- **ReAct Agent**: Intelligent tool selection and reasoning
- **REST API**: FastAPI endpoint with authentication
- **Trace Viewer**: Streamlit UI for debugging agent behavior
- **CI/CD Pipeline**: Automated testing and deployment

## 📊 Success Metrics

| Objective            | KPI                                           | Target       |
| -------------------- | --------------------------------------------- | ------------ |
| Accurate answers     | Answer Groundedness (human rubric 1‑5)        | ≥ 4.2        |
| Useful retrieval     | Precision@5 on eval question set             | ≥ 0.65       |
| Developer time saved | Avg. "first‑answer latency" vs. manual search | −70 %        |
| Reliability          | p95 API latency (question → answer)           | ≤ 2 s        |
| Maintainability      | Unit‑test coverage                            | ≥ 85 % lines |

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=repo_mind_agent --cov-report=html

# Run evaluation harness
pytest tests/test_eval_harness.py
```

## 🚀 Deployment

### Fly.io

```bash
# Deploy to Fly.io
fly launch --name repo-mind-agent
fly deploy
```

### Local Development

```bash
# Start development server
uvicorn repo_mind_agent.orchestrator:app --reload --host 0.0.0.0 --port 8000
```

## 📝 API Usage

### Ask a Question

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "question": "Where is the main function defined?",
    "repo_path": "/path/to/repo"
  }'
```

### Response Format

```json
{
  "answer": "The main function is defined in `src/main.py` at line 42...",
  "trace": [
    {
      "step": "think",
      "content": "I need to search for the main function..."
    },
    {
      "step": "act",
      "tool": "vector_search",
      "input": {"query": "main function definition"},
      "output": {"results": [...]}
    }
  ],
  "citations": [
    {"file": "src/main.py", "line": 42, "content": "def main():"}
  ]
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run pre-commit hooks
6. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🎥 Demo

See `docs/demo.md` for a step-by-step demo script.

---

**RepoMind Agent v0.1** - Making codebases instantly understandable. 🚀 