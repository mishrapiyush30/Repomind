# RepoMind Agent v0.1 Demo Script

This document provides a step-by-step demo script for showcasing the RepoMind Agent functionality.

## Prerequisites

- Python 3.9+
- OpenAI API key
- PostgreSQL (optional, for vector search)
- Git

## Demo Setup

### 1. Installation

```bash
# Clone the repository
git clone <repo-url>
cd repo-mind-agent

# Install dependencies
pip install -e .

# Set up pre-commit hooks
pre-commit install
```

### 2. Environment Setup

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Optional: Set up PostgreSQL for vector search
export DATABASE_URL="postgresql://localhost/repomind"
```

## Demo Flow

### Step 1: Repository Ingestion

```bash
# Ingest a sample repository (using a popular open-source project)
repomind ingest https://github.com/fastapi/fastapi --output fastapi_tree.json

# Verify the repository tree was created
ls -la fastapi_tree.json
```

**Expected Output:**
```
📥 Ingesting repository: https://github.com/fastapi/fastapi
Cloning https://github.com/fastapi/fastapi to /tmp/...
Successfully cloned repository. Latest commit: abc12345
Scanning repository structure...
Repository structure saved to fastapi_tree.json
Found 150 files and 25 directories
✅ Repository ingested successfully: fastapi_tree.json
```

### Step 2: Data Loading

```bash
# Load repository data into SQLite
repomind load-data ./fastapi --repo-tree fastapi_tree.json --db-path fastapi_data.db
```

**Expected Output:**
```
📊 Loading repository data: ./fastapi
Loading repository tree...
Loaded 150 files from repo tree
Loading Git history...
Processed 1000 commits...
Loaded 1000 commits from Git history
Repository data loaded successfully!
✅ Repository data loaded successfully: fastapi_data.db
```

### Step 3: Vector Embeddings

```bash
# Create vector embeddings for semantic search
repomind create-embeddings ./fastapi --repo-tree fastapi_tree.json
```

**Expected Output:**
```
🧠 Creating embeddings: ./fastapi
Processed fastapi/main.py: 5 chunks
Processed fastapi/dependencies.py: 3 chunks
Processed fastapi/routing.py: 8 chunks
...
Total: 150 files, 450 chunks
✅ Vector embeddings created successfully
```

### Step 4: Static Analysis

```bash
# Run static analysis on the repository
repomind analyze ./fastapi --output analysis_results.txt
```

**Expected Output:**
```
🔍 Analyzing repository: ./fastapi
Static Analysis Results for ./fastapi

Overall Health: A (92/100)

Issues:
- 5 linting warnings
- 2 high complexity functions

Statistics:
- Python files: 45
- Lines of code: 12,450

Ruff Analysis:
- Total issues: 5
- Errors: 0
- Warnings: 5

Radon Analysis:
- Total functions: 234
- Average complexity: 3.2
- High complexity functions: 2

TODO Analysis:
- Total TODOs: 8
- Files with TODOs: 6
```

### Step 5: Interactive Questions

Now let's ask some intelligent questions about the codebase:

```bash
# Question 1: Find the main function
repomind ask "Where is the main function defined?" ./fastapi

# Question 2: Code quality assessment
repomind ask "What is the overall code quality of this repository?" ./fastapi

# Question 3: Contributor analysis
repomind ask "Who are the main contributors to this repository?" ./fastapi

# Question 4: Error handling patterns
repomind ask "How is error handling implemented?" ./fastapi

# Question 5: API endpoints
repomind ask "What are the main API endpoints?" ./fastapi
```

**Example Output for Question 1:**
```
🤖 RepoMind Agent analyzing: ./fastapi
Question: Where is the main function defined?
--------------------------------------------------
Question: Where is the main function defined?

Answer:
The main function is defined in `fastapi/main.py` at line 42. This is the entry point for the FastAPI application.

The function signature is:
```python
def main():
    """Main entry point for the FastAPI application."""
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

This function serves as the application's entry point and starts the uvicorn server.

Citations:
- fastapi/main.py:42 - def main():
- fastapi/main.py:43 - """Main entry point for the FastAPI application."""
- fastapi/main.py:44 - uvicorn.run(app, host="0.0.0.0", port=8000)

Steps taken: 3
```

### Step 6: Streamlit UI Demo

```bash
# Start the interactive Streamlit UI
repomind ui ./fastapi --port 8501
```

**Demo Points for UI:**
1. Show the sidebar configuration
2. Demonstrate sample questions
3. Show the ReAct trace visualization
4. Display citations and file references
5. Show the analysis charts

### Step 7: API Demo

```bash
# Start the FastAPI server
repomind serve --host 0.0.0.0 --port 8000
```

**API Testing:**
```bash
# Test the health endpoint
curl http://localhost:8000/health

# Test asking a question via API
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-token" \
  -d '{
    "question": "Where is the main function defined?",
    "repo_path": "./fastapi"
  }'
```

### Step 8: Evaluation

```bash
# Run the evaluation harness
repomind evaluate ./fastapi --output eval_results.json
```

**Expected Output:**
```
📊 Evaluating repository: ./fastapi
Starting RepoMind Agent evaluation...
Repository: ./fastapi
Questions: 15
--------------------------------------------------

Evaluating: Where is the main function defined?
  Response time: 2.34s
  Overall score: 4.2

Evaluating: What are the most complex functions in the codebase?
  Response time: 3.12s
  Overall score: 3.8

...

==================================================
EVALUATION SUMMARY
==================================================
Total Questions: 15
Successful: 15
Failed: 0
Success Rate: 100.00%

Average Scores:
  exact_match: 0.723
  groundedness: 4.1
  relevance: 4.3
  completeness: 3.9
  overall: 4.0

Performance Metrics:
  avg_response_time: 2.8
  time_compliance_rate: 1.0
  citation_compliance_rate: 0.93

Category Scores:
  code_location: 4.2 (5 questions)
  code_quality: 3.8 (4 questions)
  repository_metadata: 4.1 (3 questions)
  code_analysis: 4.0 (3 questions)

Difficulty Scores:
  easy: 4.3 (6 questions)
  medium: 3.9 (6 questions)
  hard: 3.7 (3 questions)

✅ Evaluation completed successfully
```

## Key Demo Points

### 1. **Intelligent Code Understanding**
- Show how the agent can find specific functions, classes, and patterns
- Demonstrate semantic search capabilities
- Highlight citation accuracy

### 2. **Multi-Tool Integration**
- Vector search for semantic similarity
- SQL queries for historical data
- Static analysis for code quality

### 3. **ReAct Reasoning**
- Show the step-by-step reasoning process
- Demonstrate tool selection logic
- Highlight observation and synthesis

### 4. **Grounded Responses**
- All answers include specific file and line references
- Citations link to actual code
- No hallucination of non-existent code

### 5. **Performance Metrics**
- Response times under 3 seconds
- High accuracy scores
- Comprehensive evaluation results

## Troubleshooting

### Common Issues:

1. **OpenAI API Key Not Set**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

2. **PostgreSQL Connection Issues**
   - Use SQLite for local testing
   - Check database URL format

3. **Memory Issues**
   - Reduce max_steps parameter
   - Use smaller repositories for demo

4. **Network Issues**
   - Check internet connection for API calls
   - Verify repository URLs are accessible

## Next Steps

After the demo, discuss:

1. **Production Deployment**
   - Docker containerization
   - Fly.io deployment
   - CI/CD pipeline

2. **Extensibility**
   - Adding new tools
   - Custom embeddings
   - Language support

3. **Performance Optimization**
   - Caching strategies
   - Parallel processing
   - Model optimization

4. **Future Features**
   - Multi-repo analysis
   - Real-time updates
   - Advanced visualizations

This demo showcases the full capabilities of RepoMind Agent v0.1 and demonstrates its value for codebase understanding and analysis. 