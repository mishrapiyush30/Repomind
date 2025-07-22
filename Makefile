# RepoMind Agent v0.1 Makefile

.PHONY: help install test lint format clean demo deploy

# Default target
help:
	@echo "RepoMind Agent v0.1 - Available commands:"
	@echo ""
	@echo "Development:"
	@echo "  install     Install dependencies and pre-commit hooks"
	@echo "  test        Run all tests with coverage"
	@echo "  lint        Run linting checks"
	@echo "  format      Format code with black and isort"
	@echo "  clean       Clean up generated files"
	@echo ""
	@echo "Demo:"
	@echo "  demo        Run the full demo workflow"
	@echo "  demo-quick  Run a quick demo with sample questions"
	@echo ""
	@echo "Deployment:"
	@echo "  build       Build Docker image"
	@echo "  deploy      Deploy to Fly.io"
	@echo ""
	@echo "Utilities:"
	@echo "  docs        Generate documentation"
	@echo "  release     Create a new release"

# Development
install:
	@echo "Installing dependencies..."
	pip install -e .
	pip install -e ".[dev]"
	pre-commit install
	@echo "✅ Installation complete"

test:
	@echo "Running tests..."
	pytest --cov=repo_mind_agent --cov-report=term-missing --cov-report=html
	@echo "✅ Tests complete"

lint:
	@echo "Running linting checks..."
	ruff check .
	black --check .
	isort --check-only .
	@echo "✅ Linting complete"

format:
	@echo "Formatting code..."
	black .
	isort .
	@echo "✅ Formatting complete"

clean:
	@echo "Cleaning up..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	rm -rf repo_mind_agent/__pycache__/
	rm -rf repo_mind_agent/*/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf *.db
	rm -rf repo_tree.json
	rm -rf evaluation_results.json
	@echo "✅ Cleanup complete"

# Demo
demo: clean
	@echo "🚀 Starting RepoMind Agent Demo..."
	@echo ""
	@echo "Step 1: Ingesting sample repository..."
	repomind ingest https://github.com/fastapi/fastapi --output demo_tree.json
	@echo ""
	@echo "Step 2: Loading repository data..."
	repomind load-data ./fastapi --repo-tree demo_tree.json --db-path demo_data.db
	@echo ""
	@echo "Step 3: Creating embeddings..."
	repomind create-embeddings ./fastapi --repo-tree demo_tree.json
	@echo ""
	@echo "Step 4: Running static analysis..."
	repomind analyze ./fastapi --output demo_analysis.txt
	@echo ""
	@echo "Step 5: Asking sample questions..."
	repomind ask "Where is the main function defined?" ./fastapi
	@echo ""
	repomind ask "What is the overall code quality?" ./fastapi
	@echo ""
	repomind ask "Who are the main contributors?" ./fastapi
	@echo ""
	@echo "✅ Demo complete! Check the outputs above."

demo-quick:
	@echo "🚀 Quick Demo - Asking sample questions..."
	@echo ""
	repomind ask "Where is the main function defined?" .
	@echo ""
	repomind ask "What are the main dependencies?" .
	@echo ""
	repomind ask "How is error handling implemented?" .
	@echo ""
	@echo "✅ Quick demo complete!"

# Deployment
build:
	@echo "Building Docker image..."
	docker build -t repo-mind-agent:latest .
	@echo "✅ Docker image built"

deploy:
	@echo "Deploying to Fly.io..."
	flyctl deploy --remote-only
	@echo "✅ Deployment complete"

# Documentation
docs:
	@echo "Generating documentation..."
	pdoc --html repo_mind_agent --output-dir docs/api
	@echo "✅ Documentation generated"

# Release
release:
	@echo "Creating release..."
	@read -p "Enter version (e.g., 0.1.0): " version; \
	git tag -a v$$version -m "Release v$$version"; \
	git push origin v$$version; \
	echo "✅ Release v$$version created"

# Development server
serve:
	@echo "Starting development server..."
	repomind serve --reload

# UI
ui:
	@echo "Starting Streamlit UI..."
	repomind ui . --port 8501

# Evaluation
eval:
	@echo "Running evaluation..."
	repomind evaluate . --output eval_results.json
	@echo "✅ Evaluation complete"

# Database setup
db-setup:
	@echo "Setting up databases..."
	@echo "Note: This requires PostgreSQL to be running"
	createdb repomind || echo "Database 'repomind' already exists"
	@echo "✅ Database setup complete"

# Full setup
setup: install db-setup
	@echo "✅ Full setup complete"

# CI/CD
ci: lint test
	@echo "✅ CI checks passed"

# Production
prod-build:
	@echo "Building production image..."
	docker build --target production -t repo-mind-agent:prod .
	@echo "✅ Production image built"

prod-run:
	@echo "Running production container..."
	docker run -p 8000:8000 -e OPENAI_API_KEY=$$OPENAI_API_KEY repo-mind-agent:prod

# Monitoring
logs:
	@echo "Viewing logs..."
	flyctl logs

status:
	@echo "Checking status..."
	flyctl status

# Helpers
check-env:
	@if [ -z "$$OPENAI_API_KEY" ]; then \
		echo "❌ OPENAI_API_KEY environment variable not set"; \
		echo "Please set it with: export OPENAI_API_KEY='your-key-here'"; \
		exit 1; \
	else \
		echo "✅ Environment variables configured"; \
	fi

check-deps:
	@echo "Checking dependencies..."
	@python -c "import openai, fastapi, streamlit, git, click" 2>/dev/null || \
		(echo "❌ Missing dependencies. Run 'make install' first." && exit 1)
	@echo "✅ Dependencies check passed"

# Default demo with environment check
demo-safe: check-env check-deps demo 