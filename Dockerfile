# Multi-stage Dockerfile for RepoMind Agent
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml .
COPY README.md .

# Install dependencies
RUN pip install --no-cache-dir -e .

# Production stage
FROM python:3.9-slim as production

# Set working directory
WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    git \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY repo_mind_agent/ ./repo_mind_agent/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY golden_qas.yaml .

# Create non-root user
RUN useradd --create-home --shell /bin/bash repomind && \
    chown -R repomind:repomind /app

# Switch to non-root user
USER repomind

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "repo_mind_agent.orchestrator:app", "--host", "0.0.0.0", "--port", "8000"] 