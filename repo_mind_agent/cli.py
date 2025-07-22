"""
Command-line interface for RepoMind Agent.
"""

import click
import json
import os
from pathlib import Path
from typing import Optional

from .orchestrator import ask, RepoMindAgent


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """RepoMind Agent - Intelligent codebase understanding."""
    pass


@cli.command()
@click.argument("question")
@click.argument("repo_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file for results")
@click.option("--format", "output_format", type=click.Choice(["json", "markdown", "text"]), 
              default="text", help="Output format")
@click.option("--max-steps", default=5, help="Maximum ReAct steps")
@click.option("--db-path", default="repo_data.db", help="SQLite database path")
@click.option("--vector-db-url", help="PostgreSQL vector database URL")
@click.option("--openai-key", envvar="OPENAI_API_KEY", help="OpenAI API key")
def ask_command(question, repo_path, output, output_format, max_steps, db_path, vector_db_url, openai_key):
    """Ask a question about a repository."""
    
    if not openai_key:
        click.echo("Error: OpenAI API key is required. Set OPENAI_API_KEY environment variable or use --openai-key.")
        return 1
    
    try:
        click.echo(f"🤖 RepoMind Agent analyzing: {repo_path}")
        click.echo(f"Question: {question}")
        click.echo("-" * 50)
        
        # Ask the question
        result = ask(
            question=question,
            repo_path=repo_path,
            db_path=db_path,
            vector_db_url=vector_db_url,
            openai_api_key=openai_key
        )
        
        # Format output
        if output_format == "json":
            output_text = json.dumps(result, indent=2)
        elif output_format == "markdown":
            output_text = f"""# RepoMind Agent Response

## Question
{result['question']}

## Answer
{result['answer']}

## Citations
"""
            for citation in result.get('citations', []):
                if 'file' in citation:
                    output_text += f"- `{citation['file']}:{citation['line']}` - {citation['content']}\n"
                elif 'hash' in citation:
                    output_text += f"- Commit `{citation['hash']}` by {citation['author']} - {citation['message']}\n"
            
            output_text += f"\n## Steps Taken\n{result['steps_taken']} ReAct steps"
        else:
            output_text = f"""Question: {result['question']}

Answer:
{result['answer']}

Citations:"""
            for citation in result.get('citations', []):
                if 'file' in citation:
                    output_text += f"\n- {citation['file']}:{citation['line']} - {citation['content']}"
                elif 'hash' in citation:
                    output_text += f"\n- Commit {citation['hash']} by {citation['author']} - {citation['message']}"
            
            output_text += f"\n\nSteps taken: {result['steps_taken']}"
        
        # Output results
        if output:
            with open(output, 'w') as f:
                f.write(output_text)
            click.echo(f"Results saved to: {output}")
        else:
            click.echo(output_text)
        
        return 0
        
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        return 1


@cli.command()
@click.argument("repo_url")
@click.option("--output", "-o", default="repo_tree.json", help="Output JSON file")
@click.option("--temp-dir", "-t", help="Temporary directory for cloning")
def ingest(repo_url, output, temp_dir):
    """Ingest a repository from GitHub URL."""
    try:
        from scripts.ingest_repo import main as ingest_main
        
        click.echo(f"📥 Ingesting repository: {repo_url}")
        
        # Call the ingest script
        ingest_main.callback(repo_url, output, temp_dir)
        
        click.echo(f"✅ Repository ingested successfully: {output}")
        return 0
        
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        return 1


@cli.command()
@click.argument("repo_path", type=click.Path(exists=True))
@click.option("--repo-tree", default="repo_tree.json", help="Repository tree JSON file")
@click.option("--db-path", default="repo_data.db", help="SQLite database path")
def load_data(repo_path, repo_tree, db_path):
    """Load repository data into SQLite database."""
    try:
        from .tools.sql_query import load_repository_data
        
        click.echo(f"📊 Loading repository data: {repo_path}")
        
        if not Path(repo_tree).exists():
            click.echo(f"Error: Repository tree file not found: {repo_tree}")
            click.echo("Run 'repomind ingest' first to create the repository tree.")
            return 1
        
        # Load data
        loader = load_repository_data(repo_path, repo_tree, db_path)
        
        click.echo(f"✅ Repository data loaded successfully: {db_path}")
        return 0
        
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        return 1


@cli.command()
@click.argument("repo_path", type=click.Path(exists=True))
@click.option("--repo-tree", default="repo_tree.json", help="Repository tree JSON file")
@click.option("--vector-db-url", help="PostgreSQL vector database URL")
def create_embeddings(repo_path, repo_tree, vector_db_url):
    """Create vector embeddings for repository content."""
    try:
        from .tools.vector_search import create_embeddings_from_repo_tree
        
        click.echo(f"🧠 Creating embeddings: {repo_path}")
        
        if not Path(repo_tree).exists():
            click.echo(f"Error: Repository tree file not found: {repo_tree}")
            click.echo("Run 'repomind ingest' first to create the repository tree.")
            return 1
        
        # Create embeddings
        tool = create_embeddings_from_repo_tree(repo_tree, repo_path, vector_db_url)
        
        click.echo("✅ Vector embeddings created successfully")
        return 0
        
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        return 1


@cli.command()
@click.argument("repo_path", type=click.Path(exists=True))
@click.option("--output", "-o", help="Output file for analysis results")
def analyze(repo_path, output):
    """Run static analysis on repository."""
    try:
        from .tools.static_analysis import analyze_repository_static
        
        click.echo(f"🔍 Analyzing repository: {repo_path}")
        
        # Run analysis
        results = analyze_repository_static(repo_path)
        
        # Format output
        output_text = f"""Static Analysis Results for {results['repository']}

Overall Health: {results['overall_health']['grade']} ({results['overall_health']['score']}/100)

Issues:"""
        for issue in results['overall_health']['issues']:
            output_text += f"\n- {issue}"
        
        output_text += f"""

Statistics:
- Python files: {results['statistics']['total_python_files']}
- Lines of code: {results['statistics']['total_lines_of_code']}

Ruff Analysis:"""
        if results['ruff']['success']:
            ruff_summary = results['ruff']['summary']
            output_text += f"""
- Total issues: {ruff_summary['total_issues']}
- Errors: {ruff_summary['error_count']}
- Warnings: {ruff_summary['warning_count']}"""
        else:
            output_text += f"\n- Failed: {results['ruff']['error']}"
        
        output_text += "\n\nRadon Analysis:"
        if results['radon']['success']:
            radon_summary = results['radon']['summary']
            output_text += f"""
- Total functions: {radon_summary['total_functions']}
- Average complexity: {radon_summary['average_complexity']}
- High complexity functions: {radon_summary['high_complexity_functions']}"""
        else:
            output_text += f"\n- Failed: {results['radon']['error']}"
        
        output_text += "\n\nTODO Analysis:"
        if results['todos']['success']:
            todo_summary = results['todos']['summary']
            output_text += f"""
- Total TODOs: {todo_summary['total_todos']}
- Files with TODOs: {todo_summary['files_with_todos']}"""
        else:
            output_text += f"\n- Failed: {results['todos']['error']}"
        
        # Output results
        if output:
            with open(output, 'w') as f:
                f.write(output_text)
            click.echo(f"Analysis results saved to: {output}")
        else:
            click.echo(output_text)
        
        return 0
        
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        return 1


@cli.command()
@click.argument("repo_path", type=click.Path(exists=True))
@click.option("--output", "-o", help="Output file for evaluation results")
@click.option("--openai-key", envvar="OPENAI_API_KEY", help="OpenAI API key")
def evaluate(repo_path, output, openai_key):
    """Run evaluation on repository using golden Q&A dataset."""
    try:
        from tests.test_eval_harness import run_evaluation
        
        if not openai_key:
            click.echo("Error: OpenAI API key is required. Set OPENAI_API_KEY environment variable or use --openai-key.")
            return 1
        
        click.echo(f"📊 Evaluating repository: {repo_path}")
        
        # Run evaluation
        results = run_evaluation(repo_path, output)
        
        click.echo("✅ Evaluation completed successfully")
        return 0
        
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        return 1


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(host, port, reload):
    """Start the FastAPI server."""
    try:
        import uvicorn
        
        click.echo(f"🚀 Starting RepoMind Agent server on {host}:{port}")
        
        uvicorn.run(
            "repo_mind_agent.orchestrator:app",
            host=host,
            port=port,
            reload=reload
        )
        
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        return 1


@cli.command()
@click.argument("repo_path", type=click.Path(exists=True))
@click.option("--port", default=8501, help="Port for Streamlit")
def ui(repo_path, port):
    """Start the Streamlit UI."""
    try:
        import subprocess
        import sys
        
        click.echo(f"🎨 Starting RepoMind Agent UI on port {port}")
        click.echo(f"Repository: {repo_path}")
        
        # Set environment variable for repository path
        env = os.environ.copy()
        env['REPO_PATH'] = str(repo_path)
        
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "repo_mind_agent/ui/trace_viewer.py",
            "--server.port", str(port),
            "--server.address", "0.0.0.0"
        ], env=env)
        
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        return 1


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main() 