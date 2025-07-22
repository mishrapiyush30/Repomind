#!/usr/bin/env python3
"""
Repository ingestion script for RepoMind Agent.

Clones a GitHub repository and generates a structured JSON representation
of the repository tree for further processing.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Any
import click
import git


def get_file_info(file_path: Path, repo_root: Path) -> Dict[str, Any]:
    """Get information about a file."""
    try:
        stat = file_path.stat()
        return {
            "path": str(file_path.relative_to(repo_root)),
            "is_dir": file_path.is_dir(),
            "size": stat.st_size if file_path.is_file() else 0,
            "modified": stat.st_mtime,
        }
    except (OSError, ValueError) as e:
        return {
            "path": str(file_path.relative_to(repo_root)),
            "is_dir": False,  # Assume it's not a directory if we can't access it
            "size": 0,
            "error": str(e),
        }


def should_ignore(path: Path) -> bool:
    """Check if a path should be ignored."""
    ignore_patterns = {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".env",
        ".DS_Store",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".pytest_cache",
        ".coverage",
        "htmlcov",
        "dist",
        "build",
        "*.egg-info",
    }
    
    path_str = str(path)
    for pattern in ignore_patterns:
        if pattern in path_str:
            return True
    return False


def scan_repository(repo_path: Path) -> List[Dict[str, Any]]:
    """Scan repository and return structured file information."""
    files = []
    
    for item in repo_path.rglob("*"):
        if should_ignore(item):
            continue
            
        file_info = get_file_info(item, repo_path)
        files.append(file_info)
    
    # Sort by path for consistent output
    files.sort(key=lambda x: x["path"])
    return files


def clone_repository(repo_url: str, target_dir: Path) -> Path:
    """Clone a repository to the target directory."""
    try:
        print(f"Cloning {repo_url} to {target_dir}...")
        repo = git.Repo.clone_from(repo_url, target_dir)
        print(f"Successfully cloned repository. Latest commit: {repo.head.commit.hexsha[:8]}")
        return target_dir
    except git.GitCommandError as e:
        print(f"Error cloning repository: {e}")
        sys.exit(1)


@click.command()
@click.argument("repo_url", type=str)
@click.option("--output", "-o", default="repo_tree.json", help="Output JSON file path")
@click.option("--temp-dir", "-t", help="Temporary directory for cloning")
def main(repo_url: str, output: str, temp_dir: str):
    """Ingest a GitHub repository and generate structured JSON."""
    
    # Setup temporary directory
    if temp_dir:
        temp_path = Path(temp_dir)
        temp_path.mkdir(exist_ok=True)
    else:
        temp_path = Path(tempfile.mkdtemp())
    
    try:
        # Extract repo name from URL
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        repo_path = temp_path / repo_name
        
        # Clone repository
        repo_path = clone_repository(repo_url, repo_path)
        
        # Scan repository
        print("Scanning repository structure...")
        files = scan_repository(repo_path)
        
        # Create output structure
        output_data = {
            "repo_url": repo_url,
            "repo_name": repo_name,
            "total_files": len([f for f in files if not f["is_dir"]]),
            "total_dirs": len([f for f in files if f["is_dir"]]),
            "files": files,
        }
        
        # Write output
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Repository structure saved to {output}")
        print(f"Found {output_data['total_files']} files and {output_data['total_dirs']} directories")
        
    except Exception as e:
        print(f"Error processing repository: {e}")
        sys.exit(1)
    finally:
        # Cleanup temporary directory if we created it
        if not temp_dir and temp_path.exists():
            import shutil
            shutil.rmtree(temp_path)


if __name__ == "__main__":
    main() 