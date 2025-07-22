"""
RepoMind Agent v0.1

An intelligent agent that provides instant understanding of codebases through 
semantic search, SQL queries, and static analysis.
"""

__version__ = "0.1.0"
__author__ = "RepoMind Team"
__email__ = "team@repomind.ai"

from .orchestrator import RepoMindAgent, ask

__all__ = ["RepoMindAgent", "ask"] 