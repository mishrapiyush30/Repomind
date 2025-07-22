"""
SQLite ETL loader for repository data.

Handles the extraction, transformation, and loading of repository data
into a SQLite database for querying commit history, file changes, and issues.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import git
from git import Repo


class SQLiteETLLoader:
    """ETL loader for repository data into SQLite."""
    
    def __init__(self, db_path: str = "repo_data.db"):
        """Initialize the ETL loader with database path."""
        self.db_path = db_path
        self.conn = None
        self._create_schema()
    
    def _create_schema(self):
        """Create the SQLite schema for repository data."""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Commits table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS commits (
                hash TEXT PRIMARY KEY,
                author_name TEXT,
                author_email TEXT,
                commit_date TIMESTAMP,
                message TEXT,
                parent_hash TEXT,
                files_changed INTEGER,
                insertions INTEGER,
                deletions INTEGER
            )
        """)
        
        # Files table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                size INTEGER,
                last_modified TIMESTAMP,
                last_commit_hash TEXT,
                file_type TEXT,
                UNIQUE(path)
            )
        """)
        
        # File changes table (many-to-many between commits and files)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                commit_hash TEXT,
                file_path TEXT,
                change_type TEXT,  -- 'A' (added), 'M' (modified), 'D' (deleted)
                insertions INTEGER,
                deletions INTEGER,
                FOREIGN KEY (commit_hash) REFERENCES commits (hash),
                FOREIGN KEY (file_path) REFERENCES files (path)
            )
        """)
        
        # Issues table (if available from GitHub API)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS issues (
                id INTEGER PRIMARY KEY,
                number INTEGER,
                title TEXT,
                body TEXT,
                state TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                closed_at TIMESTAMP,
                author TEXT,
                assignee TEXT
            )
        """)
        
        # Labels table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                color TEXT,
                description TEXT
            )
        """)
        
        # Issue labels (many-to-many)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS issue_labels (
                issue_id INTEGER,
                label_id INTEGER,
                FOREIGN KEY (issue_id) REFERENCES issues (id),
                FOREIGN KEY (label_id) REFERENCES labels (id),
                PRIMARY KEY (issue_id, label_id)
            )
        """)
        
        # Create indexes for better query performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_commits_date ON commits (commit_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_commits_author ON commits (author_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_files_path ON files (path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_changes_commit ON file_changes (commit_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_changes_file ON file_changes (file_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_issues_state ON issues (state)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_issues_author ON issues (author)")
        
        self.conn.commit()
    
    def load_repo_tree(self, repo_tree_path: str):
        """Load repository tree data from JSON file."""
        with open(repo_tree_path, 'r') as f:
            repo_data = json.load(f)
        
        cursor = self.conn.cursor()
        
        for file_info in repo_data['files']:
            if not file_info['is_dir']:
                file_type = Path(file_info['path']).suffix
                cursor.execute("""
                    INSERT OR REPLACE INTO files (path, size, last_modified, file_type)
                    VALUES (?, ?, ?, ?)
                """, (
                    file_info['path'],
                    file_info['size'],
                    datetime.fromtimestamp(file_info['modified']),
                    file_type
                ))
        
        self.conn.commit()
        print(f"Loaded {len([f for f in repo_data['files'] if not f['is_dir']])} files from repo tree")
    
    def load_git_history(self, repo_path: str):
        """Load Git commit history into the database."""
        repo = Repo(repo_path)
        cursor = self.conn.cursor()
        
        commit_count = 0
        for commit in repo.iter_commits():
            # Insert commit
            cursor.execute("""
                INSERT OR REPLACE INTO commits 
                (hash, author_name, author_email, commit_date, message, parent_hash, files_changed, insertions, deletions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                commit.hexsha,
                commit.author.name,
                commit.author.email,
                datetime.fromtimestamp(commit.committed_date),
                commit.message,
                commit.parents[0].hexsha if commit.parents else None,
                len(commit.stats.files),
                commit.stats.total['insertions'],
                commit.stats.total['deletions']
            ))
            
            # Insert file changes
            for file_path, stats in commit.stats.files.items():
                change_type = 'M'  # Default to modified
                if stats['insertions'] > 0 and stats['deletions'] == 0:
                    change_type = 'A'  # Added
                elif stats['insertions'] == 0 and stats['deletions'] > 0:
                    change_type = 'D'  # Deleted
                
                cursor.execute("""
                    INSERT INTO file_changes (commit_hash, file_path, change_type, insertions, deletions)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    commit.hexsha,
                    file_path,
                    change_type,
                    stats['insertions'],
                    stats['deletions']
                ))
            
            commit_count += 1
            if commit_count % 100 == 0:
                print(f"Processed {commit_count} commits...")
        
        self.conn.commit()
        print(f"Loaded {commit_count} commits from Git history")
    
    def run_query(self, sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Run a SQL query and return results as a list of dictionaries."""
        cursor = self.conn.cursor()
        
        if params:
            cursor.execute(sql, params)
        else:
            cursor.execute(sql)
        
        # Get column names
        columns = [description[0] for description in cursor.description]
        
        # Convert to list of dictionaries
        results = []
        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))
        
        return results
    
    def get_commit_history(self, file_path: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get commit history, optionally filtered by file path."""
        if file_path:
            sql = """
                SELECT c.hash, c.author_name, c.commit_date, c.message, c.files_changed
                FROM commits c
                JOIN file_changes fc ON c.hash = fc.commit_hash
                WHERE fc.file_path = ?
                ORDER BY c.commit_date DESC
                LIMIT ?
            """
            return self.run_query(sql, [file_path, limit])
        else:
            sql = """
                SELECT hash, author_name, commit_date, message, files_changed
                FROM commits
                ORDER BY commit_date DESC
                LIMIT ?
            """
            return self.run_query(sql, [limit])
    
    def get_file_stats(self) -> List[Dict[str, Any]]:
        """Get statistics about files in the repository."""
        sql = """
            SELECT 
                file_type,
                COUNT(*) as count,
                AVG(size) as avg_size,
                SUM(size) as total_size
            FROM files
            WHERE file_type IS NOT NULL
            GROUP BY file_type
            ORDER BY count DESC
        """
        return self.run_query(sql)
    
    def get_most_changed_files(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get files with the most changes."""
        sql = """
            SELECT 
                fc.file_path,
                COUNT(*) as change_count,
                SUM(fc.insertions) as total_insertions,
                SUM(fc.deletions) as total_deletions
            FROM file_changes fc
            GROUP BY fc.file_path
            ORDER BY change_count DESC
            LIMIT ?
        """
        return self.run_query(sql, [limit])
    
    def get_author_stats(self) -> List[Dict[str, Any]]:
        """Get statistics about authors."""
        sql = """
            SELECT 
                author_name,
                author_email,
                COUNT(*) as commit_count,
                SUM(insertions) as total_insertions,
                SUM(deletions) as total_deletions,
                MIN(commit_date) as first_commit,
                MAX(commit_date) as last_commit
            FROM commits
            GROUP BY author_name, author_email
            ORDER BY commit_count DESC
        """
        return self.run_query(sql)
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()


def load_repository_data(repo_path: str, repo_tree_path: str, db_path: str = "repo_data.db"):
    """Convenience function to load all repository data."""
    loader = SQLiteETLLoader(db_path)
    
    try:
        print("Loading repository tree...")
        loader.load_repo_tree(repo_tree_path)
        
        print("Loading Git history...")
        loader.load_git_history(repo_path)
        
        print("Repository data loaded successfully!")
        return loader
    except Exception as e:
        print(f"Error loading repository data: {e}")
        raise
    finally:
        loader.close()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python sql_query.py <repo_path> <repo_tree.json>")
        sys.exit(1)
    
    repo_path = sys.argv[1]
    repo_tree_path = sys.argv[2]
    
    loader = load_repository_data(repo_path, repo_tree_path)
    
    # Example queries
    print("\n=== Recent Commits ===")
    commits = loader.get_commit_history(limit=5)
    for commit in commits:
        print(f"{commit['hash'][:8]} - {commit['author_name']} - {commit['message'][:50]}...")
    
    print("\n=== File Statistics ===")
    file_stats = loader.get_file_stats()
    for stat in file_stats:
        print(f"{stat['file_type']}: {stat['count']} files, {stat['total_size']} bytes")
    
    print("\n=== Most Changed Files ===")
    changed_files = loader.get_most_changed_files(limit=5)
    for file_info in changed_files:
        print(f"{file_info['file_path']}: {file_info['change_count']} changes") 