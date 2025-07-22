"""
Tests for the SQLite ETL loader.
"""

import json
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from datetime import datetime

from repo_mind_agent.tools.sql_query import SQLiteETLLoader


class TestSQLiteETLLoader:
    """Test the SQLite ETL loader functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        yield db_path
        
        # Cleanup
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def sample_repo_tree(self):
        """Create sample repository tree data."""
        return {
            "repo_url": "https://github.com/test/repo",
            "repo_name": "repo",
            "total_files": 3,
            "total_dirs": 2,
            "files": [
                {
                    "path": "src/main.py",
                    "is_dir": False,
                    "size": 1024,
                    "modified": 1234567890
                },
                {
                    "path": "tests/test_main.py",
                    "is_dir": False,
                    "size": 512,
                    "modified": 1234567890
                },
                {
                    "path": "README.md",
                    "is_dir": False,
                    "size": 256,
                    "modified": 1234567890
                },
                {
                    "path": "src",
                    "is_dir": True,
                    "size": 0,
                    "modified": 1234567890
                },
                {
                    "path": "tests",
                    "is_dir": True,
                    "size": 0,
                    "modified": 1234567890
                }
            ]
        }
    
    def test_create_schema(self, temp_db):
        """Test database schema creation."""
        loader = SQLiteETLLoader(temp_db)
        
        # Check that tables were created
        cursor = loader.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['commits', 'files', 'file_changes', 'issues', 'labels', 'issue_labels']
        for table in expected_tables:
            assert table in tables
        
        # Check that indexes were created
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = [row[0] for row in cursor.fetchall()]
        
        expected_indexes = [
            'idx_commits_date', 'idx_commits_author', 'idx_files_path',
            'idx_file_changes_commit', 'idx_file_changes_file',
            'idx_issues_state', 'idx_issues_author'
        ]
        for index in expected_indexes:
            assert index in indexes
        
        loader.close()
    
    def test_load_repo_tree(self, temp_db, sample_repo_tree):
        """Test loading repository tree data."""
        loader = SQLiteETLLoader(temp_db)
        
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_repo_tree, f)
            json_path = f.name
        
        try:
            loader.load_repo_tree(json_path)
            
            # Check that files were loaded
            cursor = loader.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM files")
            file_count = cursor.fetchone()[0]
            assert file_count == 3  # Only files, not directories
            
            # Check specific file data
            cursor.execute("SELECT path, size, file_type FROM files WHERE path = 'src/main.py'")
            file_data = cursor.fetchone()
            assert file_data[0] == "src/main.py"
            assert file_data[1] == 1024
            assert file_data[2] == ".py"
            
        finally:
            Path(json_path).unlink(missing_ok=True)
            loader.close()
    
    @patch('repo_mind_agent.tools.sql_query.Repo')
    def test_load_git_history(self, mock_repo, temp_db):
        """Test loading Git commit history."""
        loader = SQLiteETLLoader(temp_db)
        
        # Mock commit objects
        mock_commit1 = MagicMock()
        mock_commit1.hexsha = "abc123"
        mock_commit1.author.name = "Test Author"
        mock_commit1.author.email = "test@example.com"
        mock_commit1.committed_date = 1234567890
        mock_commit1.message = "Test commit 1"
        mock_commit1.parents = []
        mock_commit1.stats.files = {"src/main.py": {"insertions": 10, "deletions": 2}}
        mock_commit1.stats.total = {"insertions": 10, "deletions": 2}
        
        mock_commit2 = MagicMock()
        mock_commit2.hexsha = "def456"
        mock_commit2.author.name = "Test Author 2"
        mock_commit2.author.email = "test2@example.com"
        mock_commit2.committed_date = 1234567891
        mock_commit2.message = "Test commit 2"
        mock_commit2.parents = [mock_commit1]
        mock_commit2.stats.files = {"tests/test_main.py": {"insertions": 5, "deletions": 0}}
        mock_commit2.stats.total = {"insertions": 5, "deletions": 0}
        
        mock_repo_instance = MagicMock()
        mock_repo_instance.iter_commits.return_value = [mock_commit1, mock_commit2]
        mock_repo.return_value = mock_repo_instance
        
        loader.load_git_history("/fake/repo/path")
        
        # Check that commits were loaded
        cursor = loader.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM commits")
        commit_count = cursor.fetchone()[0]
        assert commit_count == 2
        
        # Check specific commit data
        cursor.execute("SELECT hash, author_name, message FROM commits WHERE hash = 'abc123'")
        commit_data = cursor.fetchone()
        assert commit_data[0] == "abc123"
        assert commit_data[1] == "Test Author"
        assert commit_data[2] == "Test commit 1"
        
        # Check file changes
        cursor.execute("SELECT COUNT(*) FROM file_changes")
        change_count = cursor.fetchone()[0]
        assert change_count == 2
        
        loader.close()
    
    def test_run_query(self, temp_db):
        """Test running SQL queries."""
        loader = SQLiteETLLoader(temp_db)
        
        # Insert test data
        cursor = loader.conn.cursor()
        cursor.execute("""
            INSERT INTO commits (hash, author_name, commit_date, message)
            VALUES (?, ?, ?, ?)
        """, ("abc123", "Test Author", datetime.now(), "Test commit"))
        
        # Test query with parameters
        results = loader.run_query(
            "SELECT hash, author_name FROM commits WHERE hash = ?",
            {"hash": "abc123"}
        )
        
        assert len(results) == 1
        assert results[0]["hash"] == "abc123"
        assert results[0]["author_name"] == "Test Author"
        
        # Test query without parameters
        results = loader.run_query("SELECT COUNT(*) as count FROM commits")
        assert len(results) == 1
        assert results[0]["count"] == 1
        
        loader.close()
    
    def test_get_commit_history(self, temp_db):
        """Test getting commit history."""
        loader = SQLiteETLLoader(temp_db)
        
        # Insert test data
        cursor = loader.conn.cursor()
        cursor.execute("""
            INSERT INTO commits (hash, author_name, commit_date, message, files_changed)
            VALUES 
                (?, ?, ?, ?, ?),
                (?, ?, ?, ?, ?)
        """, (
            "abc123", "Author 1", datetime.now(), "Commit 1", 2,
            "def456", "Author 2", datetime.now(), "Commit 2", 1
        ))
        
        # Test getting all commits
        commits = loader.get_commit_history(limit=5)
        assert len(commits) == 2
        
        # Test getting commits for specific file
        cursor.execute("""
            INSERT INTO file_changes (commit_hash, file_path, change_type, insertions, deletions)
            VALUES (?, ?, ?, ?, ?)
        """, ("abc123", "src/main.py", "M", 10, 2))
        
        commits = loader.get_commit_history(file_path="src/main.py", limit=5)
        assert len(commits) == 1
        assert commits[0]["hash"] == "abc123"
        
        loader.close()
    
    def test_get_file_stats(self, temp_db):
        """Test getting file statistics."""
        loader = SQLiteETLLoader(temp_db)
        
        # Insert test data
        cursor = loader.conn.cursor()
        cursor.execute("""
            INSERT INTO files (path, size, file_type)
            VALUES 
                (?, ?, ?),
                (?, ?, ?),
                (?, ?, ?)
        """, (
            "src/main.py", 1024, ".py",
            "tests/test.py", 512, ".py",
            "README.md", 256, ".md"
        ))
        
        stats = loader.get_file_stats()
        assert len(stats) == 2  # .py and .md
        
        # Check Python files
        py_stats = next(s for s in stats if s["file_type"] == ".py")
        assert py_stats["count"] == 2
        assert py_stats["total_size"] == 1536  # 1024 + 512
        
        loader.close()
    
    def test_get_most_changed_files(self, temp_db):
        """Test getting most changed files."""
        loader = SQLiteETLLoader(temp_db)
        
        # Insert test data
        cursor = loader.conn.cursor()
        cursor.execute("""
            INSERT INTO file_changes (commit_hash, file_path, change_type, insertions, deletions)
            VALUES 
                (?, ?, ?, ?, ?),
                (?, ?, ?, ?, ?),
                (?, ?, ?, ?, ?)
        """, (
            "abc123", "src/main.py", "M", 10, 2,
            "def456", "src/main.py", "M", 5, 1,
            "ghi789", "tests/test.py", "A", 20, 0
        ))
        
        changed_files = loader.get_most_changed_files(limit=5)
        assert len(changed_files) == 2
        
        # src/main.py should have more changes
        main_py = next(f for f in changed_files if f["file_path"] == "src/main.py")
        assert main_py["change_count"] == 2
        
        loader.close()
    
    def test_get_author_stats(self, temp_db):
        """Test getting author statistics."""
        loader = SQLiteETLLoader(temp_db)
        
        # Insert test data
        cursor = loader.conn.cursor()
        cursor.execute("""
            INSERT INTO commits (hash, author_name, author_email, insertions, deletions)
            VALUES 
                (?, ?, ?, ?, ?),
                (?, ?, ?, ?, ?)
        """, (
            "abc123", "Author 1", "author1@example.com", 10, 2,
            "def456", "Author 1", "author1@example.com", 5, 1
        ))
        
        author_stats = loader.get_author_stats()
        assert len(author_stats) == 1
        
        author = author_stats[0]
        assert author["author_name"] == "Author 1"
        assert author["commit_count"] == 2
        assert author["total_insertions"] == 15
        assert author["total_deletions"] == 3
        
        loader.close()


class TestIntegration:
    """Integration tests for the complete ETL process."""
    
    @patch('repo_mind_agent.tools.sql_query.Repo')
    def test_load_repository_data(self, mock_repo, temp_db, sample_repo_tree):
        """Test the complete repository data loading process."""
        from repo_mind_agent.tools.sql_query import load_repository_data
        
        # Mock commit
        mock_commit = MagicMock()
        mock_commit.hexsha = "abc123"
        mock_commit.author.name = "Test Author"
        mock_commit.author.email = "test@example.com"
        mock_commit.committed_date = 1234567890
        mock_commit.message = "Test commit"
        mock_commit.parents = []
        mock_commit.stats.files = {"src/main.py": {"insertions": 10, "deletions": 2}}
        mock_commit.stats.total = {"insertions": 10, "deletions": 2}
        
        mock_repo_instance = MagicMock()
        mock_repo_instance.iter_commits.return_value = [mock_commit]
        mock_repo.return_value = mock_repo_instance
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_repo_tree, f)
            json_path = f.name
        
        try:
            loader = load_repository_data("/fake/repo/path", json_path, temp_db)
            
            # Verify data was loaded
            cursor = loader.conn.cursor()
            
            # Check files
            cursor.execute("SELECT COUNT(*) FROM files")
            file_count = cursor.fetchone()[0]
            assert file_count == 3
            
            # Check commits
            cursor.execute("SELECT COUNT(*) FROM commits")
            commit_count = cursor.fetchone()[0]
            assert commit_count == 1
            
            # Check file changes
            cursor.execute("SELECT COUNT(*) FROM file_changes")
            change_count = cursor.fetchone()[0]
            assert change_count == 1
            
        finally:
            Path(json_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__]) 