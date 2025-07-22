"""
Tests for the repository ingestion script.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from scripts.ingest_repo import (
    get_file_info,
    should_ignore,
    scan_repository,
    clone_repository,
)


class TestFileInfo:
    """Test file information extraction."""
    
    def test_get_file_info_file(self, tmp_path):
        """Test getting info for a regular file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")
        
        info = get_file_info(test_file, tmp_path)
        
        assert info["path"] == "test.txt"
        assert info["is_dir"] is False
        assert info["size"] == 13
        assert "modified" in info
    
    def test_get_file_info_directory(self, tmp_path):
        """Test getting info for a directory."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        
        info = get_file_info(test_dir, tmp_path)
        
        assert info["path"] == "test_dir"
        assert info["is_dir"] is True
        assert info["size"] == 0
        assert "modified" in info
    
    def test_get_file_info_error(self, tmp_path):
        """Test handling of file access errors."""
        # Create a file that will cause an error when accessed
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        
        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.side_effect = OSError("Permission denied")
            
            info = get_file_info(test_file, tmp_path)
            
            assert info["path"] == "test.txt"
            assert "error" in info
            assert "Permission denied" in info["error"]


class TestIgnorePatterns:
    """Test file/directory ignore patterns."""
    
    @pytest.mark.parametrize("path,expected", [
        (".git/config", True),
        ("src/__pycache__/module.pyc", True),
        ("node_modules/package", True),
        ("venv/bin/python", True),
        ("src/main.py", False),
        ("README.md", False),
        ("tests/test_file.py", False),
        (".env", True),
        (".DS_Store", True),
        ("build/dist", True),
    ])
    def test_should_ignore(self, path, expected):
        """Test various ignore patterns."""
        result = should_ignore(Path(path))
        assert result == expected


class TestRepositoryScanning:
    """Test repository scanning functionality."""
    
    def test_scan_repository_structure(self, tmp_path):
        """Test scanning a repository structure."""
        # Create a mock repository structure
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()
        (tmp_path / "src" / "main.py").write_text("def main(): pass")
        (tmp_path / "tests" / "test_main.py").write_text("def test_main(): pass")
        (tmp_path / "README.md").write_text("# Test Repo")
        (tmp_path / ".git").mkdir()  # Should be ignored
        (tmp_path / "__pycache__").mkdir()  # Should be ignored
        
        files = scan_repository(tmp_path)
        
        # Should have 5 items: src/, tests/, src/main.py, tests/test_main.py, README.md
        assert len(files) == 5
        
        # Check that ignored directories are not included
        paths = [f["path"] for f in files]
        assert ".git" not in paths
        assert "__pycache__" not in paths
        
        # Check that all files have required fields
        for file_info in files:
            assert "path" in file_info
            assert "is_dir" in file_info
            assert "size" in file_info
            assert "modified" in file_info


class TestRepositoryCloning:
    """Test repository cloning functionality."""
    
    @patch("scripts.ingest_repo.git.Repo")
    def test_clone_repository_success(self, mock_repo, tmp_path):
        """Test successful repository cloning."""
        mock_repo_instance = MagicMock()
        mock_repo_instance.head.commit.hexsha = "abcdef1234567890"
        mock_repo.clone_from.return_value = mock_repo_instance
        
        repo_path = clone_repository("https://github.com/test/repo", tmp_path)
        
        assert repo_path == tmp_path
        mock_repo.clone_from.assert_called_once_with(
            "https://github.com/test/repo", tmp_path
        )
    
    @patch("scripts.ingest_repo.git.Repo")
    def test_clone_repository_error(self, mock_repo, tmp_path):
        """Test repository cloning error handling."""
        from git import GitCommandError
        
        mock_repo.clone_from.side_effect = GitCommandError("clone", "Repository not found")
        
        with pytest.raises(SystemExit):
            clone_repository("https://github.com/test/repo", tmp_path)


class TestIntegration:
    """Integration tests for the complete ingestion process."""
    
    @patch("scripts.ingest_repo.clone_repository")
    @patch("scripts.ingest_repo.scan_repository")
    def test_main_integration(self, mock_scan, mock_clone, tmp_path):
        """Test the main function integration."""
        from scripts.ingest_repo import main
        
        # Mock the scan results
        mock_scan.return_value = [
            {"path": "src/main.py", "is_dir": False, "size": 100, "modified": 1234567890},
            {"path": "README.md", "is_dir": False, "size": 200, "modified": 1234567890},
        ]
        
        # Mock the clone result
        mock_clone.return_value = tmp_path
        
        # Create output file path
        output_file = tmp_path / "repo_tree.json"
        
        # Test the main function
        with patch("sys.argv", ["ingest_repo.py", "https://github.com/test/repo"]):
            with patch("click.Context"):
                main.callback("https://github.com/test/repo", str(output_file), None)
        
        # Verify output file was created
        assert output_file.exists()
        
        # Verify JSON structure
        with open(output_file) as f:
            data = json.load(f)
        
        assert data["repo_url"] == "https://github.com/test/repo"
        assert data["repo_name"] == "repo"
        assert data["total_files"] == 2
        assert data["total_dirs"] == 0
        assert len(data["files"]) == 2


if __name__ == "__main__":
    pytest.main([__file__]) 