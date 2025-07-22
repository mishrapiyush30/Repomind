"""
Tests for the vector search tool.
"""

import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import numpy as np

from repo_mind_agent.tools.vector_search import (
    CodeChunker,
    VectorSearch,
    VectorSearchTool
)


class TestCodeChunker:
    """Test the code chunking functionality."""
    
    def test_chunk_text_simple(self):
        """Test simple text chunking."""
        chunker = CodeChunker(max_tokens=100)
        
        text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        chunks = chunker.chunk_text(text, "test.py")
        
        assert len(chunks) == 1
        assert chunks[0]["file_path"] == "test.py"
        assert chunks[0]["start_line"] == 1
        assert chunks[0]["end_line"] == 5
        assert chunks[0]["text"] == text
    
    def test_chunk_text_large(self):
        """Test chunking of large text."""
        chunker = CodeChunker(max_tokens=50)
        
        # Create text with many lines
        lines = [f"This is line {i} with some content" for i in range(20)]
        text = "\n".join(lines)
        
        chunks = chunker.chunk_text(text, "large_file.py")
        
        assert len(chunks) > 1  # Should be split into multiple chunks
        
        # Check that all chunks have required fields
        for chunk in chunks:
            assert "text" in chunk
            assert "file_path" in chunk
            assert "start_line" in chunk
            assert "end_line" in chunk
            assert "token_estimate" in chunk
            assert chunk["file_path"] == "large_file.py"
    
    def test_chunk_file_success(self, tmp_path):
        """Test successful file chunking."""
        chunker = CodeChunker()
        
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def main():\n    print('Hello')\n    return 0")
        
        chunks = chunker.chunk_file(test_file)
        
        assert len(chunks) == 1
        assert chunks[0]["file_path"] == str(test_file)
        assert "def main():" in chunks[0]["text"]
    
    def test_chunk_file_binary(self, tmp_path):
        """Test handling of binary files."""
        chunker = CodeChunker()
        
        # Create a binary file
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03")
        
        chunks = chunker.chunk_file(test_file)
        
        assert len(chunks) == 0  # Should skip binary files
    
    def test_chunk_file_encoding_error(self, tmp_path):
        """Test handling of encoding errors."""
        chunker = CodeChunker()
        
        # Create a file with invalid encoding
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"\xff\xfe\x00\x01")  # Invalid UTF-8
        
        chunks = chunker.chunk_file(test_file)
        
        assert len(chunks) == 0  # Should handle encoding errors gracefully


class TestVectorSearch:
    """Test the vector search functionality."""
    
    @pytest.fixture
    def mock_model(self):
        """Mock the sentence transformer model."""
        with patch('repo_mind_agent.tools.vector_search.SentenceTransformer') as mock:
            model_instance = MagicMock()
            model_instance.encode.return_value = np.random.rand(1024)
            mock.return_value = model_instance
            yield mock
    
    @pytest.fixture
    def mock_db(self):
        """Mock the database connection."""
        with patch('repo_mind_agent.tools.vector_search.psycopg2.connect') as mock:
            conn_instance = MagicMock()
            cursor_instance = MagicMock()
            conn_instance.__enter__.return_value = conn_instance
            conn_instance.cursor.return_value.__enter__.return_value = cursor_instance
            mock.return_value = conn_instance
            yield mock
    
    def test_init_database(self, mock_model, mock_db):
        """Test database initialization."""
        vector_search = VectorSearch()
        
        # Check that database initialization was called
        mock_db.assert_called()
        
        # Check that cursor was used to create tables and indexes
        cursor = mock_db.return_value.cursor.return_value.__enter__.return_value
        assert cursor.execute.call_count >= 4  # Tables + indexes
    
    def test_embed_text(self, mock_model, mock_db):
        """Test text embedding."""
        vector_search = VectorSearch()
        
        text = "Test text for embedding"
        embedding = vector_search.embed_text(text)
        
        # Check that model was called
        vector_search.model.encode.assert_called_with(text, normalize_embeddings=True)
        assert isinstance(embedding, np.ndarray)
    
    def test_embed_chunks(self, mock_model, mock_db):
        """Test chunk embedding."""
        vector_search = VectorSearch()
        
        chunks = [
            {"text": "Chunk 1", "file_path": "test1.py", "start_line": 1, "end_line": 5},
            {"text": "Chunk 2", "file_path": "test2.py", "start_line": 1, "end_line": 3}
        ]
        
        result = vector_search.embed_chunks(chunks)
        
        # Check that embeddings were added
        assert len(result) == 2
        for chunk in result:
            assert "embedding" in chunk
            assert isinstance(chunk["embedding"], np.ndarray)
        
        # Check that model was called with all texts
        vector_search.model.encode.assert_called_with(
            ["Chunk 1", "Chunk 2"], 
            normalize_embeddings=True
        )
    
    def test_search(self, mock_model, mock_db):
        """Test search functionality."""
        vector_search = VectorSearch()
        
        # Mock search results
        mock_results = [
            {
                "file_path": "test.py",
                "start_line": 1,
                "end_line": 5,
                "content": "def main(): pass",
                "similarity": 0.85
            }
        ]
        
        cursor = mock_db.return_value.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = mock_results
        
        results = vector_search.search("main function", top_k=5)
        
        # Check that search was executed
        assert cursor.execute.call_count > 0
        
        # Check results
        assert len(results) == 1
        assert results[0]["file_path"] == "test.py"
        assert results[0]["similarity"] == 0.85


class TestVectorSearchTool:
    """Test the main vector search tool."""
    
    @pytest.fixture
    def mock_vector_search(self):
        """Mock the vector search component."""
        with patch('repo_mind_agent.tools.vector_search.VectorSearch') as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture
    def mock_chunker(self):
        """Mock the chunker component."""
        with patch('repo_mind_agent.tools.vector_search.CodeChunker') as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance
    
    def test_init(self, mock_vector_search, mock_chunker):
        """Test tool initialization."""
        tool = VectorSearchTool()
        
        # Check that components were initialized
        assert tool.chunker is not None
        assert tool.vector_search is not None
    
    def test_should_skip_file(self):
        """Test file skipping logic."""
        tool = VectorSearchTool()
        
        # Files that should be skipped
        skip_files = [
            Path(".git/config"),
            Path("__pycache__/module.pyc"),
            Path("node_modules/package"),
            Path("venv/bin/python"),
            Path(".env"),
            Path(".DS_Store"),
            Path("build/dist")
        ]
        
        for file_path in skip_files:
            assert tool._should_skip_file(file_path)
        
        # Files that should not be skipped
        keep_files = [
            Path("src/main.py"),
            Path("README.md"),
            Path("tests/test_file.py")
        ]
        
        for file_path in keep_files:
            assert not tool._should_skip_file(file_path)
    
    def test_search_methods(self, mock_vector_search, mock_chunker):
        """Test search method wrappers."""
        tool = VectorSearchTool()
        
        # Mock search results
        mock_results = [{"file_path": "test.py", "content": "test"}]
        tool.vector_search.search.return_value = mock_results
        
        # Test general search
        results = tool.search("test query")
        assert results == mock_results
        tool.vector_search.search.assert_called_with("test query", 5, None)
        
        # Test function search
        results = tool.search_function("main")
        assert results == mock_results
        tool.vector_search.search.assert_called_with("function main definition", 5, None)
        
        # Test class search
        results = tool.search_class("MyClass")
        assert results == mock_results
        tool.vector_search.search.assert_called_with("class MyClass definition", 5, None)
        
        # Test error search
        results = tool.search_error("FileNotFoundError")
        assert results == mock_results
        tool.vector_search.search.assert_called_with("error handling FileNotFoundError", 5, None)


class TestIntegration:
    """Integration tests for the complete vector search workflow."""
    
    @patch('repo_mind_agent.tools.vector_search.VectorSearch')
    @patch('repo_mind_agent.tools.vector_search.CodeChunker')
    def test_create_embeddings_from_repo_tree(self, mock_chunker, mock_vector_search, tmp_path):
        """Test creating embeddings from repository tree."""
        from repo_mind_agent.tools.vector_search import create_embeddings_from_repo_tree
        
        # Create mock repository tree
        repo_tree = {
            "files": [
                {
                    "path": "src/main.py",
                    "is_dir": False,
                    "size": 100
                },
                {
                    "path": "tests/test_main.py",
                    "is_dir": False,
                    "size": 50
                },
                {
                    "path": "README.md",
                    "is_dir": False,
                    "size": 25
                },
                {
                    "path": "src",
                    "is_dir": True,
                    "size": 0
                }
            ]
        }
        
        # Create temporary files
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        (repo_path / "src").mkdir()
        (repo_path / "tests").mkdir()
        
        (repo_path / "src" / "main.py").write_text("def main(): pass")
        (repo_path / "tests" / "test_main.py").write_text("def test_main(): pass")
        (repo_path / "README.md").write_text("# Test Repo")
        
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(repo_tree, f)
            json_path = f.name
        
        try:
            # Mock chunker and vector search
            mock_chunker_instance = MagicMock()
            mock_chunker.return_value = mock_chunker_instance
            mock_chunker_instance.chunk_file.return_value = [
                {"text": "test chunk", "file_path": "test.py", "start_line": 1, "end_line": 5}
            ]
            
            mock_vector_instance = MagicMock()
            mock_vector_search.return_value = mock_vector_instance
            mock_vector_instance.embed_chunks.return_value = [
                {"text": "test chunk", "file_path": "test.py", "start_line": 1, "end_line": 5, "embedding": np.random.rand(1024)}
            ]
            
            # Create embeddings
            tool = create_embeddings_from_repo_tree(json_path, str(repo_path))
            
            # Verify that chunker was called for each text file
            assert mock_chunker_instance.chunk_file.call_count == 3
            
            # Verify that embeddings were generated and stored
            assert mock_vector_instance.embed_chunks.call_count == 3
            assert mock_vector_instance.upsert_chunks.call_count == 3
            
        finally:
            Path(json_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__]) 