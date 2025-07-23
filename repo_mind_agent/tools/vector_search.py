"""
Vector search tool for code and documentation.

Handles chunking, embedding, and semantic search of repository content
using sentence-transformers and pgvector.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import RealDictCursor
import psycopg2.extras


class CodeChunker:
    """Chunks code and documentation into searchable segments."""
    
    def __init__(self, max_tokens: int = 400):
        """Initialize the chunker with maximum token limit."""
        self.max_tokens = max_tokens
    
    def chunk_text(self, text: str, file_path: str) -> List[Dict[str, Any]]:
        """Chunk text into segments with metadata."""
        chunks = []
        
        # Simple newline-based chunking
        lines = text.split('\n')
        current_chunk = []
        current_line_start = 1
        
        for i, line in enumerate(lines, 1):
            current_chunk.append(line)
            
            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            estimated_tokens = sum(len(line) for line in current_chunk) // 4
            
            if estimated_tokens >= self.max_tokens:
                # Create chunk
                chunk_text = '\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'file_path': file_path,
                    'start_line': current_line_start,
                    'end_line': i,
                    'token_estimate': estimated_tokens
                })
                
                # Start new chunk
                current_chunk = []
                current_line_start = i + 1
        
        # Add remaining content as final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            estimated_tokens = sum(len(line) for line in current_chunk) // 4
            chunks.append({
                'text': chunk_text,
                'file_path': file_path,
                'start_line': current_line_start,
                'end_line': len(lines),
                'token_estimate': estimated_tokens
            })
        
        return chunks
    
    def chunk_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Chunk a single file into searchable segments."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return self.chunk_text(content, str(file_path))
        except (UnicodeDecodeError, IOError) as e:
            # Skip binary files or files with encoding issues
            return []


class VectorSearch:
    """Vector search functionality using sentence-transformers and pgvector."""
    
    def __init__(self, model_name: str = "BAAI/bge-large-en", db_url: Optional[str] = None):
        """Initialize vector search with model and database connection."""
        self.model = SentenceTransformer(model_name)
        self.db_url = db_url or "postgresql://localhost/repomind"
        self._init_database()
    
    def _init_database(self):
        """Initialize the PostgreSQL database with pgvector extension."""
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cursor:
                # Enable pgvector extension
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                
                # Create embeddings table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS code_embeddings (
                        id SERIAL PRIMARY KEY,
                        file_path TEXT NOT NULL,
                        start_line INTEGER,
                        end_line INTEGER,
                        content TEXT NOT NULL,
                        embedding vector(1024),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(file_path, start_line, end_line)
                    )
                """)
                
                # Create indexes
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_code_embeddings_file_path 
                    ON code_embeddings (file_path)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_code_embeddings_embedding 
                    ON code_embeddings USING ivfflat (embedding vector_cosine_ops)
                """)
                
                conn.commit()
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a text string."""
        return self.model.encode(text, normalize_embeddings=True)
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for a list of text chunks."""
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
        
        return chunks
    
    def upsert_chunks(self, chunks: List[Dict[str, Any]]):
        """Upsert chunks with embeddings into the database."""
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cursor:
                for chunk in chunks:
                    # Convert numpy array to PostgreSQL vector format
                    embedding_list = chunk['embedding'].tolist()
                    embedding_str = "[" + ",".join(str(x) for x in embedding_list) + "]"
                    
                    cursor.execute("""
                        INSERT INTO code_embeddings (file_path, start_line, end_line, content, embedding)
                        VALUES (%s, %s, %s, %s, %s::vector)
                        ON CONFLICT (file_path, start_line, end_line) 
                        DO UPDATE SET 
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            created_at = CURRENT_TIMESTAMP
                    """, (
                        chunk['file_path'],
                        chunk['start_line'],
                        chunk['end_line'],
                        chunk['text'],
                        embedding_str
                    ))
                conn.commit()
    
    def search(self, query: str, top_k: int = 5, file_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar code chunks."""
        # Generate query embedding
        query_embedding = self.embed_text(query)
        
        # Build search query
        # Convert numpy array to list and then to proper PostgreSQL vector format
        embedding_list = query_embedding.tolist()
        embedding_str = "[" + ",".join(str(x) for x in embedding_list) + "]"
        
        if file_filter:
            sql = """
                SELECT file_path, start_line, end_line, content, 
                       embedding <-> %s::vector as distance
                FROM code_embeddings
                WHERE file_path LIKE %s
                ORDER BY embedding <-> %s::vector
                LIMIT %s
            """
            params = (embedding_str, f"%{file_filter}%", embedding_str, top_k)
        else:
            sql = """
                SELECT file_path, start_line, end_line, content, 
                       embedding <-> %s::vector as distance
                FROM code_embeddings
                ORDER BY embedding <-> %s::vector
                LIMIT %s
            """
            params = (embedding_str, embedding_str, top_k)
        
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(sql, params)
                results = cursor.fetchall()
        
        return [dict(result) for result in results]
    
    def get_file_chunks(self, file_path: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific file."""
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT file_path, start_line, end_line, content
                    FROM code_embeddings
                    WHERE file_path = %s
                    ORDER BY start_line
                """, (file_path,))
                results = cursor.fetchall()
        
        return [dict(result) for result in results]
    
    def delete_file_chunks(self, file_path: str):
        """Delete all chunks for a specific file."""
        with psycopg2.connect(self.db_url) as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    DELETE FROM code_embeddings
                    WHERE file_path = %s
                """, (file_path,))
                conn.commit()


class VectorSearchTool:
    """Main tool class that combines chunking and vector search."""
    
    def __init__(self, model_name: str = "BAAI/bge-large-en", db_url: Optional[str] = None):
        """Initialize the vector search tool."""
        self.chunker = CodeChunker()
        self.vector_search = VectorSearch(model_name, db_url)
    
    def process_repository(self, repo_path: str, file_patterns: Optional[List[str]] = None):
        """Process an entire repository and create embeddings."""
        repo_path = Path(repo_path)
        
        if file_patterns is None:
            file_patterns = [
                "*.py", "*.js", "*.ts", "*.java", "*.cpp", "*.c", "*.h",
                "*.md", "*.txt", "*.rst", "*.yaml", "*.yml", "*.json"
            ]
        
        processed_files = 0
        total_chunks = 0
        
        for pattern in file_patterns:
            for file_path in repo_path.rglob(pattern):
                if self._should_skip_file(file_path):
                    continue
                
                chunks = self.chunker.chunk_file(file_path)
                if chunks:
                    # Generate embeddings
                    chunks_with_embeddings = self.vector_search.embed_chunks(chunks)
                    
                    # Upsert to database
                    self.vector_search.upsert_chunks(chunks_with_embeddings)
                    
                    processed_files += 1
                    total_chunks += len(chunks)
                    
                    print(f"Processed {file_path}: {len(chunks)} chunks")
        
        print(f"Total: {processed_files} files, {total_chunks} chunks")
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if a file should be skipped."""
        skip_patterns = [
            ".git", "__pycache__", "node_modules", ".venv", "venv",
            ".pytest_cache", ".coverage", "htmlcov", "dist", "build",
            "*.pyc", "*.pyo", "*.pyd", "*.so", "*.dll", "*.exe"
        ]
        
        file_str = str(file_path)
        for pattern in skip_patterns:
            if pattern in file_str:
                return True
        return False
    
    def search(self, query: str, top_k: int = 5, file_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for code chunks similar to the query."""
        return self.vector_search.search(query, top_k, file_filter)
    
    def search_function(self, function_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for function definitions."""
        query = f"function {function_name} definition"
        return self.search(query, top_k)
    
    def search_class(self, class_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for class definitions."""
        query = f"class {class_name} definition"
        return self.search(query, top_k)
    
    def search_error(self, error_message: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for error handling or error messages."""
        query = f"error handling {error_message}"
        return self.search(query, top_k)


def create_embeddings_from_repo_tree(repo_tree_path: str, repo_path: str, 
                                   db_url: Optional[str] = None) -> VectorSearchTool:
    """Create embeddings from a repository tree JSON file."""
    with open(repo_tree_path, 'r') as f:
        repo_data = json.load(f)
    
    tool = VectorSearchTool(db_url=db_url)
    
    # Process only text files
    text_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', 
                      '.md', '.txt', '.rst', '.yaml', '.yml', '.json'}
    
    for file_info in repo_data['files']:
        if file_info['is_dir']:
            continue
        
        file_path = Path(file_info['path'])
        if file_path.suffix not in text_extensions:
            continue
        
        full_path = Path(repo_path) / file_path
        if not full_path.exists():
            continue
        
        chunks = tool.chunker.chunk_file(full_path)
        if chunks:
            chunks_with_embeddings = tool.vector_search.embed_chunks(chunks)
            tool.vector_search.upsert_chunks(chunks_with_embeddings)
            print(f"Processed {file_path}: {len(chunks)} chunks")
    
    return tool


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python vector_search.py <repo_path> <repo_tree.json> [db_url]")
        sys.exit(1)
    
    repo_path = sys.argv[1]
    repo_tree_path = sys.argv[2]
    db_url = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Create embeddings
    tool = create_embeddings_from_repo_tree(repo_tree_path, repo_path, db_url)
    
    # Example searches
    print("\n=== Searching for main function ===")
    results = tool.search_function("main")
    for result in results:
        print(f"{result['file_path']}:{result['start_line']}-{result['end_line']} (similarity: {result['similarity']:.3f})")
        print(f"Content: {result['content'][:100]}...")
        print()
    
    print("=== Searching for error handling ===")
    results = tool.search_error("FileNotFoundError")
    for result in results:
        print(f"{result['file_path']}:{result['start_line']}-{result['end_line']} (similarity: {result['similarity']:.3f})")
        print(f"Content: {result['content'][:100]}...")
        print() 