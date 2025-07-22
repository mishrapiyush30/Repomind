"""
Tests for the RepoMind Agent orchestrator.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from datetime import datetime

from repo_mind_agent.orchestrator import (
    RepoMindAgent, 
    ToolCall, 
    ReActStep, 
    ask,
    QuestionRequest,
    QuestionResponse
)


class TestToolCall:
    """Test the ToolCall model."""
    
    def test_tool_call_creation(self):
        """Test creating a ToolCall."""
        tool_call = ToolCall(
            tool="vector_search",
            input={"query": "test query"}
        )
        
        assert tool_call.tool == "vector_search"
        assert tool_call.input == {"query": "test query"}
        assert tool_call.output is None
        assert tool_call.error is None
    
    def test_tool_call_with_output(self):
        """Test ToolCall with output."""
        tool_call = ToolCall(
            tool="vector_search",
            input={"query": "test"},
            output={"results": ["result1", "result2"]}
        )
        
        assert tool_call.output == {"results": ["result1", "result2"]}


class TestReActStep:
    """Test the ReActStep model."""
    
    def test_react_step_creation(self):
        """Test creating a ReActStep."""
        step = ReActStep(
            step_type="think",
            content="Test content"
        )
        
        assert step.step_type == "think"
        assert step.content == "Test content"
        assert step.tool_call is None
        assert isinstance(step.timestamp, datetime)


class TestRepoMindAgent:
    """Test the RepoMindAgent class."""
    
    @pytest.fixture
    def temp_repo(self):
        """Create a temporary repository for testing."""
        temp_dir = tempfile.mkdtemp()
        repo_path = Path(temp_dir)
        
        # Create some test files
        (repo_path / "test.py").write_text("print('Hello, World!')")
        (repo_path / "README.md").write_text("# Test Repository")
        
        return str(repo_path)
    
    @pytest.fixture
    def mock_agent(self, temp_repo):
        """Create a mock agent with mocked dependencies."""
        with patch('repo_mind_agent.orchestrator.VectorSearchTool'), \
             patch('repo_mind_agent.orchestrator.SQLiteETLLoader'), \
             patch('repo_mind_agent.orchestrator.StaticAnalyzer'), \
             patch('repo_mind_agent.orchestrator.openai'):
            
            agent = RepoMindAgent(
                repo_path=temp_repo,
                openai_api_key="test-key"
            )
            yield agent
    
    def test_agent_initialization(self, temp_repo):
        """Test agent initialization."""
        with patch('repo_mind_agent.orchestrator.VectorSearchTool'), \
             patch('repo_mind_agent.orchestrator.SQLiteETLLoader'), \
             patch('repo_mind_agent.orchestrator.StaticAnalyzer'):
            
            agent = RepoMindAgent(temp_repo)
            
            assert str(agent.repo_path) == temp_repo
            assert agent.db_path == "repo_data.db"
            assert len(agent.trace) == 0
    
    @patch('repo_mind_agent.orchestrator.openai.ChatCompletion.create')
    def test_think_step(self, mock_openai, mock_agent):
        """Test the think step."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test thought"
        mock_openai.return_value = mock_response
        
        result = mock_agent.think("What is the main function?")
        
        assert result == "Test thought"
        mock_openai.assert_called_once()
    
    @patch('repo_mind_agent.orchestrator.openai.ChatCompletion.create')
    def test_act_step_vector_search(self, mock_openai, mock_agent):
        """Test act step with vector search tool."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '''
        {
            "tool": "vector_search",
            "input": {"query": "main function"}
        }
        '''
        mock_openai.return_value = mock_response
        
        tool_call = mock_agent.act("Need to search for main function", "Where is main?")
        
        assert tool_call.tool == "vector_search"
        assert tool_call.input["query"] == "main function"
    
    @patch('repo_mind_agent.orchestrator.openai.ChatCompletion.create')
    def test_act_step_static_analysis(self, mock_openai, mock_agent):
        """Test act step with static analysis tool."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '''
        {
            "tool": "static_analysis",
            "input": {}
        }
        '''
        mock_openai.return_value = mock_response
        
        tool_call = mock_agent.act("Need code quality metrics", "What's the code quality?")
        
        assert tool_call.tool == "static_analysis"
        assert tool_call.input == {}
    
    def test_execute_tool_vector_search(self, mock_agent):
        """Test executing vector search tool."""
        mock_agent.vector_tool.search.return_value = [
            {"file": "test.py", "content": "def main()", "line": 1}
        ]
        
        tool_call = ToolCall(
            tool="vector_search",
            input={"query": "main function"}
        )
        
        result = mock_agent.execute_tool(tool_call)
        
        assert result["tool"] == "vector_search"
        assert result["count"] == 1
        assert "results" in result
        mock_agent.vector_tool.search.assert_called_once()
    
    def test_execute_tool_static_analysis(self, mock_agent):
        """Test executing static analysis tool."""
        mock_agent.static_analyzer.analyze_repository.return_value = {
            "ruff": {"success": True, "summary": {"error_count": 0}},
            "radon": {"success": True, "results": []},
            "todos": {"success": True, "summary": {"total_todos": 0}}
        }
        
        tool_call = ToolCall(
            tool="static_analysis",
            input={}
        )
        
        result = mock_agent.execute_tool(tool_call)
        
        assert result["tool"] == "static_analysis"
        assert "results" in result
        mock_agent.static_analyzer.analyze_repository.assert_called_once()
    
    def test_execute_tool_sql_query(self, mock_agent):
        """Test executing SQL query tool."""
        mock_agent.sql_loader.run_query.return_value = [
            {"hash": "abc123", "message": "Initial commit"}
        ]
        
        tool_call = ToolCall(
            tool="sql_query",
            input={"sql": "SELECT * FROM commits LIMIT 1"}
        )
        
        result = mock_agent.execute_tool(tool_call)
        
        assert result["tool"] == "sql_query"
        assert result["count"] == 1
        assert "results" in result
        mock_agent.sql_loader.run_query.assert_called_once()
    
    @patch('repo_mind_agent.orchestrator.openai.ChatCompletion.create')
    def test_observe_step(self, mock_openai, mock_agent):
        """Test the observe step."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Found main function in test.py"
        mock_openai.return_value = mock_response
        
        tool_output = {
            "tool": "vector_search",
            "results": [{"file": "test.py", "content": "def main()"}]
        }
        
        result = mock_agent.observe(tool_output)
        
        assert result == "Found main function in test.py"
        mock_openai.assert_called_once()
    
    @patch('repo_mind_agent.orchestrator.openai.ChatCompletion.create')
    def test_synthesize_step(self, mock_openai, mock_agent):
        """Test the synthesize step."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "The main function is in test.py"
        mock_openai.return_value = mock_response
        
        trace = [
            ReActStep(step_type="think", content="Need to find main function"),
            ReActStep(step_type="observe", content="Found main function in test.py")
        ]
        
        result = mock_agent.synthesize("Where is main?", trace)
        
        assert result == "The main function is in test.py"
        mock_openai.assert_called_once()
    
    def test_extract_citations(self, mock_agent):
        """Test citation extraction."""
        tool_output = {
            "tool": "vector_search",
            "results": [
                {
                    "file_path": "test.py",
                    "content": "def main():",
                    "start_line": 1
                }
            ]
        }
        
        citations = mock_agent._extract_citations(tool_output)
        
        assert len(citations) == 1
        assert citations[0]["file"] == "test.py"
        assert citations[0]["line"] == 1
        assert citations[0]["content"] == "def main():..."


class TestAskFunction:
    """Test the ask function."""
    
    @patch('repo_mind_agent.orchestrator.RepoMindAgent')
    def test_ask_function(self, mock_agent_class):
        """Test the ask function."""
        mock_agent = MagicMock()
        mock_agent.ask.return_value = {
            "question": "Where is main?",
            "answer": "Main is in test.py",
            "citations": [{"file": "test.py", "line": 1}],
            "steps_taken": 3
        }
        mock_agent_class.return_value = mock_agent
        
        result = ask(
            question="Where is main?",
            repo_path="/test/repo",
            openai_api_key="test-key"
        )
        
        assert result["question"] == "Where is main?"
        assert result["answer"] == "Main is in test.py"
        assert len(result["citations"]) == 1
        assert result["steps_taken"] == 3
        mock_agent_class.assert_called_once()


class TestAPIModels:
    """Test the API models."""
    
    def test_question_request(self):
        """Test QuestionRequest model."""
        request = QuestionRequest(
            question="Where is main?",
            repo_path="/test/repo"
        )
        
        assert request.question == "Where is main?"
        assert request.repo_path == "/test/repo"
    
    def test_question_response(self):
        """Test QuestionResponse model."""
        response = QuestionResponse(
            question="Where is main?",
            answer="Main is in test.py",
            trace=[],
            citations=[{"file": "test.py", "line": 1}],
            steps_taken=3
        )
        
        assert response.question == "Where is main?"
        assert response.answer == "Main is in test.py"
        assert len(response.citations) == 1
        assert response.steps_taken == 3 