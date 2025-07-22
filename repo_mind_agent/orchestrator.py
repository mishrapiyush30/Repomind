"""
ReAct orchestrator for RepoMind Agent.

Coordinates vector search, SQL queries, and static analysis tools
to provide intelligent answers about codebases.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import openai
from pydantic import BaseModel, Field

from .tools.vector_search import VectorSearchTool
from .tools.sql_query import SQLiteETLLoader
from .tools.static_analysis import StaticAnalyzer


class ToolCall(BaseModel):
    """Represents a tool call in the ReAct loop."""
    tool: str
    input: Dict[str, Any]
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ReActStep(BaseModel):
    """Represents a single step in the ReAct loop."""
    step_type: str  # "think", "act", "observe"
    content: str
    tool_call: Optional[ToolCall] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class RepoMindAgent:
    """Main ReAct agent for repository analysis."""
    
    def __init__(self, repo_path: str, db_path: str = "repo_data.db", 
                 vector_db_url: Optional[str] = None, openai_api_key: Optional[str] = None):
        """Initialize the RepoMind agent."""
        self.repo_path = Path(repo_path)
        self.db_path = db_path
        self.vector_db_url = vector_db_url
        
        # Initialize tools
        self.sql_loader = SQLiteETLLoader(db_path)
        self.vector_tool = VectorSearchTool(db_url=vector_db_url)
        self.static_analyzer = StaticAnalyzer(str(repo_path))
        
        # Initialize OpenAI client
        if openai_api_key:
            openai.api_key = openai_api_key
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # ReAct loop state
        self.max_steps = 5
        self.trace: List[ReActStep] = []
    
    def think(self, question: str, context: str = "") -> str:
        """Think step: analyze the question and plan the approach."""
        prompt = f"""
You are RepoMind Agent, an intelligent assistant that helps understand codebases.

Question: {question}
Context: {context}

Available tools:
1. vector_search - Search for code and documentation using semantic similarity
2. sql_query - Query commit history, file changes, and repository metadata
3. static_analysis - Analyze code quality, complexity, and TODO comments

Think about what information you need to answer this question. Consider:
- What specific code or documentation should I search for?
- What historical data (commits, changes) would be relevant?
- What code quality metrics would help understand the codebase?

Plan your approach step by step.
"""
        
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error in thinking step: {str(e)}"
    
    def act(self, thought: str, question: str) -> ToolCall:
        """Act step: choose and execute a tool."""
        prompt = f"""
Based on your analysis, choose the most appropriate tool to use.

Thought: {thought}
Question: {question}

Available tools:
1. vector_search(query, top_k=5, file_filter=None) - Search for code/documentation
2. sql_query(sql, params=None) - Run SQL queries on repository data
3. static_analysis() - Get code quality metrics

Respond with JSON in this format:
{{
    "tool": "tool_name",
    "input": {{"param1": "value1", "param2": "value2"}}
}}

Choose the tool that will provide the most relevant information for the question.
"""
        
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            
            # Parse the response to extract tool call
            content = response.choices[0].message.content.strip()
            
            # Try to extract JSON from the response
            if "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
                
                try:
                    tool_call_data = json.loads(json_str)
                    return ToolCall(
                        tool=tool_call_data.get("tool", ""),
                        input=tool_call_data.get("input", {})
                    )
                except json.JSONDecodeError:
                    pass
            
            # Fallback: try to infer tool from content
            if "vector_search" in content.lower() or "search" in content.lower():
                return ToolCall(tool="vector_search", input={"query": question})
            elif "sql" in content.lower() or "commit" in content.lower():
                return ToolCall(tool="sql_query", input={"sql": "SELECT * FROM commits LIMIT 5"})
            elif "static" in content.lower() or "quality" in content.lower():
                return ToolCall(tool="static_analysis", input={})
            else:
                return ToolCall(tool="vector_search", input={"query": question})
                
        except Exception as e:
            return ToolCall(tool="vector_search", input={"query": question}, error=str(e))
    
    def execute_tool(self, tool_call: ToolCall) -> Dict[str, Any]:
        """Execute the chosen tool."""
        try:
            if tool_call.tool == "vector_search":
                query = tool_call.input.get("query", "")
                top_k = tool_call.input.get("top_k", 5)
                file_filter = tool_call.input.get("file_filter")
                
                results = self.vector_tool.search(query, top_k, file_filter)
                return {
                    "tool": "vector_search",
                    "results": results,
                    "count": len(results)
                }
            
            elif tool_call.tool == "sql_query":
                sql = tool_call.input.get("sql", "")
                params = tool_call.input.get("params")
                
                if not sql:
                    # Provide some default useful queries
                    sql = """
                    SELECT hash, author_name, commit_date, message, files_changed
                    FROM commits
                    ORDER BY commit_date DESC
                    LIMIT 10
                    """
                
                results = self.sql_loader.run_query(sql, params)
                return {
                    "tool": "sql_query",
                    "results": results,
                    "count": len(results)
                }
            
            elif tool_call.tool == "static_analysis":
                analysis_type = tool_call.input.get("type", "repository")
                
                if analysis_type == "repository":
                    results = self.static_analyzer.analyze_repository()
                else:
                    file_path = tool_call.input.get("file_path")
                    if file_path:
                        results = self.static_analyzer.analyze_file(file_path)
                    else:
                        results = self.static_analyzer.analyze_repository()
                
                return {
                    "tool": "static_analysis",
                    "results": results
                }
            
            else:
                return {
                    "error": f"Unknown tool: {tool_call.tool}"
                }
                
        except Exception as e:
            return {
                "error": f"Tool execution failed: {str(e)}"
            }
    
    def observe(self, tool_output: Dict[str, Any]) -> str:
        """Observe step: analyze the tool output."""
        prompt = f"""
Analyze the tool output and extract the most relevant information.

Tool Output: {json.dumps(tool_output, indent=2)}

What are the key insights from this data? What does it tell us about the codebase?
Focus on information that directly relates to the original question.
"""
        
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error in observation step: {str(e)}"
    
    def synthesize(self, question: str, trace: List[ReActStep]) -> str:
        """Synthesize the final answer from the trace."""
        # Extract all observations and tool outputs
        observations = []
        tool_outputs = []
        
        for step in trace:
            if step.step_type == "observe":
                observations.append(step.content)
            elif step.step_type == "act" and step.tool_call and step.tool_call.output:
                tool_outputs.append(step.tool_call.output)
        
        prompt = f"""
Synthesize a comprehensive answer to the question based on all the information gathered.

Question: {question}

Observations:
{chr(10).join(f"- {obs}" for obs in observations)}

Tool Outputs:
{json.dumps(tool_outputs, indent=2)}

Provide a clear, well-structured answer that:
1. Directly addresses the question
2. Cites specific files, line numbers, or commit hashes when relevant
3. Includes relevant code snippets or metrics
4. Is grounded in the actual data from the repository

Format your answer in Markdown with proper citations like [file.py:42] for line references.
"""
        
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error synthesizing answer: {str(e)}"
    
    def ask(self, question: str) -> Dict[str, Any]:
        """Main method to ask a question about the repository."""
        self.trace = []
        context = ""
        
        for step_num in range(self.max_steps):
            # Think
            thought = self.think(question, context)
            self.trace.append(ReActStep(
                step_type="think",
                content=thought
            ))
            
            # Act
            tool_call = self.act(thought, question)
            self.trace.append(ReActStep(
                step_type="act",
                content=f"Using tool: {tool_call.tool}",
                tool_call=tool_call
            ))
            
            # Execute tool
            tool_output = self.execute_tool(tool_call)
            self.trace[-1].tool_call.output = tool_output
            
            # Observe
            observation = self.observe(tool_output)
            self.trace.append(ReActStep(
                step_type="observe",
                content=observation
            ))
            
            # Update context for next iteration
            context += f"\nStep {step_num + 1}: {observation}"
            
            # Check if we have enough information
            if "error" not in tool_output and len(tool_output.get("results", [])) > 0:
                # If we got good results, we might have enough information
                if step_num >= 1:  # At least 2 steps
                    break
        
        # Synthesize final answer
        answer = self.synthesize(question, self.trace)
        
        # Extract citations from the trace
        citations = self._extract_citations(tool_output)
        
        return {
            "question": question,
            "answer": answer,
            "trace": [step.dict() for step in self.trace],
            "citations": citations,
            "steps_taken": len(self.trace)
        }
    
    def _extract_citations(self, tool_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract citations from tool outputs."""
        citations = []
        
        if tool_output.get("tool") == "vector_search":
            for result in tool_output.get("results", []):
                citations.append({
                    "file": result.get("file_path", ""),
                    "line": result.get("start_line", 0),
                    "content": result.get("content", "")[:100] + "..."
                })
        
        elif tool_output.get("tool") == "sql_query":
            for result in tool_output.get("results", []):
                if "hash" in result:
                    citations.append({
                        "type": "commit",
                        "hash": result.get("hash", "")[:8],
                        "author": result.get("author_name", ""),
                        "message": result.get("message", "")[:100] + "..."
                    })
        
        return citations


def ask(question: str, repo_path: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to ask a question about a repository."""
    agent = RepoMindAgent(repo_path, **kwargs)
    return agent.ask(question)


# FastAPI app for the REST API
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

app = FastAPI(title="RepoMind Agent API", version="0.1.0")
security = HTTPBearer()

class QuestionRequest(BaseModel):
    question: str
    repo_path: str

class QuestionResponse(BaseModel):
    question: str
    answer: str
    trace: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    steps_taken: int

# Simple token-based authentication
VALID_TOKENS = {"test-token", "your-secret-token"}  # In production, use proper auth

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials not in VALID_TOKENS:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest, token: str = Depends(verify_token)):
    """Ask a question about a repository."""
    try:
        result = ask(request.question, request.repo_path)
        return QuestionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 