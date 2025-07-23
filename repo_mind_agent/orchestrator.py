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
            self.openai_api_key = openai_api_key
        else:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass openai_api_key parameter.")
        
        # ReAct loop state
        self.max_steps = 8  # Increased to allow for more comprehensive analysis
        self.trace: List[ReActStep] = []
    
    def think(self, question: str, context: str = "") -> str:
        """Think step: analyze the question and plan the approach."""
        prompt = f"""
You are RepoMind Agent, an expert software architect and code analyst that provides deep insights into codebases.

Question: {question}
Context: {context}

Your role is to:
- Understand the architectural context and design patterns
- Identify key relationships and dependencies
- Analyze code quality and maintainability
- Provide strategic insights about the codebase
- Suggest improvements and areas for investigation

Available tools:
1. vector_search - Search for code and documentation using semantic similarity
   - Best for: Finding specific code patterns, locating files, understanding implementation details
   - Examples: "How is authentication implemented?", "Where is the main function?", "Find files related to database"
   - Important files to search for: pyproject.toml, requirements.txt (for dependencies), README.md (for project overview)

2. sql_query - Query commit history, file changes, and repository metadata
   - Best for: Historical analysis, change patterns, contributor insights
   - Examples: "Most changed files", "Recent commits", "Who contributed to this module?"
   - Use for dependency questions: Look at which files change together (suggesting dependencies)

3. static_analysis - Analyze code quality, complexity, and code structure
   - Best for: Code quality metrics, complexity analysis, finding issues
   - Examples: "Complex functions", "Code quality issues", "TODO items"
   - For dependency questions: Analyze import statements and module relationships

4. file_read - Read specific files directly from the filesystem
   - Best for: Reading configuration files, dependency files, or any specific file
   - Examples: "Read pyproject.toml", "Read requirements.txt", "Read README.md"
   - Use when vector_search doesn't find the file or you need the complete file content

Database Schema Information:
- commits: hash, author_name, author_email, commit_date, message, parent_hash, files_changed, insertions, deletions
- files: id, path, size, last_modified, last_commit_hash, file_type
- file_changes: id, commit_hash, file_path, change_type, insertions, deletions

Common SQL Queries:
- Most changed files: SELECT file_path, COUNT(*) as change_count FROM file_changes GROUP BY file_path ORDER BY change_count DESC LIMIT 10
- Recent commits: SELECT hash, author_name, commit_date, message FROM commits ORDER BY commit_date DESC LIMIT 10
- File history: SELECT c.hash, c.commit_date, c.message FROM commits c JOIN file_changes fc ON c.hash = fc.commit_hash WHERE fc.file_path = '?' ORDER BY c.commit_date DESC

IMPORTANT: For questions about dependencies or imports:
1. First use vector_search to find dependency files (pyproject.toml, requirements.txt, setup.py)
2. Then search for import statements in key Python files
3. Consider using static_analysis to analyze import patterns
4. You may need to combine multiple tools to get a complete picture

IMPORTANT: For questions about files:
1. If asking for file names/lists: Use vector_search to find relevant files
2. If asking to explain a specific file: Use file_read to get the complete file content
3. If asking about file structure: Use static_analysis to understand file organization
4. If asking about file history: Use sql_query to get commit history

Think about what information you need to answer this question. Consider:
- What specific code or documentation should I search for?
- What historical data (commits, changes) would be relevant?
- What code quality metrics would help understand the codebase?
- What combination of tools will give the most complete answer?

Plan your approach step by step. You should often use multiple tools in sequence to build a complete understanding.
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
Based on your analysis, choose the most appropriate tool to use for the next step in answering the question.

Thought: {thought}
Question: {question}

Available tools:
1. vector_search(query, top_k=5, file_filter=None) - Search for code/documentation
   - Use for finding specific files like pyproject.toml, requirements.txt, or code patterns
   - Excellent for locating dependency definitions and import statements
   - Example query: "pyproject.toml dependencies" or "import statements in main files"

2. sql_query(sql, params=None) - Run SQL queries on repository data
   - Use for historical analysis and change patterns
   - Can help identify which files change together (suggesting dependencies)
   - Can track evolution of dependencies over time

3. static_analysis(type="repository") - Get code quality metrics for the entire repository
   - Provides overview of code structure, complexity, and quality
   - Analyzes import patterns across the codebase

4. static_analysis(type="complexity", limit=10) - Get the most complex functions in the codebase
   - Identifies functions that might need refactoring
   - Often reveals core functionality with many dependencies

5. static_analysis(type="file", file_path="path/to/file.py") - Analyze a specific file
   - Use after finding relevant files with vector_search
   - Detailed analysis of imports and dependencies in a specific file

6. file_read(file_path="path/to/file") - Read a specific file directly
   - Use when vector_search doesn't find the file you need
   - Perfect for reading configuration files like pyproject.toml, requirements.txt
   - Use when you need the complete content of a specific file
   - Use when asked to "explain" or "show me" a specific file

Database Schema Information:
- commits: hash, author_name, author_email, commit_date, message, parent_hash, files_changed, insertions, deletions
- files: id, path, size, last_modified, last_commit_hash, file_type
- file_changes: id, commit_hash, file_path, change_type, insertions, deletions

Common SQL Queries:
- Most changed files: SELECT file_path, COUNT(*) as change_count FROM file_changes GROUP BY file_path ORDER BY change_count DESC LIMIT 10
- Recent commits: SELECT hash, author_name, commit_date, message FROM commits ORDER BY commit_date DESC LIMIT 10
- File history: SELECT c.hash, c.commit_date, c.message FROM commits c JOIN file_changes fc ON c.hash = fc.commit_hash WHERE fc.file_path = '?' ORDER BY c.commit_date DESC

IMPORTANT: For questions about dependencies or imports:
1. First use vector_search to find dependency files (pyproject.toml, requirements.txt, setup.py)
2. Then search for import statements in key Python files
3. Consider using static_analysis to analyze import patterns

IMPORTANT: For questions about files:
1. If asking for file names/lists: Use vector_search with queries like "Python files", "main files", "configuration files"
2. If asking to explain a specific file: Use file_read with the exact file path
3. If asking about file structure: Use static_analysis to understand organization
4. If asking about file history: Use sql_query to get commit history

Respond with JSON in this format:
{{
    "tool": "tool_name",
    "input": {{"param1": "value1", "param2": "value2"}}
}}

Choose the tool that will provide the most relevant information for the NEXT step in your reasoning process. 
You'll have multiple opportunities to use different tools, so focus on the most important information to gather first.
"""
        
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
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
                
                try:
                    results = self.sql_loader.run_query(sql, params)
                    return {
                        "tool": "sql_query",
                        "results": results,
                        "count": len(results)
                    }
                except Exception as e:
                    return {
                        "tool": "sql_query",
                        "error": str(e),
                        "results": []
                    }
            
            elif tool_call.tool == "static_analysis":
                analysis_type = tool_call.input.get("type", "repository")
                
                if analysis_type == "complexity":
                    # For complexity analysis, use the optimized method
                    limit = tool_call.input.get("limit", 10)
                    results = self.static_analyzer.get_most_complex_functions(limit)
                    return {
                        "tool": "static_analysis",
                        "analysis_type": "complexity",
                        "results": results
                    }
                elif analysis_type == "repository":
                    # For full repository analysis, limit the output size
                    results = self.static_analyzer.analyze_repository()
                    # Remove large data structures that might cause token limits
                    if "radon" in results and "complexity_data" in results["radon"]:
                        del results["radon"]["complexity_data"]
                    return {
                        "tool": "static_analysis",
                        "analysis_type": "repository",
                        "results": results
                    }
                else:
                    file_path = tool_call.input.get("file_path")
                    if file_path:
                        results = self.static_analyzer.analyze_file(file_path)
                        return {
                            "tool": "static_analysis",
                            "analysis_type": "file",
                            "results": results
                        }
                    else:
                        # Default to repository analysis with limited output
                        results = self.static_analyzer.analyze_repository()
                        if "radon" in results and "complexity_data" in results["radon"]:
                            del results["radon"]["complexity_data"]
                        return {
                            "tool": "static_analysis",
                            "analysis_type": "repository",
                            "results": results
                        }
            
            elif tool_call.tool == "file_read":
                # New tool for reading files directly
                file_path = tool_call.input.get("file_path", "")
                if not file_path:
                    return {
                        "tool": "file_read",
                        "error": "No file path provided",
                        "results": []
                    }
                
                try:
                    full_path = Path(self.repo_path) / file_path
                    if not full_path.exists():
                        return {
                            "tool": "file_read",
                            "error": f"File not found: {file_path}",
                            "results": []
                        }
                    
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    return {
                        "tool": "file_read",
                        "file_path": file_path,
                        "content": content,
                        "size": len(content)
                    }
                except Exception as e:
                    return {
                        "tool": "file_read",
                        "error": f"Error reading file: {str(e)}",
                        "results": []
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
        """Observe step: analyze the tool output with deep context understanding."""
        tool_type = tool_output.get("tool", "unknown")
        
        # Enhanced context-aware prompts
        if tool_type == "vector_search":
            prompt_addition = """
For dependency-related questions:
- Look for pyproject.toml, requirements.txt, or setup.py files in the results
- Identify import statements in Python files
- Note which modules are imported most frequently
- Consider what additional tools might provide more information

For file-related questions:
- Analyze file patterns and relationships
- Identify key files and their purposes
- Look for architectural patterns in file organization
- Consider what additional context would be helpful

For code understanding questions:
- Identify key functions, classes, and modules
- Look for design patterns and architectural decisions
- Note important code structures and relationships
- Consider what deeper analysis would reveal
"""
        elif tool_type == "sql_query":
            prompt_addition = """
For dependency-related questions:
- Look for patterns of files that change together (suggesting dependencies)
- Identify frequently modified files (often core dependencies)
- Consider what additional tools might provide more information

For historical analysis:
- Identify development patterns and trends
- Look for key contributors and their focus areas
- Analyze change frequency and impact
- Consider what historical context reveals about architecture

For file evolution questions:
- Track how files have evolved over time
- Identify major refactoring events
- Look for patterns in commit messages
- Consider what the history tells us about code quality
"""
        elif tool_type == "static_analysis":
            prompt_addition = """
For dependency-related questions:
- Note import patterns and external dependencies
- Identify complex functions that might have many dependencies
- Consider what additional tools might provide more information

For code quality analysis:
- Identify architectural strengths and weaknesses
- Look for code complexity patterns
- Note areas that might need refactoring
- Consider what the metrics reveal about maintainability

For structural analysis:
- Understand the overall code organization
- Identify key architectural decisions
- Look for design patterns and anti-patterns
- Consider what the structure reveals about the project's purpose
"""
        elif tool_type == "file_read":
            prompt_addition = """
For file content analysis:
- Understand the file's purpose and role in the codebase
- Identify key functions, classes, and their relationships
- Look for design patterns and architectural decisions
- Note important code structures and their significance
- Consider what this file reveals about the overall system design
- Analyze the code quality and maintainability aspects
- Identify potential improvements or areas of concern
"""
        else:
            prompt_addition = ""
            
        prompt = f"""
You are an expert code analyst. Deeply analyze this tool output and provide comprehensive insights.

Tool Output: {json.dumps(tool_output, indent=2)}

Provide a detailed analysis that includes:

1. **Key Insights**: What are the most important findings from this data?
2. **Context Understanding**: How does this information fit into the broader codebase?
3. **Pattern Recognition**: What patterns, relationships, or architectural decisions do you observe?
4. **Quality Assessment**: What does this reveal about code quality, maintainability, or design?
5. **Strategic Implications**: What does this tell us about the project's architecture and development approach?
6. **Next Steps**: What additional information would provide deeper understanding?

Focus on providing expert-level analysis that goes beyond surface-level observations.
{prompt_addition}

Be thorough and analytical in your response. Consider both technical details and broader architectural implications.
"""
        
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
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
        
        # Check if this is a dependency-related question
        is_dependency_question = any(keyword in question.lower() for keyword in ["dependency", "dependencies", "import", "imports", "package", "packages", "library", "libraries"])
        
        # Check if this is a file-related question
        is_file_question = any(keyword in question.lower() for keyword in ["file", "files", "explain", "show", "read", "content", "what's in"])
        
        dependency_guidance = """
For dependency-related questions, make sure your answer includes:
1. External dependencies from package files (pyproject.toml, requirements.txt)
2. Key internal module dependencies and import patterns
3. Categorization of dependencies by purpose (core, testing, UI, etc.)
4. Version information where available
5. Distinction between direct and transitive dependencies

Structure your answer with clear sections:
- Main External Dependencies
- Internal Module Dependencies
- Development/Testing Dependencies
- Optional Dependencies
"""

        file_guidance = """
For file-related questions, make sure your answer includes:
1. If listing files: Provide a clear list of relevant files with brief descriptions
2. If explaining a file: Provide a comprehensive explanation of the file's purpose, structure, and key components
3. If analyzing file content: Break down the code structure, functions, classes, and important patterns
4. Include relevant code snippets and line references where appropriate

Structure your answer with clear sections:
- File Overview
- Key Components
- Important Functions/Classes
- Code Structure
- Usage Examples (if applicable)
"""

                prompt = f"""
You are an expert software architect and code analyst. Synthesize a comprehensive, insightful answer based on all the information gathered.

Question: {question}

Observations:
{chr(10).join(f"- {obs}" for obs in observations)}

Tool Outputs:
{json.dumps(tool_outputs, indent=2)}

{dependency_guidance if is_dependency_question else ""}
{file_guidance if is_file_question else ""}

Provide a comprehensive, expert-level analysis that includes:

1. **Direct Answer**: Clearly address the question with specific, actionable insights
2. **Deep Analysis**: Go beyond surface-level observations to provide architectural and design insights
3. **Context Understanding**: Explain how the findings relate to the broader codebase and project goals
4. **Pattern Recognition**: Identify important patterns, relationships, and architectural decisions
5. **Quality Assessment**: Evaluate code quality, maintainability, and design choices
6. **Strategic Insights**: Provide insights about the project's architecture and development approach
7. **Actionable Recommendations**: Suggest improvements or areas for further investigation

Your response should demonstrate expert-level understanding of:
- Software architecture and design patterns
- Code quality and maintainability principles
- Development practices and methodologies
- Technical debt and refactoring opportunities
- Performance and scalability considerations

Format your answer in Markdown with proper citations like [file.py:42] for line references.
Include relevant code snippets, metrics, and specific examples to support your analysis.

Be thorough, analytical, and provide insights that would be valuable to both developers and architects.
"""
        
        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
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
                elif "file_path" in result and "change_count" in result:
                    # Special case for most changed files
                    citations.append({
                        "type": "file_change",
                        "file": result.get("file_path", ""),
                        "changes": result.get("change_count", 0),
                        "details": f"Insertions: {result.get('total_insertions', 0)}, Deletions: {result.get('total_deletions', 0)}"
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