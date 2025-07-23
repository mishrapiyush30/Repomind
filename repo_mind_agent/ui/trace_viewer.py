"""
Streamlit UI for RepoMind Agent trace viewer.

Provides an interactive interface for asking questions about repositories
and viewing the agent's reasoning process.
"""

import streamlit as st
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Add the parent directory to the path to import repo_mind_agent
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from repo_mind_agent.orchestrator import RepoMindAgent, ask


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="RepoMind Agent",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 RepoMind Agent")
    st.markdown("Powered by GPT-4 - Ask questions about your codebase")
    
    # Use default configuration
    repo_path = "."
    vector_db_url = "postgresql://localhost/repomind"
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    max_steps = 8
    
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # Main content area
    st.header("Ask a Question")
    
    # Question input
    question = st.text_area("Question", placeholder="Ask anything about the codebase...", height=100)
    
    # Submit button
    if st.button("Ask", type="primary"):
        if not question.strip():
            st.error("Please enter a question.")
            return
        
        if not repo_path or not Path(repo_path).exists():
            st.error("Please provide a valid repository path.")
            return
        
        if not openai_api_key:
            st.error("Please provide an OpenAI API key.")
            return
        
        # Show progress
        with st.spinner("Analyzing..."):
            try:
                # Initialize agent
                agent = RepoMindAgent(
                    repo_path=repo_path,
                    vector_db_url=vector_db_url if vector_db_url else None,
                    openai_api_key=openai_api_key
                )
                agent.max_steps = max_steps
                
                # Ask question
                result = agent.ask(question)
                
                # Store result in session state
                st.session_state.last_result = result
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return
    
    # Display results
    if "last_result" in st.session_state:
        result = st.session_state.last_result
        
        st.header("Answer")
        st.markdown(result["answer"])
        
        # Display citations
        if result["citations"]:
            st.header("Citations")
            for citation in result["citations"]:
                if "file_path" in citation:
                    line_ref = citation.get('start_line', citation.get('line', ''))
                    st.code(f"{citation['file_path']}:{line_ref}")
                elif "file" in citation:
                    st.code(f"{citation['file']}:{citation.get('line', '')}")
                elif "type" in citation and citation["type"] == "commit":
                    st.code(f"Commit {citation['hash']}")
                else:
                    st.code(str(citation))
        
        # Display ReAct trace (hidden by default)
        with st.expander("Show Implementation Details", expanded=False):
            st.header("ReAct Trace")
            
            for i, step in enumerate(result["trace"]):
                step_type = step["step_type"]
                
                if step_type == "think":
                    with st.expander(f"Think (Step {i+1})", expanded=False):
                        st.markdown(step["content"])
                
                elif step_type == "act":
                    with st.expander(f"Act (Step {i+1})", expanded=False):
                        if step.get("tool_call"):
                            tool_call = step["tool_call"]
                            st.markdown(f"**Tool:** {tool_call['tool']}")
                            st.json(tool_call['input'])
                            
                            if tool_call.get("output"):
                                st.markdown("**Output:**")
                                st.json(tool_call["output"])
                
                elif step_type == "observe":
                    with st.expander(f"Observe (Step {i+1})", expanded=False):
                        st.markdown(step["content"])


if __name__ == "__main__":
    main() 