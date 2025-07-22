"""
Streamlit UI for RepoMind Agent trace viewer.

Provides an interactive interface for asking questions about repositories
and viewing the agent's reasoning process.
"""

import streamlit as st
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Add the parent directory to the path to import repo_mind_agent
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from repo_mind_agent.orchestrator import RepoMindAgent, ask


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="RepoMind Agent",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧠 RepoMind Agent v0.1")
    st.markdown("An intelligent agent that provides instant understanding of codebases")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Repository path input
        repo_path = st.text_input(
            "Repository Path",
            value=".",
            help="Path to the repository to analyze"
        )
        
        # Database configuration
        st.subheader("Database Settings")
        db_path = st.text_input("SQLite DB Path", value="repo_data.db")
        vector_db_url = st.text_input(
            "Vector DB URL (PostgreSQL)", 
            value="postgresql://localhost/repomind",
            help="Leave empty to use default"
        )
        
        # OpenAI API key
        st.subheader("OpenAI Configuration")
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Your OpenAI API key for GPT-3.5-turbo"
        )
        
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Agent settings
        st.subheader("Agent Settings")
        max_steps = st.slider("Max ReAct Steps", min_value=1, max_value=10, value=5)
        
        # Sample questions
        st.subheader("Sample Questions")
        sample_questions = [
            "Where is the main function defined?",
            "What are the most complex functions in the codebase?",
            "Who are the main contributors to this repository?",
            "What are the most frequently changed files?",
            "Are there any TODO comments that need attention?",
            "What is the overall code quality of this repository?",
            "How is error handling implemented?",
            "What are the main dependencies and imports?",
        ]
        
        selected_sample = st.selectbox("Choose a sample question:", [""] + sample_questions)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Ask a Question")
        
        # Question input
        if selected_sample:
            question = st.text_area("Question", value=selected_sample, height=100)
        else:
            question = st.text_area("Question", placeholder="Ask anything about the codebase...", height=100)
        
        # Submit button
        if st.button("🚀 Ask RepoMind", type="primary"):
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
            with st.spinner("RepoMind is thinking..."):
                try:
                    # Initialize agent
                    agent = RepoMindAgent(
                        repo_path=repo_path,
                        db_path=db_path,
                        vector_db_url=vector_db_url if vector_db_url else None,
                        openai_api_key=openai_api_key
                    )
                    agent.max_steps = max_steps
                    
                    # Ask question
                    result = agent.ask(question)
                    
                    # Store result in session state
                    st.session_state.last_result = result
                    st.session_state.show_trace = True
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    return
    
    with col2:
        st.header("Quick Stats")
        
        if "last_result" in st.session_state:
            result = st.session_state.last_result
            
            # Display metrics
            st.metric("Steps Taken", result["steps_taken"])
            st.metric("Citations Found", len(result["citations"]))
            
            # Show trace toggle
            if st.checkbox("Show ReAct Trace", value=st.session_state.get("show_trace", False)):
                st.session_state.show_trace = True
            else:
                st.session_state.show_trace = False
    
    # Display results
    if "last_result" in st.session_state and st.session_state.get("show_trace", False):
        result = st.session_state.last_result
        
        st.header("🤖 Agent Response")
        
        # Display the answer
        st.markdown("### Answer")
        st.markdown(result["answer"])
        
        # Display citations
        if result["citations"]:
            st.markdown("### Citations")
            for i, citation in enumerate(result["citations"]):
                if "file" in citation:
                    st.code(f"{citation['file']}:{citation['line']} - {citation['content']}")
                elif "type" in citation and citation["type"] == "commit":
                    st.code(f"Commit {citation['hash']} by {citation['author']} - {citation['message']}")
        
        # Display ReAct trace
        st.header("🔍 ReAct Trace")
        
        for i, step in enumerate(result["trace"]):
            step_type = step["step_type"]
            
            if step_type == "think":
                with st.expander(f"🧠 Think (Step {i+1})", expanded=True):
                    st.markdown(step["content"])
            
            elif step_type == "act":
                with st.expander(f"⚡ Act (Step {i+1})", expanded=True):
                    st.markdown(f"**Tool:** {step['content']}")
                    
                    if step.get("tool_call"):
                        tool_call = step["tool_call"]
                        st.markdown(f"**Tool:** {tool_call['tool']}")
                        st.json(tool_call['input'])
                        
                        if tool_call.get("output"):
                            st.markdown("**Output:**")
                            st.json(tool_call["output"])
            
            elif step_type == "observe":
                with st.expander(f"👁️ Observe (Step {i+1})", expanded=True):
                    st.markdown(step["content"])
        
        # Visualization
        st.header("📊 Analysis Visualization")
        
        # Create trace visualization
        trace_data = result["trace"]
        step_types = [step["step_type"] for step in trace_data]
        
        # Count step types
        step_counts = {}
        for step_type in step_types:
            step_counts[step_type] = step_counts.get(step_type, 0) + 1
        
        # Create pie chart
        if step_counts:
            fig = px.pie(
                values=list(step_counts.values()),
                names=list(step_counts.keys()),
                title="ReAct Step Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Tool usage analysis
        tool_usage = {}
        for step in trace_data:
            if step["step_type"] == "act" and step.get("tool_call"):
                tool = step["tool_call"]["tool"]
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
        
        if tool_usage:
            fig = px.bar(
                x=list(tool_usage.keys()),
                y=list(tool_usage.values()),
                title="Tool Usage",
                labels={"x": "Tool", "y": "Usage Count"}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Repository information
    if repo_path and Path(repo_path).exists():
        st.header("📁 Repository Information")
        
        repo_path_obj = Path(repo_path)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # File count
            try:
                python_files = list(repo_path_obj.rglob("*.py"))
                st.metric("Python Files", len(python_files))
            except:
                st.metric("Python Files", "N/A")
        
        with col2:
            # Directory count
            try:
                directories = [d for d in repo_path_obj.iterdir() if d.is_dir()]
                st.metric("Directories", len(directories))
            except:
                st.metric("Directories", "N/A")
        
        with col3:
            # Repository size
            try:
                total_size = sum(f.stat().st_size for f in repo_path_obj.rglob('*') if f.is_file())
                st.metric("Total Size", f"{total_size / 1024:.1f} KB")
            except:
                st.metric("Total Size", "N/A")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        **RepoMind Agent v0.1** - Making codebases instantly understandable.
        
        Built with ❤️ using Streamlit, FastAPI, and OpenAI GPT-3.5-turbo.
        """
    )


if __name__ == "__main__":
    main() 