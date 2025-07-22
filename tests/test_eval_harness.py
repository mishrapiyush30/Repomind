"""
Evaluation harness for RepoMind Agent.

Tests the agent against a golden Q&A dataset and computes various metrics
including exact match, groundedness, and response time.
"""

import yaml
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
import pytest
from unittest.mock import patch, MagicMock
import difflib

from repo_mind_agent.orchestrator import RepoMindAgent, ask


class EvaluationHarness:
    """Evaluation harness for testing RepoMind Agent."""
    
    def __init__(self, golden_qa_path: str = "golden_qas.yaml"):
        """Initialize the evaluation harness."""
        self.golden_qa_path = golden_qa_path
        self.load_golden_qa()
    
    def load_golden_qa(self):
        """Load the golden Q&A dataset."""
        with open(self.golden_qa_path, 'r') as f:
            self.golden_qa = yaml.safe_load(f)
        
        self.questions = self.golden_qa['evaluation_questions']
        self.metrics = self.golden_qa['evaluation_metrics']
        self.rubric = self.golden_qa['scoring_rubric']
    
    def exact_match_score(self, predicted: str, expected: str) -> float:
        """Calculate exact match score between predicted and expected answers."""
        # Normalize text for comparison
        predicted_norm = self._normalize_text(predicted)
        expected_norm = self._normalize_text(expected)
        
        # Use difflib for sequence matching
        matcher = difflib.SequenceMatcher(None, expected_norm, predicted_norm)
        return matcher.ratio()
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common punctuation
        import re
        text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def groundedness_score(self, answer: str, citations: List[Dict[str, Any]]) -> int:
        """Calculate groundedness score based on citations and answer quality."""
        score = 1  # Start with minimum score
        
        # Check for citations
        if citations and len(citations) > 0:
            score += 1
            
            # Bonus for multiple citations
            if len(citations) >= 3:
                score += 0.5
        
        # Check for file references
        if any('file' in citation for citation in citations):
            score += 1
        
        # Check for line number references
        if any('line' in citation for citation in citations):
            score += 1
        
        # Check for commit references
        if any('hash' in citation for citation in citations):
            score += 0.5
        
        # Check answer quality indicators
        if '[' in answer and ']' in answer:  # Likely has citations
            score += 0.5
        
        # Check for specific code patterns
        if any(pattern in answer.lower() for pattern in ['function', 'class', 'method', 'def ', 'class ']):
            score += 0.5
        
        # Check for file path patterns
        if '/' in answer or '\\' in answer:
            score += 0.5
        
        # Penalty for generic responses
        generic_phrases = ['i don\'t know', 'cannot find', 'not available', 'no information']
        if any(phrase in answer.lower() for phrase in generic_phrases):
            score = max(1, score - 1)
        
        return min(int(score), 5)  # Cap at 5 and convert to int
    
    def relevance_score(self, question: str, answer: str) -> int:
        """Calculate relevance score based on question-answer alignment."""
        # Simple keyword matching for now
        question_keywords = set(self._normalize_text(question).split())
        answer_keywords = set(self._normalize_text(answer).split())
        
        if not question_keywords:
            return 1
        
        overlap = len(question_keywords.intersection(answer_keywords))
        coverage = overlap / len(question_keywords)
        
        if coverage >= 0.8:
            return 5
        elif coverage >= 0.6:
            return 4
        elif coverage >= 0.4:
            return 3
        elif coverage >= 0.2:
            return 2
        else:
            return 1
    
    def completeness_score(self, expected: str, answer: str) -> int:
        """Calculate completeness score based on expected vs actual answer."""
        expected_keywords = set(self._normalize_text(expected).split())
        answer_keywords = set(self._normalize_text(answer).split())
        
        if not expected_keywords:
            return 1
        
        coverage = len(answer_keywords.intersection(expected_keywords)) / len(expected_keywords)
        
        if coverage >= 0.8:
            return 5
        elif coverage >= 0.6:
            return 4
        elif coverage >= 0.4:
            return 3
        elif coverage >= 0.2:
            return 2
        else:
            return 1
    
    def evaluate_single_question(self, question_data: Dict[str, Any], 
                               agent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single question-answer pair."""
        question = question_data['question']
        expected = question_data['expected_answer']
        predicted = agent_result['answer']
        citations = agent_result.get('citations', [])
        
        # Calculate metrics
        exact_match = self.exact_match_score(predicted, expected)
        groundedness = self.groundedness_score(predicted, citations)
        relevance = self.relevance_score(question, predicted)
        completeness = self.completeness_score(expected, predicted)
        
        # Check response time
        response_time = agent_result.get('response_time', 0)
        time_ok = response_time <= self.metrics['max_response_time']
        
        # Check citation count
        citation_count = len(citations)
        citations_ok = (self.metrics['min_citations'] <= citation_count <= self.metrics['max_citations'])
        
        return {
            'question': question,
            'category': question_data.get('category', 'unknown'),
            'difficulty': question_data.get('difficulty', 'unknown'),
            'exact_match': exact_match,
            'groundedness': groundedness,
            'relevance': relevance,
            'completeness': completeness,
            'response_time': response_time,
            'citation_count': citation_count,
            'time_ok': time_ok,
            'citations_ok': citations_ok,
            'overall_score': (exact_match + groundedness + relevance + completeness) / 4
        }
    
    def evaluate_agent(self, repo_path: str, agent_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate the agent on all questions in the golden dataset."""
        if agent_kwargs is None:
            agent_kwargs = {}
        
        results = []
        total_response_time = 0
        
        for question_data in self.questions:
            question = question_data['question']
            
            print(f"Evaluating: {question}")
            
            # Time the response
            start_time = time.time()
            
            try:
                agent_result = ask(question, repo_path, **agent_kwargs)
                response_time = time.time() - start_time
                agent_result['response_time'] = response_time
                
                # Evaluate the result
                eval_result = self.evaluate_single_question(question_data, agent_result)
                results.append(eval_result)
                
                total_response_time += response_time
                
                print(f"  Response time: {response_time:.2f}s")
                print(f"  Overall score: {eval_result['overall_score']:.2f}")
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                # Add failed result
                results.append({
                    'question': question,
                    'category': question_data.get('category', 'unknown'),
                    'difficulty': question_data.get('difficulty', 'unknown'),
                    'exact_match': 0,
                    'groundedness': 1,
                    'relevance': 1,
                    'completeness': 1,
                    'response_time': time.time() - start_time,
                    'citation_count': 0,
                    'time_ok': False,
                    'citations_ok': False,
                    'overall_score': 0,
                    'error': str(e)
                })
        
        # Calculate aggregate metrics
        return self._calculate_aggregate_metrics(results, total_response_time)
    
    def _calculate_aggregate_metrics(self, results: List[Dict[str, Any]], 
                                   total_response_time: float) -> Dict[str, Any]:
        """Calculate aggregate metrics from individual results."""
        if not results:
            return {}
        
        # Filter out failed results
        successful_results = [r for r in results if 'error' not in r]
        failed_count = len(results) - len(successful_results)
        
        if not successful_results:
            return {
                'total_questions': len(results),
                'successful_questions': 0,
                'failed_questions': failed_count,
                'success_rate': 0.0,
                'average_scores': {},
                'category_scores': {},
                'difficulty_scores': {},
                'performance_metrics': {}
            }
        
        # Calculate average scores
        avg_exact_match = sum(r['exact_match'] for r in successful_results) / len(successful_results)
        avg_groundedness = sum(r['groundedness'] for r in successful_results) / len(successful_results)
        avg_relevance = sum(r['relevance'] for r in successful_results) / len(successful_results)
        avg_completeness = sum(r['completeness'] for r in successful_results) / len(successful_results)
        avg_overall = sum(r['overall_score'] for r in successful_results) / len(successful_results)
        
        # Calculate category scores
        category_scores = {}
        for category in set(r['category'] for r in successful_results):
            category_results = [r for r in successful_results if r['category'] == category]
            if category_results:
                category_scores[category] = {
                    'count': len(category_results),
                    'avg_score': sum(r['overall_score'] for r in category_results) / len(category_results)
                }
        
        # Calculate difficulty scores
        difficulty_scores = {}
        for difficulty in set(r['difficulty'] for r in successful_results):
            difficulty_results = [r for r in successful_results if r['difficulty'] == difficulty]
            if difficulty_results:
                difficulty_scores[difficulty] = {
                    'count': len(difficulty_results),
                    'avg_score': sum(r['overall_score'] for r in difficulty_results) / len(difficulty_results)
                }
        
        # Performance metrics
        avg_response_time = total_response_time / len(successful_results)
        time_compliance = sum(1 for r in successful_results if r['time_ok']) / len(successful_results)
        citation_compliance = sum(1 for r in successful_results if r['citations_ok']) / len(successful_results)
        
        return {
            'total_questions': len(results),
            'successful_questions': len(successful_results),
            'failed_questions': failed_count,
            'success_rate': len(successful_results) / len(results),
            'average_scores': {
                'exact_match': avg_exact_match,
                'groundedness': avg_groundedness,
                'relevance': avg_relevance,
                'completeness': avg_completeness,
                'overall': avg_overall
            },
            'category_scores': category_scores,
            'difficulty_scores': difficulty_scores,
            'performance_metrics': {
                'avg_response_time': avg_response_time,
                'time_compliance_rate': time_compliance,
                'citation_compliance_rate': citation_compliance
            },
            'detailed_results': results
        }


class TestEvaluationHarness:
    """Tests for the evaluation harness."""
    
    def test_exact_match_score(self):
        """Test exact match score calculation."""
        harness = EvaluationHarness()
        
        # Test identical strings
        score = harness.exact_match_score("Hello world", "Hello world")
        assert score == 1.0
        
        # Test similar strings
        score = harness.exact_match_score("Hello world", "Hello world!")
        assert score > 0.8
        
        # Test different strings
        score = harness.exact_match_score("Hello world", "Goodbye world")
        assert score < 0.6
    
    def test_groundedness_score(self):
        """Test groundedness score calculation."""
        harness = EvaluationHarness()
        
        # Test with citations
        citations = [{"file": "test.py", "line": 42}]
        score = harness.groundedness_score("Answer with citations", citations)
        assert score >= 3
        
        # Test without citations
        score = harness.groundedness_score("Answer without citations", [])
        assert score == 1
    
    def test_relevance_score(self):
        """Test relevance score calculation."""
        harness = EvaluationHarness()
        
        # Test relevant answer
        score = harness.relevance_score("Where is main function?", "The main function is in main.py")
        assert score >= 4
        
        # Test irrelevant answer
        score = harness.relevance_score("Where is main function?", "The weather is sunny today")
        assert score <= 2
    
    def test_completeness_score(self):
        """Test completeness score calculation."""
        harness = EvaluationHarness()
        
        expected = "The answer should include file path, line number, and function signature"
        complete_answer = "The main function is in main.py at line 42 with signature def main():"
        incomplete_answer = "The main function exists"
        
        complete_score = harness.completeness_score(expected, complete_answer)
        incomplete_score = harness.completeness_score(expected, incomplete_answer)
        
        assert complete_score > incomplete_score
    
    @patch('repo_mind_agent.orchestrator.RepoMindAgent')
    def test_evaluate_single_question(self, mock_agent_class):
        """Test evaluation of a single question."""
        harness = EvaluationHarness()
        
        # Mock agent result
        mock_agent = MagicMock()
        mock_agent.ask.return_value = {
            'answer': 'The main function is in main.py at line 42',
            'citations': [{'file': 'main.py', 'line': 42}],
            'response_time': 2.5
        }
        mock_agent_class.return_value = mock_agent
        
        question_data = {
            'question': 'Where is the main function?',
            'expected_answer': 'The main function should be found in main.py',
            'category': 'code_location',
            'difficulty': 'easy'
        }
        
        agent_result = {
            'answer': 'The main function is in main.py at line 42',
            'citations': [{'file': 'main.py', 'line': 42}],
            'response_time': 2.5
        }
        
        result = harness.evaluate_single_question(question_data, agent_result)
        
        assert 'exact_match' in result
        assert 'groundedness' in result
        assert 'relevance' in result
        assert 'completeness' in result
        assert result['response_time'] == 2.5
        assert result['citation_count'] == 1


def run_evaluation(repo_path: str, output_file: Optional[str] = None) -> Dict[str, Any]:
    """Run the full evaluation and optionally save results."""
    harness = EvaluationHarness()
    
    print("Starting RepoMind Agent evaluation...")
    print(f"Repository: {repo_path}")
    print(f"Questions: {len(harness.questions)}")
    print("-" * 50)
    
    results = harness.evaluate_agent(repo_path)
    
    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total Questions: {results['total_questions']}")
    print(f"Successful: {results['successful_questions']}")
    print(f"Failed: {results['failed_questions']}")
    print(f"Success Rate: {results['success_rate']:.2%}")
    
    print("\nAverage Scores:")
    for metric, score in results['average_scores'].items():
        print(f"  {metric}: {score:.3f}")
    
    print("\nPerformance Metrics:")
    for metric, value in results['performance_metrics'].items():
        print(f"  {metric}: {value:.3f}")
    
    print("\nCategory Scores:")
    for category, data in results['category_scores'].items():
        print(f"  {category}: {data['avg_score']:.3f} ({data['count']} questions)")
    
    print("\nDifficulty Scores:")
    for difficulty, data in results['difficulty_scores'].items():
        print(f"  {difficulty}: {data['avg_score']:.3f} ({data['count']} questions)")
    
    # Save results if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_eval_harness.py <repo_path> [output_file]")
        sys.exit(1)
    
    repo_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    results = run_evaluation(repo_path, output_file) 