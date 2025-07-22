"""
Static analysis tool for code quality and complexity metrics.

Uses ruff for linting and radon for cyclomatic complexity analysis.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
import re


class StaticAnalyzer:
    """Static analysis tool using ruff and radon."""
    
    def __init__(self, repo_path: str):
        """Initialize the static analyzer with repository path."""
        self.repo_path = Path(repo_path)
    
    def run_ruff_analysis(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Run ruff analysis for linting and code quality checks."""
        try:
            cmd = ["ruff", "check", "--select", "C,F", "--output-format", "json"]
            
            if file_path:
                cmd.append(str(self.repo_path / file_path))
            else:
                cmd.append(str(self.repo_path))
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.repo_path,
                timeout=60
            )
            
            if result.returncode == 0:
                # No issues found
                return {
                    "success": True,
                    "issues": [],
                    "summary": {
                        "total_issues": 0,
                        "error_count": 0,
                        "warning_count": 0
                    }
                }
            else:
                # Parse JSON output
                try:
                    issues = json.loads(result.stdout)
                    error_count = sum(1 for issue in issues if issue.get("code", "").startswith("E"))
                    warning_count = sum(1 for issue in issues if issue.get("code", "").startswith("W"))
                    
                    return {
                        "success": True,
                        "issues": issues,
                        "summary": {
                            "total_issues": len(issues),
                            "error_count": error_count,
                            "warning_count": warning_count
                        }
                    }
                except json.JSONDecodeError:
                    return {
                        "success": False,
                        "error": "Failed to parse ruff output",
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }
                    
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Ruff analysis timed out"
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "Ruff not found. Please install ruff: pip install ruff"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Ruff analysis failed: {str(e)}"
            }
    
    def run_radon_analysis(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Run radon analysis for cyclomatic complexity."""
        try:
            cmd = ["radon", "cc", "--json"]
            
            if file_path:
                cmd.append(str(self.repo_path / file_path))
            else:
                cmd.append(str(self.repo_path))
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.repo_path,
                timeout=60
            )
            
            if result.returncode == 0:
                try:
                    complexity_data = json.loads(result.stdout)
                    
                    # Calculate summary statistics
                    total_functions = 0
                    total_complexity = 0
                    high_complexity_functions = 0
                    
                    for file_data in complexity_data.values():
                        for function_data in file_data:
                            complexity = function_data.get("complexity", 0)
                            total_functions += 1
                            total_complexity += complexity
                            if complexity > 10:  # High complexity threshold
                                high_complexity_functions += 1
                    
                    avg_complexity = total_complexity / total_functions if total_functions > 0 else 0
                    
                    return {
                        "success": True,
                        "complexity_data": complexity_data,
                        "summary": {
                            "total_functions": total_functions,
                            "total_complexity": total_complexity,
                            "average_complexity": round(avg_complexity, 2),
                            "high_complexity_functions": high_complexity_functions
                        }
                    }
                except json.JSONDecodeError:
                    return {
                        "success": False,
                        "error": "Failed to parse radon output",
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }
            else:
                return {
                    "success": False,
                    "error": f"Radon analysis failed: {result.stderr}"
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Radon analysis timed out"
            }
        except FileNotFoundError:
            return {
                "success": False,
                "error": "Radon not found. Please install radon: pip install radon"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Radon analysis failed: {str(e)}"
            }
    
    def count_todos(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Count TODO comments in the codebase."""
        try:
            search_path = self.repo_path / file_path if file_path else self.repo_path
            
            if file_path and search_path.is_file():
                files_to_search = [search_path]
            else:
                # Find all Python files
                files_to_search = list(search_path.rglob("*.py"))
            
            todo_patterns = [
                r"#\s*TODO[:\s]*(.+)",
                r"#\s*FIXME[:\s]*(.+)",
                r"#\s*XXX[:\s]*(.+)",
                r"#\s*HACK[:\s]*(.+)",
                r"#\s*BUG[:\s]*(.+)"
            ]
            
            todos = []
            total_count = 0
            
            for file_path in files_to_search:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    file_todos = []
                    for line_num, line in enumerate(content.split('\n'), 1):
                        for pattern in todo_patterns:
                            match = re.search(pattern, line, re.IGNORECASE)
                            if match:
                                file_todos.append({
                                    "line": line_num,
                                    "type": pattern.split()[1].replace(r"[:\s]*", "").replace("(.+)", ""),
                                    "message": match.group(1).strip() if match.group(1) else "",
                                    "full_line": line.strip()
                                })
                                total_count += 1
                    
                    if file_todos:
                        todos.append({
                            "file": str(file_path.relative_to(self.repo_path)),
                            "todos": file_todos,
                            "count": len(file_todos)
                        })
                        
                except (UnicodeDecodeError, IOError):
                    # Skip files with encoding issues
                    continue
            
            return {
                "success": True,
                "todos": todos,
                "summary": {
                    "total_todos": total_count,
                    "files_with_todos": len(todos)
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"TODO analysis failed: {str(e)}"
            }
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single file for all metrics."""
        file_path = str(file_path)
        
        ruff_result = self.run_ruff_analysis(file_path)
        radon_result = self.run_radon_analysis(file_path)
        todo_result = self.count_todos(file_path)
        
        return {
            "file_path": file_path,
            "ruff": ruff_result,
            "radon": radon_result,
            "todos": todo_result,
            "overall_health": self._calculate_health_score(ruff_result, radon_result, todo_result)
        }
    
    def analyze_repository(self) -> Dict[str, Any]:
        """Analyze the entire repository."""
        ruff_result = self.run_ruff_analysis()
        radon_result = self.run_radon_analysis()
        todo_result = self.count_todos()
        
        # Get file statistics
        python_files = list(self.repo_path.rglob("*.py"))
        total_lines = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    total_lines += len(f.readlines())
            except (UnicodeDecodeError, IOError):
                continue
        
        return {
            "repository": str(self.repo_path),
            "ruff": ruff_result,
            "radon": radon_result,
            "todos": todo_result,
            "statistics": {
                "total_python_files": len(python_files),
                "total_lines_of_code": total_lines
            },
            "overall_health": self._calculate_health_score(ruff_result, radon_result, todo_result)
        }
    
    def _calculate_health_score(self, ruff_result: Dict, radon_result: Dict, todo_result: Dict) -> Dict[str, Any]:
        """Calculate an overall health score for the codebase."""
        score = 100
        issues = []
        
        # Ruff score (deduct points for issues)
        if ruff_result.get("success"):
            ruff_summary = ruff_result.get("summary", {})
            error_count = ruff_summary.get("error_count", 0)
            warning_count = ruff_summary.get("warning_count", 0)
            
            score -= error_count * 5  # 5 points per error
            score -= warning_count * 1  # 1 point per warning
            
            if error_count > 0:
                issues.append(f"{error_count} linting errors")
            if warning_count > 0:
                issues.append(f"{warning_count} linting warnings")
        else:
            score -= 10
            issues.append("Ruff analysis failed")
        
        # Radon score (deduct points for high complexity)
        if radon_result.get("success"):
            radon_summary = radon_result.get("summary", {})
            high_complexity = radon_summary.get("high_complexity_functions", 0)
            avg_complexity = radon_summary.get("average_complexity", 0)
            
            score -= high_complexity * 3  # 3 points per high complexity function
            if avg_complexity > 10:
                score -= 10
                issues.append(f"High average complexity: {avg_complexity}")
            
            if high_complexity > 0:
                issues.append(f"{high_complexity} high complexity functions")
        else:
            score -= 10
            issues.append("Radon analysis failed")
        
        # TODO score (deduct points for too many TODOs)
        if todo_result.get("success"):
            total_todos = todo_result.get("summary", {}).get("total_todos", 0)
            if total_todos > 50:
                score -= 10
                issues.append(f"Too many TODOs: {total_todos}")
            elif total_todos > 20:
                score -= 5
                issues.append(f"Many TODOs: {total_todos}")
        else:
            score -= 5
            issues.append("TODO analysis failed")
        
        # Ensure score doesn't go below 0
        score = max(0, score)
        
        # Determine grade
        if score >= 90:
            grade = "A"
        elif score >= 80:
            grade = "B"
        elif score >= 70:
            grade = "C"
        elif score >= 60:
            grade = "D"
        else:
            grade = "F"
        
        return {
            "score": score,
            "grade": grade,
            "issues": issues,
            "max_score": 100
        }
    
    def get_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Ruff recommendations
        ruff_result = analysis_result.get("ruff", {})
        if ruff_result.get("success"):
            ruff_summary = ruff_result.get("summary", {})
            error_count = ruff_summary.get("error_count", 0)
            warning_count = ruff_summary.get("warning_count", 0)
            
            if error_count > 0:
                recommendations.append(f"Fix {error_count} linting errors to improve code quality")
            if warning_count > 0:
                recommendations.append(f"Address {warning_count} linting warnings")
        else:
            recommendations.append("Install and configure ruff for automated code quality checks")
        
        # Radon recommendations
        radon_result = analysis_result.get("radon", {})
        if radon_result.get("success"):
            radon_summary = radon_result.get("summary", {})
            high_complexity = radon_summary.get("high_complexity_functions", 0)
            avg_complexity = radon_summary.get("average_complexity", 0)
            
            if high_complexity > 0:
                recommendations.append(f"Refactor {high_complexity} functions with high cyclomatic complexity")
            if avg_complexity > 10:
                recommendations.append("Consider breaking down complex functions into smaller, more manageable pieces")
        else:
            recommendations.append("Install radon for cyclomatic complexity analysis")
        
        # TODO recommendations
        todo_result = analysis_result.get("todos", {})
        if todo_result.get("success"):
            total_todos = todo_result.get("summary", {}).get("total_todos", 0)
            if total_todos > 20:
                recommendations.append(f"Address {total_todos} TODO comments to improve code completeness")
        else:
            recommendations.append("Review and address TODO comments in the codebase")
        
        return recommendations


def analyze_repository_static(repo_path: str) -> Dict[str, Any]:
    """Convenience function to analyze a repository."""
    analyzer = StaticAnalyzer(repo_path)
    return analyzer.analyze_repository()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python static_analysis.py <repo_path>")
        sys.exit(1)
    
    repo_path = sys.argv[1]
    
    # Analyze repository
    result = analyze_repository_static(repo_path)
    
    # Print results
    print("=== Static Analysis Results ===")
    print(f"Repository: {result['repository']}")
    print(f"Overall Health: {result['overall_health']['grade']} ({result['overall_health']['score']}/100)")
    
    if result['overall_health']['issues']:
        print("Issues:")
        for issue in result['overall_health']['issues']:
            print(f"  - {issue}")
    
    print(f"\nStatistics:")
    stats = result['statistics']
    print(f"  Python files: {stats['total_python_files']}")
    print(f"  Lines of code: {stats['total_lines_of_code']}")
    
    # Ruff results
    if result['ruff']['success']:
        ruff_summary = result['ruff']['summary']
        print(f"\nRuff Analysis:")
        print(f"  Total issues: {ruff_summary['total_issues']}")
        print(f"  Errors: {ruff_summary['error_count']}")
        print(f"  Warnings: {ruff_summary['warning_count']}")
    
    # Radon results
    if result['radon']['success']:
        radon_summary = result['radon']['summary']
        print(f"\nRadon Analysis:")
        print(f"  Total functions: {radon_summary['total_functions']}")
        print(f"  Average complexity: {radon_summary['average_complexity']}")
        print(f"  High complexity functions: {radon_summary['high_complexity_functions']}")
    
    # TODO results
    if result['todos']['success']:
        todo_summary = result['todos']['summary']
        print(f"\nTODO Analysis:")
        print(f"  Total TODOs: {todo_summary['total_todos']}")
        print(f"  Files with TODOs: {todo_summary['files_with_todos']}")
    
    # Recommendations
    analyzer = StaticAnalyzer(repo_path)
    recommendations = analyzer.get_recommendations(result)
    
    if recommendations:
        print(f"\nRecommendations:")
        for rec in recommendations:
            print(f"  - {rec}") 