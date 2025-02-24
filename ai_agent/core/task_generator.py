from typing import List, Dict, Any, Optional
import os
from pathlib import Path
import re
from collections import defaultdict

class TaskGenerator:
    """Generates coding tasks from documentation and common patterns"""
    
    def __init__(self):
        self.common_patterns = {
            'testing': [
                "Add unit tests for {component}",
                "Fix failing test in {test_file}",
                "Improve test coverage for {module}"
            ],
            'refactoring': [
                "Extract {functionality} into separate module",
                "Refactor {component} to improve performance",
                "Clean up code duplication in {file}"
            ],
            'feature': [
                "Implement {feature_name} in {component}",
                "Add support for {feature} to {module}",
                "Create new {component_type} for {purpose}"
            ],
            'bugfix': [
                "Fix {error_type} in {component}",
                "Resolve {issue} affecting {feature}",
                "Debug and fix {problem} in {module}"
            ]
        }

    def extract_tasks_from_docs(self, docs_path: Path) -> List[Dict[str, Any]]:
        """Extract potential tasks from project documentation"""
        tasks = []
        
        for doc_file in docs_path.glob('**/*.md'):
            content = doc_file.read_text()
            
            # Look for TODO comments or task-like items
            todo_pattern = r'(?:TODO|FIXME|HACK|XXX):\s*(.+?)(?:\n|$)'
            todos = re.finditer(todo_pattern, content, re.IGNORECASE)
            for todo in todos:
                tasks.append({
                    'type': 'documentation',
                    'instruction': todo.group(1).strip(),
                    'source': str(doc_file),
                    'priority': 'high' if 'FIXME' in todo.group(0) else 'normal'
                })
            
            # Look for feature descriptions or requirements
            feature_pattern = r'(?:should|must|will|adds?|implements?)\s+(.+?)(?:\.|$)'
            features = re.finditer(feature_pattern, content)
            for feature in features:
                tasks.append({
                    'type': 'feature',
                    'instruction': feature.group(1).strip(),
                    'source': str(doc_file),
                    'priority': 'normal'
                })

        return tasks

    def generate_framework_tasks(self, framework_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate framework-specific tasks"""
        tasks = []
        
        frameworks = framework_info.get('frameworks', [])
        languages = framework_info.get('languages', [])
        
        # Generate testing tasks based on framework
        if 'pytest' in frameworks or 'python' in languages:
            tasks.extend([
                {
                    'type': 'testing',
                    'instruction': "Add pytest fixtures for database testing",
                    'priority': 'high'
                },
                {
                    'type': 'testing',
                    'instruction': "Implement test coverage reporting",
                    'priority': 'normal'
                }
            ])
        
        # Generate tasks for common framework patterns
        for framework in frameworks:
            if framework == 'flask':
                tasks.extend([
                    {
                        'type': 'feature',
                        'instruction': "Add error handling middleware",
                        'priority': 'high'
                    },
                    {
                        'type': 'security',
                        'instruction': "Implement request rate limiting",
                        'priority': 'normal'
                    }
                ])
            elif framework == 'django':
                tasks.extend([
                    {
                        'type': 'feature',
                        'instruction': "Create custom model manager",
                        'priority': 'normal'
                    },
                    {
                        'type': 'security',
                        'instruction': "Add field-level permissions",
                        'priority': 'high'
                    }
                ])
        
        return tasks

    def generate_codebase_tasks(self, file_patterns: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Generate tasks based on codebase patterns"""
        tasks = []
        
        # Check for common code patterns that might need attention
        if 'duplicate_code' in file_patterns:
            for file in file_patterns['duplicate_code']:
                tasks.append({
                    'type': 'refactoring',
                    'instruction': f"Remove code duplication in {file}",
                    'priority': 'high'
                })
        
        if 'complex_functions' in file_patterns:
            for file in file_patterns['complex_functions']:
                tasks.append({
                    'type': 'refactoring',
                    'instruction': f"Simplify complex functions in {file}",
                    'priority': 'high'
                })
        
        if 'low_coverage' in file_patterns:
            for file in file_patterns['low_coverage']:
                tasks.append({
                    'type': 'testing',
                    'instruction': f"Improve test coverage for {file}",
                    'priority': 'normal'
                })
        
        return tasks

    def filter_duplicate_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate or very similar tasks"""
        unique_tasks = []
        seen_instructions = set()
        
        for task in tasks:
            # Normalize instruction for comparison
            normalized = re.sub(r'\s+', ' ', task['instruction'].lower().strip())
            
            # Check if we've seen a very similar instruction
            if normalized not in seen_instructions:
                similar_exists = False
                for existing in seen_instructions:
                    if self._calculate_similarity(normalized, existing) > 0.8:
                        similar_exists = True
                        break
                
                if not similar_exists:
                    unique_tasks.append(task)
                    seen_instructions.add(normalized)
        
        return unique_tasks

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (simple implementation)"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0