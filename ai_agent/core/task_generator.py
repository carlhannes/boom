from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json
import numpy as np
from collections import defaultdict
import os

class Task:
    def __init__(self, type: str, description: str, priority: str = 'medium', **kwargs):
        self.type = type
        self.description = description
        self.priority = priority
        self.__dict__.update(kwargs)
    
    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

class TaskGenerator:
    def __init__(self, api_key: str = None, client=None):
        self.api_key = api_key
        self.client = client
        self.generated_tasks = []
        self.task_history = []
        self.file_types = {
            'models': ['models', 'schema', 'entity'],
            'views': ['views', 'templates', 'pages'],
            'controllers': ['controllers', 'handlers', 'routes'],
            'tests': ['test', 'spec'],
            'utils': ['utils', 'helpers', 'lib']
        }
        
    def generate_tasks_from_files(self, files: List[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tasks based on codebase files and context"""
        tasks = []
        
        # Group files by type/purpose
        grouped_files = self._group_files(files)
        
        for group, group_files in grouped_files.items():
            # Generate group-specific tasks
            group_tasks = self._generate_group_tasks(group, group_files, context)
            tasks.extend(group_tasks)
            
        # Track generated tasks
        self.generated_tasks.extend(tasks)
        return tasks
        
    def _group_files(self, files: List[str]) -> Dict[str, List[str]]:
        """Group files by their type/role in the project"""
        groups = {}
        
        for file in files:
            path = Path(file)
            parts = path.parts
            
            # Extract filename for use in test matching
            filename = path.name
            
            # Handle special directories
            if 'tests' in parts:
                group = 'tests'
            elif 'models' in parts:
                group = 'models'
            elif 'views' in parts:
                group = 'views'
            elif 'utils' in parts:
                group = 'utils'
            else:
                # Default to first directory or 'src'
                group = parts[0] if len(parts) > 1 else 'src'
                
            if group not in groups:
                groups[group] = []
                
            # Use just the filename for tests
            if group == 'tests':
                groups[group].append(filename)
            else:
                groups[group].append(file)
            
        return groups
        
    def _generate_group_tasks(self, group: str, files: List[str], 
                            context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tasks specific to file group"""
        tasks = []
        
        if group == 'tests':
            tasks.extend(self._generate_test_tasks(files, context))
        elif group == 'models':
            tasks.extend(self._generate_model_tasks(files, context))
        elif group == 'views':
            tasks.extend(self._generate_view_tasks(files, context))
        elif group == 'utils':
            tasks.extend(self._generate_util_tasks(files, context))
            
        return tasks
        
    def _generate_test_tasks(self, files: List[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test-related tasks"""
        tasks = []
        
        # Find files without tests
        all_files = set(context.get('files', []))
        tested_files = {self._get_tested_file(f) for f in files}
        untested_files = [f for f in all_files if f not in tested_files and f.endswith('.py')]
        
        # Generate tasks for missing tests
        for file in untested_files:
            tasks.append({
                'type': 'create_test',
                'description': f'Create tests for {file}',
                'file': file,
                'priority': 'high' if file in context.get('modified_files', []) else 'medium'
            })
            
        # Generate tasks for improving test coverage
        for test_file in files:
            tasks.append({
                'type': 'improve_tests',
                'description': f'Improve test coverage in {test_file}',
                'file': test_file,
                'priority': 'medium'
            })
            
        return tasks
        
    def _generate_model_tasks(self, files: List[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate model-related tasks"""
        tasks = []
        
        for file in files:
            # Task for adding validation
            tasks.append({
                'type': 'add_validation',
                'description': f'Add input validation to {file}',
                'file': file,
                'priority': 'medium'
            })
            
            # Task for improving documentation
            tasks.append({
                'type': 'improve_docs',
                'description': f'Improve model documentation in {file}',
                'file': file,
                'priority': 'low'
            })
            
        return tasks
        
    def _generate_view_tasks(self, files: List[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate view-related tasks"""
        tasks = []
        
        for file in files:
            # Task for error handling
            tasks.append({
                'type': 'add_error_handling',
                'description': f'Add error handling to views in {file}',
                'file': file,
                'priority': 'high'
            })
            
            # Task for response validation
            tasks.append({
                'type': 'validate_responses',
                'description': f'Add response validation to {file}',
                'file': file,
                'priority': 'medium'
            })
            
        return tasks
        
    def _generate_util_tasks(self, files: List[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate utility-related tasks"""
        tasks = []
        
        for file in files:
            # Task for adding type hints
            tasks.append({
                'type': 'add_type_hints',
                'description': f'Add type hints to utilities in {file}',
                'file': file,
                'priority': 'low'
            })
            
        return tasks
        
    def _get_tested_file(self, test_file: str) -> str:
        """Get the source file that a test file is testing"""
        path = Path(test_file)
        if path.stem.startswith('test_'):
            # Convert test path back to source path correctly
            source_file = path.stem[5:] + path.suffix
            return str(path.parent.parent / source_file)
        return test_file
        
    def validate_task(self, task: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Validate if a task is appropriate for current context"""
        if not task or 'type' not in task or 'file' not in task:
            return False
            
        # Check if file exists
        if task['file'] not in context.get('files', []):
            return False
            
        # Check if task is relevant
        if task['type'] == 'create_test':
            # Don't create tests for test files
            if 'test' in Path(task['file']).stem:
                return False
                
        elif task['type'] == 'improve_tests':
            # Only improve existing test files
            if 'test' not in Path(task['file']).stem:
                return False
                
        return True
        
    def get_task_history(self) -> List[Dict[str, Any]]:
        """Get history of generated tasks"""
        return self.task_history
        
    def save_task_history(self, storage_path: str) -> None:
        """Save task history to disk"""
        history_file = Path(storage_path) / 'task_history.json'
        history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Debug info
        print(f"Saving {len(self.task_history)} tasks to {history_file}")
        
        # Convert all tasks to dictionaries for serialization
        serialized_tasks = []
        for task in self.task_history:
            if isinstance(task, Task):
                serialized_tasks.append(task.to_dict())
            elif isinstance(task, dict):
                serialized_tasks.append(task)
        
        # Debug info
        print(f"Serialized {len(serialized_tasks)} tasks")
            
        with open(history_file, 'w') as f:
            json.dump(serialized_tasks, f)
            
    def load_task_history(self, storage_path: str) -> None:
        """Load task history from disk"""
        history_file = Path(storage_path) / 'task_history.json'
        if not history_file.exists():
            print(f"History file not found: {history_file}")
            self.task_history = []
            return
            
        with open(history_file, 'r') as f:
            data = json.load(f)
            print(f"Loaded {len(data)} tasks from {history_file}")
            # Keep tasks as dictionaries for consistent comparison in tests
            self.task_history = data
            
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated tasks"""
        if not self.task_history:
            return {}
            
        stats = {
            'total_tasks': len(self.task_history),
            'by_type': {},
            'by_priority': {},
            'by_file': {}
        }
        
        for task in self.task_history:
            # Count by type
            task_type = task.get('type')
            stats['by_type'][task_type] = stats['by_type'].get(task_type, 0) + 1
            
            # Count by priority
            priority = task.get('priority')
            stats['by_priority'][priority] = stats['by_priority'].get(priority, 0) + 1
            
            # Count by file
            file = task.get('file')
            stats['by_file'][file] = stats['by_file'].get(file, 0) + 1
            
        return stats

    def extract_tasks_from_docs(self, docs_path: str) -> List[Dict]:
        """Extract tasks from documentation files"""
        tasks = []
        docs_dir = Path(docs_path)
        
        for file in docs_dir.glob('**/*.md'):
            content = file.read_text()
            # Extract TODOs and FIXMEs
            for line in content.split('\n'):
                if 'TODO:' in line:
                    tasks.append({
                        'type': 'feature',
                        'description': line.split('TODO:')[1].strip(),
                        'priority': 'medium'
                    })
                elif 'FIXME:' in line:
                    tasks.append({
                        'type': 'bugfix',
                        'description': line.split('FIXME:')[1].strip(),
                        'priority': 'high'
                    })

        return tasks

    def generate_framework_tasks(self, framework_info: Dict) -> List[Dict]:
        """Generate tasks based on framework requirements"""
        tasks = []
        frameworks = framework_info.get('frameworks', [])
        languages = framework_info.get('languages', [])

        if 'pytest' in frameworks:
            tasks.append({
                'type': 'test',
                'description': 'Add pytest configuration',
                'priority': 'high'
            })

        if 'python' in languages:
            tasks.append({
                'type': 'quality',
                'description': 'Add type hints to functions',
                'priority': 'medium'
            })

        return tasks

    def generate_codebase_tasks(self, file_patterns: Dict) -> List[Dict]:
        """Generate tasks based on codebase patterns"""
        tasks = []
        
        for pattern, files in file_patterns.items():
            if (pattern == 'duplicate_code'):
                tasks.append({
                    'type': 'refactor',
                    'description': f'Remove code duplication in {", ".join(files)}',
                    'priority': 'medium'
                })
            elif pattern == 'complex_functions':
                tasks.append({
                    'type': 'refactor',
                    'description': f'Simplify complex functions in {", ".join(files)}',
                    'priority': 'high'
                })

        return tasks

    def filter_duplicate_tasks(self, tasks: List[Dict]) -> List[Dict]:
        """Filter out duplicate or highly similar tasks"""
        # Sort tasks by priority first
        priority_values = {'high': 3, 'medium': 2, 'low': 1}
        sorted_tasks = sorted(
            tasks,
            key=lambda x: priority_values.get(x.get('priority', 'medium'), 0),
            reverse=True
        )
        
        filtered = []
        for task in sorted_tasks:
            # Check if task is too similar to any existing one
            if not any(self._compute_task_similarity(task, existing) > 0.8 
                      for existing in filtered):
                filtered.append(task)
        
        return filtered
    
    def _compute_task_similarity(self, task1, task2):
        """Compute similarity between two tasks"""
        if task1['type'] != task2['type']:
            return 0.0
            
        desc1_words = set(task1['description'].lower().split())
        desc2_words = set(task2['description'].lower().split())
        
        if not desc1_words or not desc2_words:
            return 0.0
            
        common_words = desc1_words.intersection(desc2_words)
        similarity = len(common_words) / max(len(desc1_words), len(desc2_words))
        
        return similarity