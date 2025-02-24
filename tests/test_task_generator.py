import pytest
from pathlib import Path
from ai_agent.core.task_generator import TaskGenerator

def test_extract_tasks_from_docs(tmp_path):
    # Create test documentation files
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    
    # Create a test markdown file with TODOs and feature descriptions
    doc_file = docs_dir / "readme.md"
    doc_file.write_text("""
    # Project Documentation
    
    TODO: Implement error handling for API requests
    FIXME: Fix security vulnerability in auth module
    
    The system should provide real-time updates for user status.
    This feature implements automatic data synchronization.
    """)
    
    generator = TaskGenerator()
    tasks = generator.extract_tasks_from_docs(docs_dir)
    
    # Verify task extraction
    assert len(tasks) >= 4  # 2 todos + 2 feature descriptions
    assert any(task['instruction'].startswith('Implement error handling') for task in tasks)
    assert any(task['instruction'].startswith('Fix security vulnerability') for task in tasks)
    assert any('real-time updates' in task['instruction'] for task in tasks)
    
    # Verify FIXME tasks are high priority
    security_task = next(task for task in tasks if 'security' in task['instruction'].lower())
    assert security_task['priority'] == 'high'

def test_generate_framework_tasks():
    generator = TaskGenerator()
    
    # Test Python/pytest framework tasks
    framework_info = {
        'frameworks': ['pytest'],
        'languages': ['python']
    }
    
    tasks = generator.generate_framework_tasks(framework_info)
    
    # Verify pytest-specific tasks are generated
    assert any('pytest fixtures' in task['instruction'].lower() for task in tasks)
    assert any('test coverage' in task['instruction'].lower() for task in tasks)
    
    # Test Flask framework tasks
    framework_info = {
        'frameworks': ['flask'],
        'languages': ['python']
    }
    
    tasks = generator.generate_framework_tasks(framework_info)
    
    # Verify Flask-specific tasks are generated
    assert any('middleware' in task['instruction'].lower() for task in tasks)
    assert any('rate limiting' in task['instruction'].lower() for task in tasks)

def test_generate_codebase_tasks():
    generator = TaskGenerator()
    
    file_patterns = {
        'duplicate_code': ['utils.py', 'helpers.py'],
        'complex_functions': ['service.py'],
        'low_coverage': ['models.py']
    }
    
    tasks = generator.generate_codebase_tasks(file_patterns)
    
    # Verify pattern-based tasks are generated
    assert any('duplication' in task['instruction'].lower() for task in tasks)
    assert any('complex functions' in task['instruction'].lower() for task in tasks)
    assert any('test coverage' in task['instruction'].lower() for task in tasks)
    
    # Verify file references are included
    assert any('utils.py' in task['instruction'] for task in tasks)
    assert any('service.py' in task['instruction'] for task in tasks)

def test_filter_duplicate_tasks():
    generator = TaskGenerator()
    
    tasks = [
        {'instruction': 'Implement error handling', 'type': 'feature'},
        {'instruction': 'Add error handling', 'type': 'feature'},  # Similar to first
        {'instruction': 'Fix security bug', 'type': 'bugfix'},
        {'instruction': 'Update documentation', 'type': 'docs'}
    ]
    
    filtered_tasks = generator.filter_duplicate_tasks(tasks)
    
    # Verify similar tasks are filtered out
    assert len(filtered_tasks) < len(tasks)
    
    # Verify unique tasks are preserved
    instructions = [task['instruction'].lower() for task in filtered_tasks]
    assert any('error handling' in instr for instr in instructions)
    assert any('security' in instr for instr in instructions)
    assert any('documentation' in instr for instr in instructions)