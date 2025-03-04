import pytest
from pathlib import Path
from ai_agent.core.task_generator import TaskGenerator

@pytest.fixture
def test_files():
    return [
        'src/models/user.py',
        'src/views/auth.py',
        'src/utils/helpers.py',
        'tests/test_user.py',
        'tests/test_auth.py'
    ]

@pytest.fixture
def test_context():
    return {
        'files': [
            'src/models/user.py',
            'src/views/auth.py',
            'src/utils/helpers.py',
            'tests/test_user.py',
            'tests/test_auth.py'
        ],
        'modified_files': ['src/models/user.py'],
        'frameworks': ['flask'],
        'languages': ['python']
    }

def test_task_generation(test_files, test_context):
    """Test generation of tasks from files"""
    generator = TaskGenerator()
    
    tasks = generator.generate_tasks_from_files(test_files, test_context)
    
    assert len(tasks) > 0
    assert all('type' in task for task in tasks)
    assert all('description' in task for task in tasks)
    assert all('priority' in task for task in tasks)

def test_file_grouping(test_files):
    """Test grouping of files by type"""
    generator = TaskGenerator()
    
    groups = generator._group_files(test_files)
    
    assert 'tests' in groups
    assert 'models' in groups
    assert 'views' in groups
    assert len(groups['tests']) == 2
    assert 'test_user.py' in groups['tests']

def test_test_task_generation(test_files, test_context):
    """Test generation of test-related tasks"""
    generator = TaskGenerator()
    
    tasks = generator._generate_test_tasks(
        [f for f in test_files if 'test' in f],
        test_context
    )
    
    assert len(tasks) > 0
    assert any(t['type'] == 'create_test' for t in tasks)
    assert any(t['type'] == 'improve_tests' for t in tasks)
    
    # Check priority assignment
    modified_tasks = [t for t in tasks 
                     if t['file'] in test_context['modified_files']]
    assert all(t['priority'] == 'high' for t in modified_tasks)

def test_model_task_generation(test_files, test_context):
    """Test generation of model-related tasks"""
    generator = TaskGenerator()
    
    model_files = [f for f in test_files if 'models' in f]
    tasks = generator._generate_model_tasks(model_files, test_context)
    
    assert len(tasks) > 0
    assert any(t['type'] == 'add_validation' for t in tasks)
    assert any(t['type'] == 'improve_docs' for t in tasks)

def test_task_validation(test_context):
    """Test task validation logic"""
    generator = TaskGenerator()
    
    # Valid task
    valid_task = {
        'type': 'create_test',
        'file': 'src/models/user.py',
        'priority': 'high',
        'description': 'Create tests'
    }
    assert generator.validate_task(valid_task, test_context)
    
    # Invalid task - non-existent file
    invalid_task = {
        'type': 'create_test',
        'file': 'src/models/nonexistent.py',
        'priority': 'high',
        'description': 'Create tests'
    }
    assert not generator.validate_task(invalid_task, test_context)
    
    # Invalid task - creating test for test file
    invalid_test_task = {
        'type': 'create_test',
        'file': 'tests/test_user.py',
        'priority': 'high',
        'description': 'Create tests'
    }
    assert not generator.validate_task(invalid_test_task, test_context)

def test_task_history_persistence(tmp_path):
    """Test saving and loading task history"""
    generator = TaskGenerator()
    
    # Create sample tasks directly
    sample_tasks = [
        {'type': 'create_test', 'file': 'test1.py', 'priority': 'high', 'description': 'Create test'},
        {'type': 'add_validation', 'file': 'model.py', 'priority': 'high', 'description': 'Add validation'}
    ]
    
    # Add to task history
    generator.task_history = sample_tasks
    
    # Save history
    generator.save_task_history(str(tmp_path))
    
    # Create new generator and load history
    new_generator = TaskGenerator()
    new_generator.load_task_history(str(tmp_path))
    
    # Verify tasks were saved and loaded correctly
    assert len(new_generator.task_history) == 2
    assert new_generator.task_history == generator.task_history

def test_task_statistics():
    """Test task statistics calculation"""
    generator = TaskGenerator()
    
    # Add some test tasks to history
    generator.task_history = [
        {'type': 'create_test', 'file': 'test1.py', 'priority': 'high', 'description': 'Create test'},
        {'type': 'create_test', 'file': 'test2.py', 'priority': 'medium', 'description': 'Create test'},
        {'type': 'add_validation', 'file': 'model.py', 'priority': 'high', 'description': 'Add validation'}
    ]
    
    stats = generator.get_task_statistics()
    
    assert stats['total_tasks'] == 3
    assert stats['by_type']['create_test'] == 2
    assert stats['by_priority']['high'] == 2
    assert 'test1.py' in stats['by_file']

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
    tasks = generator.extract_tasks_from_docs(str(docs_dir))
    
    # Verify task extraction
    assert len(tasks) >= 2  # At least TODO and FIXME
    assert any(task['description'].startswith('Implement error handling') for task in tasks)
    assert any(task['description'].startswith('Fix security vulnerability') for task in tasks)
    
    # Verify FIXME tasks are high priority
    security_task = next(task for task in tasks 
                        if 'security' in task['description'].lower())
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
    assert any('pytest' in task['description'].lower() for task in tasks)
    assert len(tasks) > 0

def test_generate_codebase_tasks():
    generator = TaskGenerator()
    
    file_patterns = {
        'duplicate_code': ['utils.py', 'helpers.py'],
        'complex_functions': ['service.py'],
        'low_coverage': ['models.py']
    }
    
    tasks = generator.generate_codebase_tasks(file_patterns)
    
    # Verify pattern-based tasks are generated
    assert any('duplication' in task['description'].lower() for task in tasks)
    assert any('complex' in task['description'].lower() for task in tasks)
    assert any('utils.py' in task['description'] for task in tasks)
    assert any('service.py' in task['description'] for task in tasks)

def test_filter_duplicate_tasks():
    generator = TaskGenerator()
    
    tasks = [
        {'type': 'feature', 'description': 'Implement error handling', 'priority': 'high'},
        {'type': 'feature', 'description': 'Add error handling', 'priority': 'medium'},  # Similar to first
        {'type': 'bugfix', 'description': 'Fix security bug', 'priority': 'high'},
        {'type': 'docs', 'description': 'Update documentation', 'priority': 'low'}
    ]
    
    filtered_tasks = generator.filter_duplicate_tasks(tasks)
    
    # Verify similar tasks are filtered out
    assert len(filtered_tasks) < len(tasks)
    
    # Verify unique tasks are preserved
    descriptions = [task['description'].lower() for task in filtered_tasks]
    assert any('error handling' in desc for desc in descriptions)
    assert any('security' in desc for desc in descriptions)
    assert any('documentation' in desc for desc in descriptions)