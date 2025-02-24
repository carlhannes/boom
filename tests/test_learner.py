import pytest
import numpy as np
from unittest.mock import Mock, patch
from ai_agent.core.learner import SelfLearner

@pytest.fixture
def mock_openai_client():
    client = Mock()
    client.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content='{"action": "test"}'))]
    )
    return client

@pytest.fixture
def learner_with_mock(mock_openai_client):
    learner = SelfLearner(api_key="mock-key", client=mock_openai_client)
    return learner

def test_analyze_task_context(learner_with_mock):
    """Test task context analysis with repository state"""
    repo_state = {
        'files': ['src/app.py', 'tests/test_app.py'],
        'git_status': {'modified': ['src/app.py']},
        'frameworks': ['python'],
        'languages': ['python'],
        'patterns': ['testing']
    }
    
    context = learner_with_mock._analyze_task_context(
        "Add input validation to user form",
        repo_state
    )
    
    assert isinstance(context, dict)
    assert context  # Should not be empty

def test_evaluate_action_risk(learner_with_mock):
    """Test risk evaluation for different action types"""
    # Test edit file action
    edit_action = {
        'type': 'edit_file',
        'path': 'src/app.py',
        'content': 'def validate_form():\n    pass'
    }
    context = {'risk_factors': []}
    risk = learner_with_mock._evaluate_action_risk(edit_action, context)
    assert 0 <= risk <= 1
    
    # Test high-risk action
    risky_action = {
        'type': 'resolve_conflict',
        'path': 'src/app.py'
    }
    risk = learner_with_mock._evaluate_action_risk(risky_action, context)
    assert risk > 0.7  # Should be considered high risk

def test_validate_action(learner_with_mock):
    """Test action validation against context"""
    context = {
        'required_operations': [
            {'file': 'src/app.py', 'type': 'edit'}
        ]
    }
    
    # Valid action
    valid_action = {
        'type': 'edit_file',
        'path': 'src/app.py',
        'content': 'def new_function():\n    pass'
    }
    assert learner_with_mock._validate_action(valid_action, context)
    
    # Invalid action (missing required fields)
    invalid_action = {
        'type': 'edit_file',
        'path': 'src/app.py'
    }
    assert not learner_with_mock._validate_action(invalid_action, context)
    
    # Invalid action (wrong file)
    wrong_file_action = {
        'type': 'edit_file',
        'path': 'src/wrong.py',
        'content': 'test'
    }
    assert not learner_with_mock._validate_action(wrong_file_action, context)

def test_plan_next_action(learner_with_mock):
    """Test action planning with state and history"""
    current_state = {
        'files': ['src/app.py', 'tests/test_app.py'],
        'git_status': {'modified': []},
        'frameworks': ['python'],
        'languages': ['python'],
        'patterns': ['testing']
    }
    
    history = [
        {
            'type': 'edit_file',
            'path': 'src/app.py',
            'content': 'def old_function():\n    pass'
        }
    ]
    
    action = learner_with_mock.plan_next_action(
        current_state,
        "Add input validation",
        history
    )
    
    assert isinstance(action, dict)
    assert 'type' in action
    assert learner_with_mock._validate_action(action, {})

def test_chat_completion_system_context(learner_with_mock):
    """Test chat completion with system context preservation"""
    messages = [
        {
            "role": "system",
            "content": "Original system message"
        },
        {
            "role": "user",
            "content": "Test message"
        }
    ]
    
    response = learner_with_mock.chat_completion(messages)
    assert response.choices[0].message.content
    
    # Check that system context was enhanced
    called_messages = learner_with_mock.client.chat.completions.create.call_args[1]['messages']
    assert "Original system message" in called_messages[0]['content']
    assert "expert coding agent" in called_messages[0]['content'].lower()

def test_high_risk_action_rejection(learner_with_mock):
    """Test that high-risk actions are rejected"""
    context = {
        'risk_factors': [
            {'severity': 0.9, 'type': 'data_loss'}
        ]
    }
    
    dangerous_action = {
        'type': 'resolve_conflict',
        'path': 'critical_data.json',
        'strategy': 'force'
    }
    
    # Should reject high-risk action
    assert not learner_with_mock._validate_action(dangerous_action, context)

@pytest.mark.parametrize("action_type,expected_base_risk", [
    ('edit_file', 0.4),
    ('git_commit', 0.2),
    ('git_checkout', 0.6),
    ('run_tests', 0.1),
    ('fix_permissions', 0.7),
    ('resolve_conflict', 0.8)
])
def test_action_type_risk_levels(learner_with_mock, action_type, expected_base_risk):
    """Test risk levels for different action types"""
    action = {'type': action_type}
    risk = learner_with_mock._evaluate_action_risk(action, {})
    assert abs(risk - expected_base_risk) < 0.01

def test_get_pattern_suggestions(learner_with_mock):
    """Test extraction of pattern suggestions from examples"""
    current_state = {
        'files': ['src/app.py'],
        'frameworks': ['python'],
        'languages': ['python']
    }
    
    examples = [
        {
            'instruction': 'Create validation module',
            'actions': [
                {'type': 'create_file', 'path': 'src/validation.py'},
                {'type': 'edit_file', 'path': 'src/validation.py', 'content': 'def validate(): pass'}
            ],
            'observations': [
                {'status': 'success', 'state': {}},
                {'status': 'success', 'state': {}}
            ],
            'final_state': {'files': ['src/validation.py']}
        },
        {
            'instruction': 'Create helper module',
            'actions': [
                {'type': 'create_file', 'path': 'src/helper.py'},
                {'type': 'edit_file', 'path': 'src/helper.py', 'content': 'def help(): pass'}
            ],
            'observations': [
                {'status': 'success', 'state': {}},
                {'status': 'success', 'state': {}}
            ],
            'final_state': {'files': ['src/helper.py']}
        }
    ]
    
    suggestions = learner_with_mock._get_pattern_suggestions(
        "Create new auth module",
        current_state,
        examples
    )
    
    assert len(suggestions) > 0
    suggestion = suggestions[0]
    assert suggestion['pattern_type'] == 'create'
    assert suggestion['success_rate'] > 0.8
    assert len(suggestion['suggested_actions']) > 0
    assert suggestion['suggested_actions'][0]['type'] == 'create_file'

def test_pattern_based_planning(learner_with_mock):
    """Test action planning with pattern suggestions"""
    current_state = {
        'files': ['src/app.py'],
        'frameworks': ['python'],
        'languages': ['python']
    }
    
    history = []
    examples = [
        {
            'instruction': 'Create validation module',
            'actions': [
                {'type': 'create_file', 'path': 'src/validation.py'},
                {'type': 'edit_file', 'path': 'src/validation.py', 'content': 'def validate(): pass'}
            ],
            'observations': [
                {'status': 'success', 'state': {}},
                {'status': 'success', 'state': {}}
            ],
            'final_state': {'files': ['src/validation.py']}
        }
    ]
    
    # First action should follow pattern
    action = learner_with_mock.plan_next_action(
        current_state,
        "Create new auth module",
        history,
        examples
    )
    
    assert action is not None
    assert action['type'] == 'create_file'
    assert 'auth' in action['path'].lower()
    
    # Second action should continue pattern
    history.append(action)
    action2 = learner_with_mock.plan_next_action(
        current_state,
        "Create new auth module",
        history,
        examples
    )
    
    assert action2 is not None
    assert action2['type'] == 'edit_file'
    assert action2['path'] == action['path']

def test_planning_fallback(learner_with_mock):
    """Test fallback to pattern suggestions when LLM fails"""
    current_state = {
        'files': ['src/app.py'],
        'frameworks': ['python'],
        'languages': ['python']
    }
    
    # Configure mock to fail JSON parsing
    learner_with_mock.client.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content='invalid json'))]
    )
    
    history = []
    examples = [
        {
            'instruction': 'Create validation module',
            'actions': [
                {'type': 'create_file', 'path': 'src/validation.py'},
                {'type': 'edit_file', 'path': 'src/validation.py', 'content': 'def validate(): pass'}
            ],
            'observations': [
                {'status': 'success', 'state': {}},
                {'status': 'success', 'state': {}}
            ],
            'final_state': {'files': ['src/validation.py']}
        }
    ]
    
    # Should fall back to pattern suggestion
    action = learner_with_mock.plan_next_action(
        current_state,
        "Create new auth module",
        history,
        examples
    )
    
    assert action is not None
    assert action['type'] == 'create_file'