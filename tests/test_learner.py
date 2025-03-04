import pytest
import numpy as np
import json
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock
from ai_agent.core.learner import SelfLearner, LearningRate
from ai_agent.data.trajectory_manager import TrajectoryManager
from ai_agent.data.quality_metrics import QualityMetrics, QualityScore

class MockLearner:
    """Mock learner class for testing"""
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.embedding_model = Mock()
        self.embedding_model.encode.return_value = np.ones(384)  # Standard embedding dimension
        
    def compute_embedding(self, text: str) -> np.ndarray:
        return np.ones(384)  # Return constant embedding for testing
        
    def backward_construct(self, data: dict) -> str:
        return self.responses.get('backward_construct', "Refined instruction")
    
    def chat_completion(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        return self.responses.get('chat_completion', {
            "choices": [{"message": {"content": '{"type": "test", "action": "mock"}'}}]
        })

@pytest.fixture
def mock_openai_client():
    client = Mock()
    # Mock chat completions
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content='{"type": "test", "path": "test.py", "content": "test content"}'))]
    client.chat.completions.create.return_value = mock_response
    return client

@pytest.fixture
def learner_with_mock(mock_openai_client):
    return SelfLearner(client=mock_openai_client, api_key="mock-key")

def test_analyze_task_context(learner_with_mock):
    """Test task context analysis with mocked responses"""
    # Prepare mock response
    mock_content = '{"required_operations": [{"file": "test.py"}], "dependencies": ["pytest"], "risk_factors": []}'
    learner_with_mock.client.chat.completions.create.return_value.choices[0].message.content = mock_content
    
    result = learner_with_mock._analyze_task_context(
        "Add unit tests",
        {'files': ['test.py'], 'git_status': {}}
    )
    
    assert result.get('required_operations')[0]['file'] == 'test.py'
    assert 'pytest' in result.get('dependencies', [])

def test_evaluate_action_risk(learner_with_mock):
    """Test risk evaluation for different action types"""
    # Test file edit risk
    edit_action = {
        'type': 'edit_file',
        'path': 'test.py',
        'content': 'print("test")'
    }
    context = {'risk_factors': []}
    risk = learner_with_mock._evaluate_action_risk(edit_action, context)
    assert 0 <= risk <= 1  # Risk should be normalized
    
    # Test high-risk action
    delete_action = {
        'type': 'edit_file',
        'path': 'config.py',
        'content': 'delete sensitive data'
    }
    risk = learner_with_mock._evaluate_action_risk(delete_action, context)
    assert risk > 0.5  # Should be considered higher risk

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
    
    # Configure mock to return valid action
    learner_with_mock.client.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content='{"type": "edit_file", "path": "test.py"}'))]
    )

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
    # Configure mock to return pattern suggestions
    learner_with_mock.client.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content='["validation", "input_handling"]'))]
    )
    
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
        }
    ]
    
    suggestions = learner_with_mock._get_pattern_suggestions(
        "Create new auth module",
        current_state,
        examples
    )
    
    assert len(suggestions) > 0
    suggestion = suggestions[0]
    assert suggestion['pattern_type'] == 'create_file'
    
    # Test auth module path
    assert 'auth.py' in suggestion['suggested_actions'][0]['path']

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
    
    # Configure mock to return pattern-based action
    learner_with_mock.client.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content='{"type": "create_file", "path": "src/auth.py"}'))]
    )
    
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

def test_semantic_task_matching(learner_with_mock):
    """Test semantic similarity for task matching"""
    task1 = {
        'instruction': 'Add input validation to login form',
        'context': {'frameworks': ['django']}
    }
    task2 = {
        'instruction': 'Implement form validation for login',
        'context': {'frameworks': ['django']}
    }
    task3 = {
        'instruction': 'Setup database connection',
        'context': {'frameworks': ['django']}
    }
    
    # Configure embeddings mock
    base_embedding = np.ones(384)  # all-MiniLM-L6-v2 dimension
    learner_with_mock.embedding_model.encode.return_value = base_embedding
    
    # Test similar tasks have high similarity
    sim1 = learner_with_mock._calculate_semantic_similarity(
        task1['instruction'],
        task2['instruction']
    )
    assert sim1 > 0.7  # High similarity threshold
    
    # Test different tasks have low similarity
    sim2 = learner_with_mock._calculate_semantic_similarity(
        task1['instruction'],
        task3['instruction']
    )
    assert sim2 < 0.5  # Low similarity threshold

def test_backward_construction(learner_with_mock):
    """Test generation of refined instructions from trajectories"""
    # Create a test trajectory
    trajectory = {
        'instruction': 'Update code',
        'actions': [
            {'type': 'add_validation', 'path': 'login.py'},
            {'type': 'run_tests', 'path': 'test_login.py'}
        ],
        'observations': [
            {'status': 'success'},
            {'status': 'success'}
        ],
        'final_state': {
            'files': ['login.py', 'test_login.py'],
            'git_status': {'modified': ['login.py']}
        }
    }
    
    # Configure mock to return more specific instruction
    learner_with_mock.client.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content='"Add input validation to login form and verify with tests"'))]
    )
    
    refined = learner_with_mock.backward_construct(trajectory)
    assert 'validation' in refined.lower()
    assert 'login' in refined.lower()
    assert 'test' in refined.lower()

def test_learning_rate_decay():
    """Test learning rate decay over time"""
    rate = LearningRate()
    
    # Add some successes
    rate.update(True)
    rate.update(True)
    initial_confidence = rate.rate
    
    # Wait a bit and check decay
    import time
    time.sleep(0.1)
    
    assert rate.rate < initial_confidence