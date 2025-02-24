import pytest
import numpy as np
from unittest.mock import Mock, patch
from ai_agent.core.learner import SelfLearner
from ai_agent.data.trajectory_manager import TrajectoryManager

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

def test_backward_construction():
    learner = SelfLearner()
    
    # Create a test trajectory that created and modified files
    trajectory = type('Trajectory', (), {
        'instruction': "Update the project",  # Original vague instruction
        'actions': [
            {'type': 'create_file', 'file': 'new_feature.py'},
            {'type': 'edit_file', 'file': 'new_feature.py'},
            {'type': 'edit_file', 'file': 'new_feature.py'},
            {'type': 'run_tests'},
            {'type': 'fix_imports', 'file': 'new_feature.py'},
            {'type': 'run_tests'}
        ],
        'observations': [
            {'status': 'success'},
            {'status': 'success'},
            {'status': 'success'},
            {'status': 'error'},
            {'status': 'success'},
            {'status': 'success'}
        ],
        'final_state': {
            'test_results': {'status': 'pass'},
            'git_status': {'staged': True}
        }
    })()
    
    # Generate more specific instruction
    new_instruction = learner.backward_construct(trajectory)
    
    # Verify instruction captures what was actually done
    assert "Create new_feature.py" in new_instruction
    assert "Modify new_feature.py" in new_instruction
    assert "tests" in new_instruction.lower()
    assert "stage" in new_instruction.lower()

def test_instruction_refinement():
    learner = SelfLearner()
    tm = TrajectoryManager("test_storage")
    
    # Create test trajectories
    trajectories = [
        type('Trajectory', (), {
            'instruction': "Make changes",
            'actions': [
                {'type': 'edit_file', 'file': 'app.py'},
                {'type': 'run_tests'}
            ],
            'observations': [
                {'status': 'success'},
                {'status': 'success'}
            ],
            'final_state': {'test_results': {'status': 'pass'}},
            'compute_quality_metrics': lambda self: type('Metrics', (), {'success_rate': 1.0})()
        })(),
        type('Trajectory', (), {
            'instruction': "Update code",
            'actions': [
                {'type': 'create_file', 'file': 'utils.py'},
                {'type': 'add_dependency', 'what': 'requests'}
            ],
            'observations': [
                {'status': 'success'},
                {'status': 'success'}
            ],
            'final_state': {},
            'compute_quality_metrics': lambda self: type('Metrics', (), {'success_rate': 1.0})()
        })()
    ]
    
    # Mock trajectory manager
    tm.trajectories = trajectories
    tm.store_trajectory = lambda t: None
    
    # Refine instructions
    updated = learner.refine_instruction_library(tm)
    
    # Verify refinement happened
    assert updated > 0

def test_action_grouping():
    learner = SelfLearner()
    
    actions = [
        {'type': 'edit_file', 'file': 'test.py'},
        {'type': 'edit_file', 'file': 'test.py'},
        {'type': 'run_tests'},
        {'type': 'fix_imports', 'file': 'test.py'},
        {'type': 'run_tests'}
    ]
    
    groups = learner._group_related_actions(actions)
    
    # Verify grouping logic
    assert len(groups) == 3  # edit+edit, run, fix+run
    assert len(groups[0]) == 2  # Two edit actions grouped together
    assert groups[0][0]['type'] == 'edit_file'
    assert groups[1][0]['type'] == 'run_tests'

import pytest
from unittest.mock import Mock, MagicMock
from ai_agent.core.learner import SelfLearner
from pathlib import Path

def test_bootstrap_learning(tmp_path):
    # Setup test environment
    learner = SelfLearner()
    
    # Mock dependencies
    learner.agent = Mock()
    trajectory_manager = Mock()
    
    # Create test docs directory with some content
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    (docs_path / "README.md").write_text("""
        TODO: Implement feature X
        The system should do Y
    """)
    
    # Mock successful task execution
    learner.agent.execute_task.return_value = {
        'status': 'success',
        'trajectory': type('Trajectory', (), {
            'actions': [{'type': 'edit_file'}],
            'observations': [{'state_before': {}}],
            'final_state': {},
            'compute_quality_metrics': lambda: type('Metrics', (), {'success_rate': 1.0})()
        })()
    }
    
    # Test bootstrap learning
    framework_info = {'frameworks': ['pytest'], 'languages': ['python']}
    file_patterns = {'complex_functions': ['service.py']}
    
    success_count = learner.bootstrap_learning(
        str(docs_path),
        framework_info,
        file_patterns,
        trajectory_manager
    )
    
    # Verify tasks were generated and executed
    assert success_count > 0
    assert learner.agent.execute_task.called
    assert trajectory_manager.store_trajectory.called

def test_pattern_learning():
    learner = SelfLearner()
    
    # Create test trajectory
    trajectory = type('Trajectory', (), {
        'actions': [
            {'type': 'create_file', 'file': 'test.py'},
            {'type': 'edit_file', 'file': 'test.py'},
            {'type': 'run_tests'}
        ],
        'observations': [{'state_before': {'files': []}}],
        'compute_quality_metrics': lambda: type('Metrics', (), {'success_rate': 0.9})()
    })()
    
    # Learn patterns from trajectory
    learner._learn_patterns(trajectory)
    
    # Verify patterns were learned
    assert len(learner.learned_patterns) > 0
    
    # Test pattern matching
    similar = learner.get_similar_patterns(
        [{'type': 'create_file'}, {'type': 'edit_file'}],
        {'files': []}
    )
    
    assert len(similar) > 0
    assert all(p['success_rate'] >= 0.8 for p in similar)

def test_pattern_matching_with_subsequences():
    learner = SelfLearner()
    
    # Add some test patterns
    test_pattern = [
        {'type': 'test_setup'},
        {'type': 'run_test'},
        {'type': 'verify_result'}
    ]
    
    learner.learned_patterns[('test_setup', 'run_test', 'verify_result')] = [{
        'actions': test_pattern,
        'state_before': {},
        'success_rate': 0.9
    }]
    
    # Test finding subsequence matches
    matches = learner.get_similar_patterns(
        [{'type': 'test_setup'}, {'type': 'run_test'}],
        {}
    )
    
    assert len(matches) > 0  # Should find the larger pattern that contains this subsequence

import pytest
from ai_agent.core.learner import SelfLearner, LearningRate
from ai_agent.core.config import AgentConfig

def test_learning_rate_adaptation():
    rate = LearningRate(initial_rate=0.1)
    
    # Test success increases rate
    rate.update(True)
    assert rate.rate > 0.1
    assert rate.success_rate == 1.0
    
    # Test failure decreases rate
    rate.update(False)
    assert rate.rate < 0.11  # Should have decreased from previous value
    assert rate.success_rate == 0.5  # One success out of two attempts

def test_pattern_learning_with_rates():
    learner = SelfLearner()
    
    # Create test trajectory
    trajectory = type('Trajectory', (), {
        'instruction': "Add feature",
        'actions': [
            {'type': 'create_file', 'file': 'feature.py'},
            {'type': 'write_code', 'file': 'feature.py'}
        ],
        'observations': [
            {'status': 'success'},
            {'status': 'success'}
        ],
        'compute_quality_metrics': lambda self: type('Metrics', (), {'success_rate': 0.9})(),
        'state_changes': [
            type('StateChange', (), {
                'type': 'file_created',
                'path': 'feature.py',
                'impact': 0.5
            })()
        ]
    })()
    
    # Learn from trajectory multiple times
    refined = None
    for _ in range(5):
        result = learner.learn_from_trajectory(trajectory)
        if result:
            refined = result
    
    # Verify pattern was learned with increasing confidence
    pattern_key = learner._get_pattern_key(trajectory)
    assert learner.get_pattern_confidence(pattern_key) > 0.3
    
    # Verify refinement happened after high success rate
    assert refined is not None
    assert refined.instruction != trajectory.instruction

def test_exploration_vs_exploitation():
    learner = SelfLearner()
    
    # Create a successful pattern
    successful_trajectory = type('Trajectory', (), {
        'instruction': "Successful pattern",
        'actions': [{'type': 'successful_action'}],
        'observations': [{'status': 'success'}],
        'compute_quality_metrics': lambda self: type('Metrics', (), {'success_rate': 1.0})(),
        'state_changes': []
    })()
    
    # Learn successful pattern multiple times
    for _ in range(5):
        learner.learn_from_trajectory(successful_trajectory)
    
    # Test exploitation of successful pattern
    state = {'test': 'state'}
    plan = learner.generate_plan("Test instruction", state)
    
    # Should use successful pattern when confidence is high
    pattern_key = learner._get_pattern_key(successful_trajectory)
    assert learner.get_pattern_confidence(pattern_key) >= 0.7
    
    # Test exploration of new pattern
    new_state = {'different': 'state'}
    new_plan = learner.generate_plan("New instruction", new_state)
    
    # Should explore when no high-confidence pattern matches
    assert new_plan != successful_trajectory.actions

def test_pattern_confidence_thresholds():
    learner = SelfLearner()
    
    # Test new pattern
    new_pattern = "test:pattern"
    assert learner.should_explore_pattern(new_pattern)
    
    # Test pattern with mixed success
    rate = LearningRate()
    rate.update(True)
    rate.update(False)
    learner.learning_rates[new_pattern] = rate
    
    # Should explore if confidence is above minimum
    assert learner.should_explore_pattern(new_pattern) == (
        learner.get_pattern_confidence(new_pattern) >= 0.3
    )

from typing import List, Dict, Any
from unittest.mock import Mock
import numpy as np

class MockLearner:
    """Mock learner for testing"""
    def __init__(self):
        self.client = None  # No need for real OpenAI client
        self.mock_responses = {}

    def compute_embedding(self, text: str) -> List[float]:
        """Return mock embedding"""
        return [0.0] * 1536  # OpenAI embeddings are 1536-dimensional

    def plan_next_action(self, current_state: Dict[str, Any], instruction: str, 
                        history: List[Dict[str, Any]], examples: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return a mock action"""
        return {
            'type': 'edit_file',
            'path': 'test.py',
            'content': '# Mock content'
        }

    def _get_pattern_suggestions(self, instruction: str, current_state: Dict[str, Any], 
                               examples: List[Dict[str, Any]]) -> List[str]:
        """Return mock pattern suggestions"""
        return ['validation', 'input_handling']