import pytest
from ai_agent.data.state_analyzer import StateChangeAnalyzer, StateChange
import time

def test_state_change_analysis():
    analyzer = StateChangeAnalyzer()
    
    # Test file changes
    state_before = {
        'files': ['test.py', 'main.py'],
        'file_hashes': {
            'test.py': 'abc123',
            'main.py': 'def456'
        }
    }
    
    state_after = {
        'files': ['test.py', 'new.py'],
        'file_hashes': {
            'test.py': 'abc123',
            'new.py': 'xyz789'
        }
    }
    
    changes = analyzer.analyze_change(state_before, state_after)
    
    # Verify changes detected
    assert len(changes) == 2
    assert any(c.type == 'file_created' and c.path == 'new.py' for c in changes)
    assert any(c.type == 'file_deleted' and c.path == 'main.py' for c in changes)

def test_impact_calculation():
    analyzer = StateChangeAnalyzer()
    
    # Test file impact scores
    high_impact = analyzer._compute_file_impact('critical.py')
    medium_impact = analyzer._compute_file_impact('readme.md')
    low_impact = analyzer._compute_file_impact('test.log')
    
    assert high_impact > medium_impact > low_impact
    assert 0 <= high_impact <= 1
    
    # Test git impact scores
    git_before = {'commit': 'abc', 'branch': 'main', 'staged': []}
    git_after = {'commit': 'def', 'branch': 'feature', 'staged': ['file.py']}
    
    impact = analyzer._compute_git_impact(git_before, git_after)
    assert impact > 0.5  # Significant changes should have high impact

def test_pattern_tracking():
    analyzer = StateChangeAnalyzer()
    
    # Create repeated pattern
    changes = [
        StateChange('file_modified', 'test.py', {'hash': 'old'}, {'hash': 'new'}, 0.8),
        StateChange('file_created', 'new.py', None, {'exists': True}, 0.6)
    ]
    
    # Add pattern multiple times
    for _ in range(3):
        analyzer._update_patterns(changes)
        time.sleep(0.1)  # Ensure different timestamps
    
    # Verify pattern recognition
    similar = analyzer.get_similar_changes(changes)
    assert len(similar) >= 3
    
    # Verify safe paths learning
    assert analyzer.is_safe_path('test.py')

def test_state_history():
    analyzer = StateChangeAnalyzer()
    
    # Add multiple states
    states = [
        {'files': ['a.py']},
        {'files': ['a.py', 'b.py']},
        {'files': ['a.py', 'b.py', 'c.py']}
    ]
    
    for state in states:
        analyzer._update_history(state)
        
    # Test retrieving history window
    recent = analyzer.get_recent_states(2)
    assert len(recent) == 2
    assert recent[-1] == states[-1]
    assert recent[0] == states[-2]
    
    # Test history limit
    for i in range(100):  # Add many states
        analyzer._update_history({'files': [f'file{i}.py']})
    assert len(analyzer.state_history) <= analyzer.MAX_HISTORY_SIZE

def test_pattern_similarity():
    analyzer = StateChangeAnalyzer()
    
    # Create test patterns
    pattern1 = [
        StateChange('file_modified', 'test.py', {'old': 1}, {'new': 2}, 0.5),
        StateChange('file_created', 'new.py', None, {'exists': True}, 0.3)
    ]
    
    pattern2 = [
        StateChange('file_modified', 'test.py', {'old': 3}, {'new': 4}, 0.5),
        StateChange('file_created', 'other.py', None, {'exists': True}, 0.3)
    ]
    
    pattern3 = [
        StateChange('file_deleted', 'old.py', {'exists': True}, None, 0.7),
        StateChange('file_moved', 'src.py', {'path': 'old'}, {'path': 'new'}, 0.4)
    ]
    
    # Test similarity calculations
    assert analyzer._compute_pattern_similarity(pattern1, pattern2) > 0.7  # Similar patterns
    assert analyzer._compute_pattern_similarity(pattern1, pattern3) < 0.3  # Different patterns
    
    # Test with empty patterns
    assert analyzer._compute_pattern_similarity([], []) == 1.0
    assert analyzer._compute_pattern_similarity(pattern1, []) == 0.0

def test_impact_aggregation():
    analyzer = StateChangeAnalyzer()
    
    # Create a set of changes
    changes = [
        StateChange('file_modified', 'core.py', {'hash': 'old'}, {'hash': 'new'}, 0.9),
        StateChange('file_created', 'utils.py', None, {'exists': True}, 0.5),
        StateChange('file_deleted', 'temp.py', {'exists': True}, None, 0.2)
    ]
    
    # Test total impact calculation
    total_impact = analyzer.compute_total_impact(changes)
    assert 0 <= total_impact <= 1
    
    # High impact changes should have more weight
    high_impact = [StateChange('file_modified', 'core.py', {}, {}, 0.9)]
    low_impact = [StateChange('file_modified', 'test.log', {}, {}, 0.1)]
    
    assert analyzer.compute_total_impact(high_impact) > analyzer.compute_total_impact(low_impact)