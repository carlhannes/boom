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
    assert 'test.py' in analyzer.safe_paths

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
    
    # Verify history maintenance
    assert len(analyzer.state_history) == len(states)
    assert analyzer.state_history[-1] == states[-1]

def test_pattern_similarity():
    analyzer = StateChangeAnalyzer()
    
    pattern1 = (('file_modified', True), ('file_created', False))
    pattern2 = (('file_modified', True), ('file_created', False))
    pattern3 = (('file_deleted', True), ('file_moved', True))
    
    # Test exact match
    assert analyzer._pattern_similarity(pattern1, pattern2) == 1.0
    
    # Test different patterns
    assert analyzer._pattern_similarity(pattern1, pattern3) < 0.5