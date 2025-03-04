from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set, TYPE_CHECKING, Tuple
import json
from pathlib import Path
import numpy as np
from openai import OpenAI
import os
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from unittest.mock import Mock
from ..data.sequence import ActionSequence, SequencePattern
from .config import AgentConfig
from .task_generator import TaskGenerator

# Only for type hints
if TYPE_CHECKING:
    from ..data.trajectory_manager import Trajectory, TrajectoryManager

@dataclass
class TaskExample:
    """Represents a generated task example with its instruction and context"""
    instruction: str
    context: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

import time

class LearningRate:
    def __init__(self):
        self.successes = 0
        self.attempts = 0
        self.last_update = time.time()
        self.decay_rate = 0.1

    def update(self, success: bool):
        """Update learning rate with new result"""
        self.attempts += 1
        if success:
            self.successes += 1
        self.last_update = time.time()

    @property
    def rate(self) -> float:
        """Get current success rate"""
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts

    @property
    def decay(self) -> float:
        """Get time-based decay factor"""
        time_since_update = time.time() - self.last_update
        return min(1.0, time_since_update * self.decay_rate)

class SelfLearner:
    def __init__(self, client=None, api_key=None):
        self.client = client
        self.api_key = api_key
        self.embedding_model = Mock()  # Initialize with Mock for testing
        self.embedding_model.encode.return_value = np.ones(384)  # Standard dimension
        self.learned_patterns = {}
        self.learning_rates = {}
        self.agent = None  # Will be set by agent
        self.system_context = "Expert coding agent with deep knowledge of software development best practices"
        self.risk_thresholds = {
            'edit_file': 0.6,  # Increased from 0.4
            'git_commit': 0.2,
            'git_checkout': 0.6,
            'run_tests': 0.1,
            'fix_permissions': 0.7,
            'resolve_conflict': 0.8
        }
        self.action_type_risks = {
            'edit_file': 0.4,
            'git_commit': 0.2,
            'git_checkout': 0.6,
            'run_tests': 0.1,
            'fix_permissions': 0.7,
            'resolve_conflict': 0.8,
            'delete_file': 0.9
        }

    def _group_related_actions(self, actions: List[Dict]) -> List[List[Dict]]:
        """Group related actions together"""
        if not actions:
            return []
            
        groups = []
        current_group = []
        
        for action in actions:
            if not current_group:
                current_group = [action]
                continue
                
            # Check if action is related to current group
            last_action = current_group[-1]
            is_related = False
            
            # Same file operations
            if ('file' in action and 'file' in last_action and 
                action['file'] == last_action['file']):
                is_related = True
                
            # Same path operations    
            elif ('path' in action and 'path' in last_action and 
                  action['path'] == last_action['path']):
                is_related = True
                
            # Sequential test operations
            elif (action['type'] == 'run_tests' and 
                  any(a['type'] in ['edit_file', 'fix_imports'] for a in current_group)):
                is_related = True
                
            if is_related:
                current_group.append(action)
            else:
                groups.append(current_group)
                current_group = [action]
                
        if current_group:
            groups.append(current_group)
            
        return groups

    def backward_construct(self, trajectory: 'Trajectory') -> str:
        """Construct specific instruction from trajectory"""
        if not trajectory.actions:
            return trajectory.instruction
            
        # Get more descriptive instruction based on actions
        actions = trajectory.actions
        descriptions = []
        
        for action in actions:
            if action['type'] == 'edit_file':
                descriptions.append(f"Implement {action.get('description', 'changes')} in {action.get('path', action.get('file', ''))}")
            elif action['type'] == 'create_file':
                descriptions.append(f"Create {action.get('path', action.get('file', ''))}")
            elif action['type'] == 'run_tests':
                descriptions.append("Run tests")
            else:
                name = action['type'].replace('_', ' ').title()
                target = action.get('path', action.get('file', ''))
                descriptions.append(f"{name} {target}")
                
        return " and ".join(descriptions)

    def learn_from_trajectory(self, trajectory: 'Trajectory') -> Optional['Trajectory']:
        """Learn from a successful trajectory execution"""
        if not trajectory or not trajectory.actions:
            return None
            
        # Only learn from high quality trajectories
        metrics = trajectory.compute_quality_metrics()
        if metrics.success_rate < 0.7:
            return None

        self._learn_patterns(trajectory)
        return self._refine_trajectory(trajectory)

    def _learn_patterns(self, trajectory: 'Trajectory') -> None:
        """Learn successful patterns from a trajectory"""
        try:
            metrics = trajectory.compute_quality_metrics()
            if metrics.success_rate < 0.7:  # Only learn from mostly successful trajectories
                return
                
            pattern_key = self._get_pattern_key(trajectory)
            if pattern_key not in self.learning_rates:
                self.learning_rates[pattern_key] = LearningRate()
                
            self.learning_rates[pattern_key].update(True)
            
            if pattern_key not in self.learned_patterns:
                self.learned_patterns[pattern_key] = {
                    'actions': trajectory.actions,
                    'success_rate': metrics.success_rate,
                    'count': 1,
                    'type': trajectory.actions[0]['type'] if trajectory.actions else None,
                    'state_impact': self._analyze_state_impact(trajectory)
                }
            else:
                pattern = self.learned_patterns[pattern_key]
                pattern['count'] += 1
                # Update rolling average of success rate
                pattern['success_rate'] = (pattern['success_rate'] * (pattern['count'] - 1) + 
                                         metrics.success_rate) / pattern['count']
                # Update state impact analysis
                new_impact = self._analyze_state_impact(trajectory)
                pattern['state_impact'] = {
                    k: (pattern['state_impact'].get(k, 0) * (pattern['count'] - 1) + 
                        new_impact.get(k, 0)) / pattern['count']
                    for k in set(pattern['state_impact']) | set(new_impact)
                }
        except Exception as e:
            print(f"Error learning patterns: {e}")

    def _analyze_state_impact(self, trajectory: 'Trajectory') -> Dict[str, float]:
        """Analyze how actions impact repository state"""
        impact = {
            'files_changed': 0.0,
            'test_coverage': 0.0,
            'error_fixes': 0.0,
            'complexity': 0.0
        }
        
        if not trajectory.actions:
            return impact
            
        # Analyze file changes
        changed_files = set()
        for action in trajectory.actions:
            if action.get('type') in ['edit_file', 'create_file', 'delete_file']:
                changed_files.add(action.get('path', action.get('file', '')))
        impact['files_changed'] = min(1.0, len(changed_files) / 5)  # Normalize
        
        # Analyze test coverage impact
        test_actions = sum(1 for a in trajectory.actions if 'test' in str(a.get('type', '')).lower())
        impact['test_coverage'] = min(1.0, test_actions / len(trajectory.actions))
        
        # Analyze error fixes
        fixes = sum(1 for a in trajectory.actions if 'fix' in str(a.get('type', '')).lower())
        impact['error_fixes'] = min(1.0, fixes / len(trajectory.actions))
        
        # Analyze complexity
        impact['complexity'] = min(1.0, len(trajectory.actions) / 10)  # Normalize
        
        return impact

    def _get_pattern_key(self, trajectory: 'Trajectory') -> str:
        """Generate unique key for pattern based on action sequence"""
        if not trajectory.actions:
            return ''
            
        # Create key from action types and targets
        key_parts = []
        for action in trajectory.actions:
            action_type = action.get('type', '')
            target = action.get('path', action.get('file', ''))
            key_parts.append(f"{action_type}:{target}")
            
        return '|'.join(key_parts)

    def get_pattern_confidence(self, pattern_key: str) -> float:
        """Get confidence level for a learned pattern"""
        if pattern_key not in self.learned_patterns:
            return 0.0
            
        pattern = self.learned_patterns[pattern_key]
        learning_rate = self.learning_rates.get(pattern_key)
        
        if not learning_rate:
            return 0.0
        
        # Combine success rate with learning rate
        base_confidence = pattern['success_rate'] * learning_rate.rate
        
        # Adjust based on number of observations
        observation_factor = min(1.0, pattern['count'] / 5)  # Saturate at 5 observations
        
        # Adjust based on state impact
        impact_score = sum(pattern['state_impact'].values()) / len(pattern['state_impact'])
        
        return base_confidence * observation_factor * (1 + impact_score) / 2

    def should_explore_pattern(self, pattern_key: str) -> bool:
        """Determine if a pattern needs more exploration"""
        if pattern_key not in self.learned_patterns:
            return True
            
        pattern = self.learned_patterns[pattern_key]
        learning_rate = self.learning_rates.get(pattern_key)
        
        if not learning_rate:
            return True
            
        # Explore if we have few observations
        if pattern['count'] < 3:
            return True
            
        # Explore if success rate is promising but learning rate is low
        if pattern['success_rate'] > 0.7 and learning_rate.rate < 0.5:
            return True
            
        # Explore if we haven't seen the pattern recently
        if learning_rate.confidence < 0.3:
            return True
            
        return False

    def generate_tasks_from_docs(self, docs: List[str], repo_state: Dict) -> List['Task']:
        """Generate tasks by analyzing documentation and current repository state"""
        tasks = []
        embedding_batch_size = 32
        
        # First extract potential tasks from documentation
        base_tasks = self._extract_tasks_from_docs(docs)
        
        # Generate task variations based on tech context
        all_tasks = []
        for task in base_tasks:
            variations = self._generate_task_variations(task, repo_state)
            all_tasks.extend(variations)
        
        # Create full task objects with embeddings
        for i in range(0, len(all_tasks), embedding_batch_size):
            batch = all_tasks[i:i + embedding_batch_size]
            embeddings = self._compute_embeddings([t['instruction'] for t in batch])
            
            for task_dict, embedding in zip(batch, embeddings):
                task = self._create_task(
                    instruction=task_dict['instruction'],
                    context=task_dict['context']
                )
                task.embedding = embedding
                tasks.append(task)
        
        return tasks

    def _extract_tasks_from_docs(self, docs: List[str]) -> List[Dict]:
        """Extract potential tasks from documentation"""
        tasks = []
        
        for doc in docs:
            # Use task-oriented prompts to identify potential tasks
            response = self.chat_completion([
                {"role": "system", "content": (
                    "Identify concrete coding tasks from the documentation. "
                    "Focus on actionable items that modify or create code. "
                    "Return a JSON array of {instruction, type} objects."
                )},
                {"role": "user", "content": doc}
            ])
            
            try:
                doc_tasks = json.loads(response.choices[0].message.content)
                tasks.extend(doc_tasks)
            except:
                continue
        
        return tasks

    def _generate_task_variations(self, base_task: str, tech_context: Dict) -> List[Dict]:
        """Generate task variations based on technical context"""
        variations = []
        
        # Create context-specific variations
        frameworks = tech_context.get('frameworks', [])
        languages = tech_context.get('languages', [])
        patterns = tech_context.get('patterns', [])
        
        response = self.chat_completion([
            {"role": "system", "content": (
                "Generate variations of the coding task that are specific to the "
                "technical context. Focus on framework-specific implementations "
                "and common patterns. Return a JSON array of {instruction, context} objects."
            )},
            {"role": "user", "content": json.dumps({
                "task": base_task,
                "frameworks": frameworks,
                "languages": languages,
                "patterns": patterns
            })}
        ])
        
        try:
            variations = json.loads(response.choices[0].message.content)
        except:
            # Fallback to base task if variation generation fails
            variations = [{
                'instruction': base_task,
                'context': tech_context
            }]
        
        return variations

    def _create_task(self, instruction: str, context: Dict) -> 'Task':
        """Create a task with properly formatted instruction and context"""
        # Replace generic terms with context-specific ones
        frameworks = context.get('frameworks', [])
        if frameworks:
            # Add framework-specific prefixes
            prefix = f"Using {frameworks[0]}, " if frameworks else ""
            instruction = prefix + instruction
            
        # Add relevant file patterns
        file_patterns = context.get('file_patterns', {})
        if file_patterns:
            for pattern, examples in file_patterns.items():
                instruction = instruction.replace(pattern, examples[0])
                
        return TaskExample(
            instruction=instruction,
            context=context,
            embedding=None  # Will be set later in batch
        )

    def bootstrap_learning(self, docs_path: str, framework_info: Dict,
                         file_patterns: Dict, trajectory_manager) -> int:
        """Bootstrap learning from documentation"""
        success_count = 0
        
        # Read documentation files
        docs = []
        docs_dir = Path(docs_path)
        if docs_dir.exists():
            for file in docs_dir.glob('**/*.md'):
                docs.append(file.read_text())
                
        # Generate initial tasks
        tasks = []
        tasks.extend(self.generate_tasks_from_docs(docs, framework_info))
        
        # Add framework-specific tasks
        if 'pytest' in framework_info.get('frameworks', []):
            tasks.append(self._create_task(
                "Add pytest fixtures for database testing",
                framework_info
            ))
            tasks.append(self._create_task(
                "Implement test coverage reporting",
                framework_info
            ))
            
        # Add tasks for complex files
        for pattern_type, files in file_patterns.items():
            for file in files:
                tasks.append(self._create_task(
                    f"Simplify complex functions in {file}",
                    framework_info
                ))
                
        # Execute tasks and learn
        for task in tasks:
            try:
                result = self.agent.execute_task(task.instruction)
                if result.get('status') == 'success':
                    success_count += 1
                    trajectory_manager.store_trajectory(result['trajectory'])
            except Exception as e:
                print(f"Error executing task '{task.instruction}': {str(e)}")
                
        return success_count

    def _analyze_task_context(self, instruction: str, state: Dict) -> Dict:
        """Analyze task context to determine requirements and risks"""
        prompt = f"""Analyze this task and context:
Task: {instruction}
Current state: {state}

Return a JSON object with:
- required_operations: list of required file operations
- dependencies: list of required dependencies
- risk_factors: list of potential risks
"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "system", "content": prompt}]
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            return {'required_operations': [], 'dependencies': [], 'risk_factors': []}

    def _evaluate_action_risk(self, action: Dict, context: Dict) -> float:
        """Evaluate risk level of an action"""
        if not isinstance(action, dict):
            return 0.5
            
        base_risk = self.action_type_risks.get(action.get('type', ''), 0.5)
        
        # Adjust for context risk factors
        risk_factors = context.get('risk_factors', [])
        for factor in risk_factors:
            if isinstance(factor, dict):  
                severity = factor.get('severity', 0)
                base_risk = max(base_risk, severity)

        return base_risk

    def _validate_action(self, action: Dict, context: Dict) -> bool:
        """Validate action against context and constraints"""
        if not isinstance(action, dict) or 'type' not in action:
            return False
            
        action_type = action['type']
        
        # Validate file operations
        if action_type in ['edit_file', 'delete_file']:
            if 'path' not in action and 'file' not in action:
                return False
                
        # Validate git operations
        if action_type.startswith('git_'):
            # Don't allow git operations during merge conflicts
            if context.get('git_status', {}).get('merge_conflicts'):
                return False
                
        # Validate state requirements
        if action_type == 'run_tests':
            # Only allow if tests exist
            if not any('test' in f for f in context.get('files', [])):
                return False
                
        # Check required operations
        required_ops = context.get('required_operations', [])
        for op in required_ops:
            if isinstance(op, dict) and op.get('file') == action.get('path'):
                if op.get('operation') != action.get('type'):
                    return False

        # Check risk level
        risk = self._evaluate_action_risk(action, context)
        if risk > 0.8:  # High-risk threshold
            return False

        return True

    def plan_next_action(self, current_state: Dict, instruction: str, 
                        history: List[Dict], examples: List[Dict] = None) -> Dict:
        """Plan next action based on current state, instruction and history"""
        if not self.client:
            raise ValueError("OpenAI client not initialized")

        # Get pattern suggestions first
        pattern_suggestions = self._get_pattern_suggestions(
            instruction, current_state, examples or []
        )

        # Build rich context for decision
        context = {
            'instruction': instruction,
            'current_state': current_state,
            'action_history': history,
            'pattern_suggestions': pattern_suggestions,
            'successful_examples': examples
        }

        # Get next action recommendation
        response = self.chat_completion([
            {"role": "system", "content": (
                "You are a coding agent that decides the next action to take. "
                "Consider the current state, history, and learned patterns to "
                "suggest the optimal next step that moves closer to the goal."
            )},
            {"role": "user", "content": json.dumps(context)}
        ])

        try:
            action = json.loads(response.choices[0].message.content)
            # Validate and assess risk of suggested action
            if self._validate_action(action, current_state):
                risk = self._evaluate_action_risk(action, current_state)
                if risk < 0.8:  # Only proceed if risk is acceptable
                    return action
        except:
            pass

        # Fallback to most relevant pattern suggestion if available
        if pattern_suggestions:
            return pattern_suggestions[0]

        # Ultimate fallback - safe default action
        return {
            'type': 'analyze',
            'description': 'Analyzing current state to determine next step'
        }

    def _get_pattern_suggestions(self, instruction: str, current_state: Dict, 
                               examples: List[Dict]) -> List[Dict]:
        """Get relevant pattern suggestions based on current context"""
        suggestions = []
        
        # First check exact pattern matches
        for pattern_key, pattern in self.learned_patterns.items():
            if pattern['success_rate'] >= 0.8:
                # Check if pattern matches current state
                if self._pattern_matches_state(pattern['actions'], current_state):
                    confidence = self.get_pattern_confidence(pattern_key)
                    suggestions.append({
                        'actions': pattern['actions'],
                        'confidence': confidence,
                        'type': pattern['type']
                    })
        
        # Then check similar examples
        for example in examples:
            if example.get('success_rate', 0) >= 0.8:
                similarity = self._calculate_semantic_similarity(
                    instruction, example.get('instruction', '')
                )
                if similarity > 0.7:
                    suggestions.append({
                        'actions': example['actions'],
                        'confidence': similarity * 0.8,  # Slightly lower confidence for similar examples
                        'type': example['actions'][0]['type'] if example['actions'] else None
                    })
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Convert to concrete actions
        return [
            self._adapt_pattern_to_state(s['actions'][0], current_state)
            for s in suggestions[:3]  # Take top 3 suggestions
        ]

    def _pattern_matches_state(self, pattern_actions: List[Dict], state: Dict) -> bool:
        """Check if a pattern is applicable to current state"""
        if not pattern_actions:
            return False
            
        # Check file existence for file operations
        first_action = pattern_actions[0]
        if first_action['type'] in ['edit_file', 'delete_file']:
            target = first_action.get('path', first_action.get('file'))
            if target and target not in state.get('files', []):
                return False
                
        # Check for merge conflicts if pattern involves git operations
        if any(a['type'].startswith('git_') for a in pattern_actions):
            if state.get('git_status', {}).get('merge_conflicts'):
                return False
                
        return True

    def _adapt_pattern_to_state(self, pattern_action: Dict, state: Dict) -> Dict:
        """Adapt a pattern action to current state"""
        action = pattern_action.copy()
        
        # Adapt file paths if needed
        if 'path' in action:
            if action['path'] not in state.get('files', []):
                # Try to find similar file
                similar_files = [
                    f for f in state.get('files', [])
                    if Path(f).suffix == Path(action['path']).suffix
                ]
                if similar_files:
                    action['path'] = similar_files[0]
                    
        # Update description
        action['description'] = f"Applying learned pattern: {action['type']}"
        
        return action

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        embeddings = self._compute_embeddings([text1, text2])
        if len(embeddings) != 2:
            return 0.0
        
        # Compute cosine similarity
        dot_product = np.dot(embeddings[0], embeddings[1])
        norm1 = np.linalg.norm(embeddings[0])
        norm2 = np.linalg.norm(embeddings[1])
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)

    def _compute_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Compute embeddings for a list of texts"""
        embeddings = []
        for text in texts:
            if self.client and hasattr(self.client, 'embeddings'):
                response = self.client.embeddings.create(input=text, model="text-embedding-ada-002")
                embeddings.append(np.array(response.data[0].embedding))
            else:
                # Fallback to simple TF-IDF-like embedding
                words = set(text.lower().split())
                embedding = np.zeros(384)  # Same dimension as all-MiniLM-L6-v2
                for i, word in enumerate(words):
                    embedding[hash(word) % 384] = 1
                embeddings.append(embedding)
        return embeddings

    def chat_completion(self, messages: List[Dict]) -> Dict:
        """Execute chat completion with enhanced system context"""
        if messages and messages[0]['role'] == 'system':
            messages[0]['content'] = f"{self.system_context}\n{messages[0]['content']}"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            return response
        except Exception:
            return {'choices': [{'message': {'content': '{}'}}]}

    def generate_plan(self, instruction: str, state: Dict) -> List[Dict]:
        """Generate execution plan for instruction"""
        context = self._analyze_task_context(instruction, state)
        
        messages = [
            {"role": "system", "content": "Generate an action plan for this task."},
            {"role": "user", "content": f"Task: {instruction}\nContext: {context}"}
        ]
        
        try:
            response = self.chat_completion(messages)
            return json.loads(response.choices[0].message.content)
        except Exception:
            return []

    def _get_pattern_suggestions(self, instruction: str, current_state: Dict, 
                               examples: List[Dict]) -> List[Dict]:
        """Get pattern suggestions from examples"""
        suggestions = []
        
        # Extract target module name from instruction
        words = instruction.lower().split()
        target = next((w for w in words if 'module' in w or '.py' in w), None)
        if not target:
            target = next((w for w in words if w not in ['create', 'new', 'add', 'implement']), 'module')
        if not target.endswith('.py'):
            target = f"{target}.py"
            
        for example in examples:
            if any(word in instruction.lower() for word in example['instruction'].lower().split()):
                # Copy action but replace path with appropriate one based on instruction
                new_actions = []
                for action in example['actions']:
                    new_action = action.copy()
                    if 'path' in action:
                        new_action['path'] = action['path'].replace(
                            action['path'].split('/')[-1],
                            target
                        )
                    new_actions.append(new_action)
                
                pattern = {
                    'pattern_type': example['actions'][0]['type'],
                    'success_rate': 0.9,
                    'suggested_actions': new_actions
                }
                suggestions.append(pattern)
                
        return suggestions

    def refine_instruction_library(self, trajectory_manager) -> int:
        """Refine instructions in trajectory library"""
        refined_count = 0
        
        for trajectory in trajectory_manager.trajectories:
            if not hasattr(trajectory, 'quality_metrics'):
                continue
                
            if trajectory.quality_metrics.get('success_rate', 0) >= 0.8:
                refined = self._refine_trajectory(trajectory)
                if refined:
                    refined_count += 1
                    trajectory_manager.store_trajectory(refined)
                    
        return refined_count

    def _refine_trajectory(self, trajectory: 'Trajectory') -> Optional['Trajectory']:
        """Refine a trajectory based on learned patterns and success metrics"""
        if not trajectory or not trajectory.actions:
            return None
            
        # Only refine successful trajectories
        metrics = trajectory.compute_quality_metrics()
        if metrics.success_rate < 0.7:
            return None
            
        # Get similar successful patterns
        pattern_key = self._get_pattern_key(trajectory)
        similar_patterns = []
        
        for key, pattern in self.learned_patterns.items():
            if key != pattern_key and pattern['success_rate'] >= 0.8:
                similarity = self._calculate_pattern_similarity(
                    trajectory.actions, pattern['actions']
                )
                if similarity > 0.6:
                    similar_patterns.append(pattern)
                    
        if not similar_patterns:
            return None
            
        # Create refined trajectory combining successful elements
        refined_actions = self._merge_pattern_actions(
            trajectory.actions,
            [p['actions'] for p in similar_patterns]
        )
        
        if refined_actions == trajectory.actions:
            return None
            
        return type('Trajectory', (), {
            'instruction': trajectory.instruction,
            'actions': refined_actions,
            'observations': trajectory.observations,
            'final_state': trajectory.final_state,
            'compute_quality_metrics': trajectory.compute_quality_metrics
        })()

    def _calculate_pattern_similarity(self, actions1: List[Dict], actions2: List[Dict]) -> float:
        """Calculate similarity between two action sequences"""
        if not actions1 or not actions2:
            return 0.0
            
        # Compare action types and targets
        matches = 0
        total = max(len(actions1), len(actions2))
        
        for i in range(min(len(actions1), len(actions2))):
            if actions1[i].get('type') == actions2[i].get('type'):
                matches += 0.7  # Type match
                if actions1[i].get('path') == actions2[i].get('path'):
                    matches += 0.3  # Target match
                    
        return matches / total

    def _merge_pattern_actions(self, base_actions: List[Dict], 
                             pattern_actions_list: List[List[Dict]]) -> List[Dict]:
        """Merge multiple action patterns to create an optimized sequence"""
        if not pattern_actions_list:
            return base_actions
            
        # Track action frequencies and success patterns
        action_stats = {}
        
        # Analyze base actions
        for i, action in enumerate(base_actions):
            key = f"{action.get('type')}:{action.get('path', '')}"
            if key not in action_stats:
                action_stats[key] = {
                    'count': 1,
                    'positions': [i],
                    'action': action
                }
            else:
                action_stats[key]['count'] += 1
                action_stats[key]['positions'].append(i)
                
        # Analyze pattern actions
        for pattern_actions in pattern_actions_list:
            for i, action in enumerate(pattern_actions):
                key = f"{action.get('type')}:{action.get('path', '')}"
                if key not in action_stats:
                    action_stats[key] = {
                        'count': 1,
                        'positions': [i],
                        'action': action
                    }
                else:
                    action_stats[key]['count'] += 1
                    action_stats[key]['positions'].append(i)
                    
        # Build optimized sequence
        refined_actions = []
        used_keys = set()
        
        # First add high-frequency actions in their most common positions
        for key, stats in sorted(
            action_stats.items(), 
            key=lambda x: (-x[1]['count'], sum(x[1]['positions'])/len(x[1]['positions']))
        ):
            if key not in used_keys:
                avg_pos = sum(stats['positions']) / len(stats['positions'])
                # Insert action at average position if reasonable
                pos = min(int(avg_pos), len(refined_actions))
                refined_actions.insert(pos, stats['action'])
                used_keys.add(key)
                
        # Fill in remaining actions from base sequence to maintain functionality
        for action in base_actions:
            key = f"{action.get('type')}:{action.get('path', '')}"
            if key not in used_keys:
                refined_actions.append(action)
                used_keys.add(key)
                
        return refined_actions