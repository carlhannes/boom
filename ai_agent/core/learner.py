from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set
import json
from pathlib import Path
import numpy as np
from openai import OpenAI
import os
from sentence_transformers import SentenceTransformer

@dataclass
class TaskExample:
    """Represents a generated task example with its instruction and context"""
    instruction: str
    context: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

class SelfLearner:
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None, client=None, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize SelfLearner with optional API key for testing
        
        Args:
            model: The LLM model to use for task generation and planning
            api_key: Optional OpenAI API key for testing
            client: Optional pre-configured OpenAI client
            embedding_model: Model to use for embeddings, defaults to all-MiniLM-L6-v2
        """
        self.model = model
        if client is not None:
            self.client = client
        else:
            try:
                self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            except Exception as e:
                # For testing without API key
                if api_key == "mock-key":
                    self.client = None
                else:
                    raise e
                    
        # Initialize the embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
        except Exception as e:
            if api_key == "mock-key":
                self.embedding_model = None
            else:
                raise e

    def _analyze_repository_state(self, repo_files: List[str], git_status: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze repository state to inform task generation"""
        analysis = {
            'file_types': set(),
            'frameworks': set(),
            'languages': set(),
            'modified_files': git_status.get('modified', []),
            'patterns': []
        }
        
        # Analyze file types and tech stack
        for file in repo_files:
            ext = Path(file).suffix
            if ext:
                analysis['file_types'].add(ext[1:])  # Remove dot
                
            # Detect frameworks/languages from file patterns
            if ext in ['.js', '.jsx', '.ts', '.tsx']:
                if any(f.endswith('package.json') for f in repo_files):
                    analysis['frameworks'].add('node')
                if any('react' in f.lower() for f in repo_files):
                    analysis['frameworks'].add('react')
            elif ext == '.py':
                analysis['languages'].add('python')
                if any(f.endswith('requirements.txt') for f in repo_files):
                    analysis['patterns'].append('python_package')
                    
        return analysis

    def _extract_technical_context(self, docs: List[str], repo_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract technical details from documentation and repository analysis"""
        context_prompt = f"""Analyze this documentation and repository state to extract technical context:

Documentation:
{chr(10).join(docs)}

Repository Analysis:
- File types: {', '.join(repo_analysis['file_types'])}
- Frameworks: {', '.join(repo_analysis['frameworks'])}
- Languages: {', '.join(repo_analysis['languages'])}
- Patterns: {', '.join(repo_analysis['patterns'])}

Extract:
1. Main technologies
2. Architecture patterns
3. Development practices
4. Testing approaches
5. Key technical requirements"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a technical analyst extracting structured information from documentation and code repositories."},
                {"role": "user", "content": context_prompt}
            ]
        )
        
        # Parse the structured response
        return json.loads(response.choices[0].message.content)

    def _generate_task_variations(self, base_task: str, tech_context: Dict[str, Any]) -> List[str]:
        """Generate variations of a task considering different technical approaches"""
        variation_prompt = f"""Generate 3 specific variations of this coding task:
Task: {base_task}

Technical Context:
{json.dumps(tech_context, indent=2)}

Generate variations that:
1. Use different technical approaches
2. Consider different edge cases
3. Apply different patterns or practices
4. Have different levels of complexity

Return as a JSON array of strings."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a technical lead generating specific task variations for developers."},
                {"role": "user", "content": variation_prompt}
            ]
        )
        
        return json.loads(response.choices[0].message.content)

    def _filter_tasks(self, tasks: List[TaskExample], repo_analysis: Dict[str, Any]) -> List[TaskExample]:
        """Filter and validate generated tasks"""
        filtered = []
        seen_instructions = set()
        
        for task in tasks:
            instruction = task.instruction.lower().strip()
            
            # Skip duplicate or very similar tasks
            if instruction in seen_instructions:
                continue
                
            # Validate task matches repository context
            matches_context = any(
                tech in instruction or tech in str(task.context)
                for tech in repo_analysis['frameworks'] | repo_analysis['languages']
            )
            
            if matches_context:
                seen_instructions.add(instruction)
                filtered.append(task)
                
        return filtered

    def generate_tasks_from_docs(self, docs: List[str], repo_state: Optional[Dict[str, Any]] = None) -> List[TaskExample]:
        """Generate tasks by analyzing documentation and repository state
        
        Args:
            docs: List of documentation strings to analyze
            repo_state: Optional repository state containing files and git status
        """
        if repo_state is None:
            repo_state = {'files': [], 'git_status': {}}
            
        # Analyze repository state
        repo_analysis = self._analyze_repository_state(
            repo_state['files'],
            repo_state['git_status']
        )
        
        # Extract technical context
        tech_context = self._extract_technical_context(docs, repo_analysis)
        
        # Generate initial tasks
        tasks = []
        for doc in docs:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": (
                        "You are a coding agent that generates realistic coding tasks "
                        "from documentation. Generate specific, actionable tasks that "
                        "could be performed in a codebase."
                    )},
                    {"role": "user", "content": f"""Generate 3 specific coding tasks based on this documentation and technical context:

Documentation:
{doc}

Technical Context:
{json.dumps(tech_context, indent=2)}

Tasks should be:
1. Specific and actionable
2. Aligned with the technical stack
3. Following identified patterns
4. Consider edge cases
5. Include testing/validation"""}
                ]
            )
            
            # Parse base tasks
            task_text = response.choices[0].message.content
            for line in task_text.split('\n'):
                if line.strip():
                    # Generate variations for each base task
                    variations = self._generate_task_variations(line.strip(), tech_context)
                    
                    # Create task examples with context
                    for task in [line.strip()] + variations:
                        tasks.append(TaskExample(
                            instruction=task,
                            context={
                                "source_doc": doc,
                                "tech_context": tech_context,
                                "repo_analysis": repo_analysis
                            }
                        ))
        
        # Filter and validate tasks
        return self._filter_tasks(tasks, repo_analysis)
    
    def backward_construct(self, trajectory: Dict[str, Any]) -> str:
        """
        Given a trajectory of actions and observations, construct a precise
        instruction that matches what actually happened. Uses error states and similar
        trajectories to generate more accurate instructions.
        """
        # Format actions with their outcomes
        action_outcomes = []
        for i, (action, obs) in enumerate(zip(trajectory['actions'], trajectory['observations'])):
            outcome = "success" if isinstance(obs, dict) and obs.get('status') == 'success' else "failed"
            desc = action.get('description', '')
            action_outcomes.append(f"Action {i+1}: {action['type']} - {desc} ({outcome})")
        
        actions_text = "\n".join(action_outcomes)
        
        # Analyze execution patterns from similar trajectories
        similar_patterns = []
        if 'similar_trajectories' in trajectory:
            for t in trajectory['similar_trajectories']:
                if all(o.get('status') == 'success' for o in t.get('observations', [])):
                    similar_patterns.append({
                        'instruction': t['instruction'],
                        'action_count': len(t['actions']),
                        'success': True
                    })
        
        # Extract success patterns
        success_patterns = [
            p for p in similar_patterns 
            if p['success'] and p['action_count'] == len(trajectory['actions'])
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": (
                    "You are a coding agent that writes precise instructions "
                    "based on actual sequences of actions taken in a codebase. "
                    "Write a single, specific instruction that accurately describes "
                    "the sequence of actions, focusing on what was actually done.\n\n"
                    "Consider:\n"
                    "1. Action outcomes (success/failure)\n"
                    "2. Similar successful trajectories\n"
                    "3. Technical accuracy and specificity\n"
                    "4. Edge cases and error handling"
                )},
                {"role": "user", "content": (
                    f"Original instruction: {trajectory['instruction']}\n\n"
                    f"Actual actions and outcomes:\n{actions_text}\n\n"
                    "Similar successful patterns:\n" + 
                    ("\n".join(f"- {p['instruction']}" for p in success_patterns) if success_patterns else "None") +
                    "\n\nConstruct a precise instruction that describes what was actually accomplished."
                )}
            ]
        )
        
        # Extract and validate the constructed instruction
        new_instruction = response.choices[0].message.content.strip()
        
        # If the trajectory had failures, reflect that in the instruction
        if any(isinstance(obs, dict) and obs.get('error') for obs in trajectory['observations']):
            new_instruction = f"Attempt to {new_instruction} (partially completed with errors)"
            
        return new_instruction
    
    def compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for a text using Sentence Transformers model"""
        if self.embedding_model is None:
            # For testing scenarios
            return np.ones(384)  # Default embedding dimension for all-MiniLM-L6-v2
            
        # Get embeddings using Sentence Transformers
        return self.embedding_model.encode(text, convert_to_numpy=True)
    
    def retrieve_similar_trajectories(self,
                                   query: str,
                                   trajectories: List[Dict[str, Any]],
                                   k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve k most similar trajectories using embedding similarity
        """
        if not trajectories:
            return []
            
        query_embedding = self.compute_embedding(query)
        
        # Compute similarities
        similarities = []
        for traj in trajectories:
            if 'embedding' not in traj:
                traj['embedding'] = self.compute_embedding(traj['instruction'])
            
            similarity = np.dot(query_embedding, traj['embedding'])
            similarities.append((similarity, traj))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [traj for _, traj in similarities[:k]]

    def chat_completion(self, messages: List[Dict[str, str]]) -> Any:
        """
        Execute a chat completion with enhanced context understanding and task planning
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
        """
        # Add system context about code-specific understanding
        base_context = {
            "role": "system",
            "content": (
                "You are an expert coding agent specialized in understanding and "
                "executing coding tasks in Git repositories. Consider:\n"
                "1. Code patterns and best practices\n"
                "2. Repository structure and state\n"
                "3. Technical context and frameworks\n"
                "4. Error handling and recovery\n"
                "5. Testing and validation"
            )
        }
        
        # Ensure we preserve the system message if it exists
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = base_context["content"] + "\n\n" + messages[0]["content"]
        else:
            messages.insert(0, base_context)
            
        # Execute completion
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,  # Balance between creativity and consistency
            max_tokens=1000,  # Generous limit for complex responses
            top_p=0.95,      # Focus on more likely tokens
            presence_penalty=0.2,  # Slight penalty to avoid repetition
            frequency_penalty=0.2   # Slight penalty for too-frequent tokens
        )
        
        return response

    def _analyze_task_context(self, task: str, repo_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a task and repository state to extract relevant context for planning
        """
        # Prepare analysis prompt
        analysis_prompt = f"""Analyze this coding task and repository state to extract relevant context:

Task: {task}

Repository State:
{json.dumps(repo_state, indent=2)}

Extract:
1. Required file operations
2. Dependencies and frameworks needed
3. Potential error cases
4. Testing requirements
5. Success criteria
6. Risk factors

Return as JSON object."""

        response = self.chat_completion([
            {"role": "system", "content": "You are a technical analyst extracting structured context from coding tasks."},
            {"role": "user", "content": analysis_prompt}
        ])
        
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {}

    def _evaluate_action_risk(self, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        """
        Evaluate the risk level of an action (0-1 scale)
        """
        risk_score = 0.0
        
        # Action type risks
        type_risks = {
            'edit_file': 0.4,      # Moderate risk - file changes
            'git_commit': 0.2,     # Lower risk - reversible
            'git_checkout': 0.6,   # Higher risk - branch changes
            'run_tests': 0.1,      # Very low risk
            'fix_permissions': 0.7, # High risk - system changes
            'resolve_conflict': 0.8 # High risk - data loss possible
        }
        
        risk_score += type_risks.get(action.get('type', ''), 0.5)
        
        # Content-based risks
        if action.get('type') == 'edit_file':
            # Check for sensitive operations
            content = str(action.get('content', '')).lower()
            if 'delete' in content or 'remove' in content:
                risk_score += 0.2
            if 'password' in content or 'secret' in content:
                risk_score += 0.3
                
        # Context-based adjustments
        if context.get('risk_factors'):
            for factor in context['risk_factors']:
                if isinstance(factor, dict) and factor.get('severity'):
                    risk_score += float(factor['severity']) * 0.1
                    
        return min(1.0, risk_score)

    def _validate_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        Validate an action against task context and constraints
        """
        # Basic action structure validation
        if not isinstance(action, dict) or 'type' not in action:
            return False
            
        # Action type-specific validation
        if action['type'] == 'edit_file':
            if not action.get('path') or not action.get('content'):
                return False
            # Validate against required operations
            if context.get('required_operations'):
                file_ops = [op.get('file') for op in context['required_operations']]
                if action['path'] not in file_ops:
                    return False
                    
        elif action['type'] == 'git_commit':
            if not action.get('message'):
                return False
                
        elif action['type'] == 'git_checkout':
            if not action.get('branch'):
                return False
                
        # Check risk level
        risk_score = self._evaluate_action_risk(action, context)
        if risk_score > 0.8:  # High-risk threshold
            return False
            
        return True

    def _get_pattern_suggestions(self, task: str, current_state: Dict[str, Any], examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract action suggestions from successful patterns in example trajectories
        """
        # Convert examples to sequences
        sequences = []
        for example in examples:
            try:
                sequence = ActionSequence.from_trajectory(example)
                if sequence.success_rate > 0.8:
                    sequences.append(sequence)
            except:
                continue
        
        # Group by semantic type
        type_patterns = {}
        for seq in sequences:
            if seq.semantic_type not in type_patterns:
                type_patterns[seq.semantic_type] = []
            type_patterns[seq.semantic_type].append(seq)
        
        # Create pattern suggestions
        suggestions = []
        task_lower = task.lower()
        
        for sem_type, seqs in type_patterns.items():
            if len(seqs) >= 2:  # Need multiple examples to form a pattern
                pattern = SequencePattern(seqs)
                best_example = pattern.get_best_example()
                
                # Check if pattern matches task semantics
                matches_semantics = (
                    (sem_type == 'create' and any(word in task_lower for word in ['create', 'add', 'new'])) or
                    (sem_type == 'modify' and any(word in task_lower for word in ['update', 'change', 'modify'])) or
                    (sem_type == 'test' and any(word in task_lower for word in ['test', 'verify'])) or
                    (sem_type == 'fix' and any(word in task_lower for word in ['fix', 'repair', 'solve']))
                )
                
                if matches_semantics:
                    suggestions.append({
                        'pattern_type': sem_type,
                        'success_rate': pattern.avg_success_rate,
                        'suggested_actions': [
                            step.action for step in best_example.steps
                        ],
                        'example_instruction': best_example.steps[0].action.get('description', '')
                    })
        
        return suggestions

    def plan_next_action(self, current_state: Dict[str, Any], task: str, history: List[Dict[str, Any]], examples: List[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Plan the next action for a task using pattern matching and LLM guidance
        """
        # Get pattern suggestions if examples are provided
        pattern_suggestions = []
        if examples:
            pattern_suggestions = self._get_pattern_suggestions(task, current_state, examples)
        
        # Analyze task context
        context = self._analyze_task_context(task, current_state)
        
        # Prepare planning prompt with pattern information
        planning_prompt = f"""Plan the next action for this coding task:

Task: {task}

Current State:
{json.dumps(current_state, indent=2)}

Action History:
{json.dumps(history, indent=2)}

Task Context:
{json.dumps(context, indent=2)}

Pattern Suggestions:
{json.dumps(pattern_suggestions, indent=2)}

Requirements:
1. Action must be valid and safe
2. Must make progress toward goal
3. Consider successful patterns from similar tasks
4. Handle potential errors
5. Follow best practices

Return a single action as JSON object with 'type' and required parameters."""

        # Get action recommendation
        response = self.chat_completion([
            {"role": "system", "content": "You are a coding agent planning the next action in a sequence."},
            {"role": "user", "content": planning_prompt}
        ])
        
        try:
            action = json.loads(response.choices[0].message.content)
            
            # Validate action before returning
            if self._validate_action(action, context):
                return action
                
        except:
            # Fallback: Use most relevant pattern suggestion if available
            if pattern_suggestions:
                best_pattern = max(pattern_suggestions, key=lambda x: x['success_rate'])
                current_step = len(history)
                if current_step < len(best_pattern['suggested_actions']):
                    return best_pattern['suggested_actions'][current_step]
        
        return None