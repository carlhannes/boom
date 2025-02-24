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