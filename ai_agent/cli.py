#!/usr/bin/env python3
import click
import json
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console
from .core.agent import Agent, CodingAgent
from .core.learner import SelfLearner
from .core.task_generator import TaskGenerator
from .data.trajectory_manager import TrajectoryManager
from .environment.git_env import GitEnvironment
from .core.config import ConfigManager

console = Console()

@click.group()
def cli():
    """AI Agent CLI for automated coding tasks"""
    pass

@cli.command()
@click.argument('repo_path', type=click.Path(exists=True))
@click.option('--docs-path', type=click.Path(exists=True), help='Path to project documentation')
def learn(repo_path: str, docs_path: Optional[str]):
    """Learn from existing repository and documentation"""
    # Initialize components
    env = GitEnvironment(repo_path)
    tm = TrajectoryManager(str(Path(repo_path) / '.ai_agent' / 'trajectories'))
    learner = SelfLearner()
    agent = Agent(env, tm)
    
    click.echo("Starting learning process...")
    
    # Analyze repository
    framework_info = env.analyze_frameworks()
    file_patterns = env.analyze_patterns()
    
    # Bootstrap learning if documentation available
    if docs_path:
        count = learner.bootstrap_learning(
            docs_path=docs_path,
            framework_info=framework_info,
            file_patterns=file_patterns,
            trajectory_manager=tm
        )
        click.echo(f"Generated {count} successful trajectories from documentation")
    
    # Run quality maintenance
    removed = tm.maintain_quality()
    if removed > 0:
        click.echo(f"Removed {removed} low-quality trajectories")
    
    click.echo("Learning complete!")

@cli.command()
@click.argument('repo_path', type=click.Path(exists=True))
@click.argument('task_description', type=str)
@click.option('--dry-run', is_flag=True, help='Show planned actions without executing')
def execute(repo_path: str, task_description: str, dry_run: bool):
    """Execute a coding task in the repository"""
    # Initialize components
    env = GitEnvironment(repo_path)
    tm = TrajectoryManager(str(Path(repo_path) / '.ai_agent' / 'trajectories'))
    learner = SelfLearner()
    agent = Agent(env, tm)
    
    click.echo(f"Executing task: {task_description}")
    
    # Get current state
    state = env.get_state()
    
    if dry_run:
        # Show planned actions
        plan = agent.learner.generate_plan(task_description, state)
        click.echo("\nPlanned actions:")
        for action in plan:
            click.echo(f"- {action['type']}: {action.get('file', '')}")
        return
    
    # Execute task
    result = agent.execute_task(task_description)
    
    if result['status'] == 'success':
        click.echo("\nTask completed successfully!")
    else:
        click.echo(f"\nTask failed: {result.get('error', 'Unknown error')}")

@cli.command()
@click.argument('repo_path', type=click.Path(exists=True))
def analyze(repo_path: str):
    """Analyze repository and suggest tasks"""
    env = GitEnvironment(repo_path)
    task_gen = TaskGenerator()
    
    click.echo("Analyzing repository...")
    
    # Analyze frameworks and patterns
    framework_info = env.analyze_frameworks()
    file_patterns = env.analyze_patterns()
    
    # Generate suggested tasks
    tasks = []
    tasks.extend(task_gen.generate_framework_tasks(framework_info))
    tasks.extend(task_gen.generate_codebase_tasks(file_patterns))
    
    # Filter and sort by priority
    tasks = task_gen.filter_duplicate_tasks(tasks)
    tasks.sort(key=lambda t: 0 if t['priority'] == 'high' else 1)
    
    # Display results
    click.echo("\nFrameworks detected:")
    for framework in framework_info.get('frameworks', []):
        click.echo(f"- {framework}")
    
    click.echo("\nSuggested tasks:")
    for task in tasks:
        priority_mark = "(!)" if task['priority'] == 'high' else "   "
        click.echo(f"{priority_mark} {task['instruction']}")

@cli.command()
@click.argument('repo_path', type=click.Path(exists=True))
def status(repo_path: str):
    """Show learning status and statistics"""
    tm = TrajectoryManager(str(Path(repo_path) / '.ai_agent' / 'trajectories'))
    
    click.echo("AI Agent Status")
    click.echo("==============")
    
    # Show trajectory statistics
    total = len(tm.trajectories)
    click.echo(f"\nTotal trajectories: {total}")
    
    if total > 0:
        # Calculate quality distribution
        quality_counts = {'high': 0, 'medium': 0, 'low': 0}
        for tid in range(total):
            score = tm.get_trajectory_quality(tid)
            if score:
                if score.total_score >= 0.8:
                    quality_counts['high'] += 1
                elif score.total_score >= 0.6:
                    quality_counts['medium'] += 1
                else:
                    quality_counts['low'] += 1
        
        click.echo("\nQuality distribution:")
        click.echo(f"- High quality  : {quality_counts['high']}")
        click.echo(f"- Medium quality: {quality_counts['medium']}")
        click.echo(f"- Low quality   : {quality_counts['low']}")
    
    # Show pattern statistics
    patterns = tm.quality_metrics.action_patterns
    if patterns:
        click.echo("\nTop action patterns:")
        top_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        for pattern, count in top_patterns:
            click.echo(f"- {' -> '.join(pattern)}: {count} occurrences")

@cli.group()
def config():
    """Manage AI agent configuration"""
    pass

@config.command()
@click.argument('repo_path', type=click.Path(exists=True))
def show(repo_path: str):
    """Show current configuration"""
    config_manager = ConfigManager(repo_path)
    config = config_manager.get_config()
    
    click.echo("\nCurrent Configuration")
    click.echo("===================")
    
    for key, value in config.__dict__.items():
        click.echo(f"{key}: {value}")

@config.command()
@click.argument('repo_path', type=click.Path(exists=True))
@click.argument('key', type=str)
@click.argument('value', type=str)
def set(repo_path: str, key: str, value: str):
    """Set a configuration value"""
    config_manager = ConfigManager(repo_path)
    
    # Convert value to appropriate type
    if value.lower() in ('true', 'false'):
        value = value.lower() == 'true'
    elif value.replace('.', '').isdigit():
        value = float(value) if '.' in value else int(value)
    
    try:
        config_manager.update_config({key: value})
        click.echo(f"Updated {key} = {value}")
    except Exception as e:
        click.echo(f"Error updating config: {e}", err=True)

@config.command()
@click.argument('repo_path', type=click.Path(exists=True))
def reset(repo_path: str):
    """Reset configuration to defaults"""
    config_manager = ConfigManager(repo_path)
    config_manager.save_config(config_manager._create_default_config())
    click.echo("Configuration reset to defaults")

# Update the main cli group to include config commands
cli.add_command(config)

if __name__ == '__main__':
    cli()