import click
from pathlib import Path
from rich.console import Console
from .core.agent import CodingAgent

console = Console()

@click.group()
def cli():
    """AI Coding Agent CLI"""
    pass

@cli.command()
@click.argument('instruction')
@click.option('--repo-path', '-r', default='.',
              help='Path to the Git repository')
@click.option('--storage-path', '-s', default='~/.ai-agent/trajectories',
              help='Path to store agent trajectories')
@click.option('--bm25-candidates', '-k', default=50,
              help='Number of candidates to retrieve in first-stage BM25 retrieval')
def execute(instruction: str, repo_path: str, storage_path: str, bm25_candidates: int):
    """Execute a coding task in the repository"""
    try:
        repo_path = Path(repo_path).resolve()
        storage_path = Path(storage_path).expanduser().resolve()
        
        console.print(f"[bold blue]Executing task:[/] {instruction}")
        console.print(f"[bold blue]Repository:[/] {repo_path}")
        
        agent = CodingAgent(str(repo_path), str(storage_path), bm25_top_k=bm25_candidates)
        trajectory = agent.execute_task(instruction)
        
        console.print("\n[bold green]Task completed![/]")
        console.print("[bold blue]Actions taken:[/]")
        for i, action in enumerate(trajectory.actions, 1):
            console.print(f"{i}. {action['type']}: {action.get('description', '')}")
            
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        raise click.Abort()

@cli.command()
@click.option('--storage-path', '-s', default='~/.ai-agent/trajectories',
              help='Path to store agent trajectories')
def generate_tasks(storage_path: str):
    """Generate new tasks from repository documentation"""
    # TODO: Implement task generation from docs
    console.print("[yellow]Task generation not yet implemented[/]")

def main():
    cli()