import click
import subprocess
import sys
from pathlib import Path

@click.command()
@click.option('--env', type=click.Choice(['dev', 'prod']), default='dev')
def deploy(env: str):
    """Deploy the prediction system"""
    try:
        # Create required directories
        dirs = ['data', 'logs', 'reports', 'models']
        for d in dirs:
            Path(d).mkdir(exist_ok=True)
            
        # Install requirements
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
        
        # Run setup verification
        subprocess.run([sys.executable, 'scripts/verify_setup.py'], check=True)
        
        click.echo("Deployment successful!")
        
    except Exception as e:
        click.echo(f"Deployment failed: {e}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    deploy()
