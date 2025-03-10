import click
import sys
from scripts.verify_deployment import DeploymentVerifier
from scripts.automate_runs import AutomatedRunner
from config.platform_config import PlatformConfig
import logging
from pathlib import Path

def setup_deployment_logging():
    """Setup detailed deployment logging"""
    log_dir = Path("logs/deployment")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"deployment_{time.strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

@click.command()
@click.option('--env', type=click.Choice(['dev', 'prod']), default='dev')
@click.option('--verify-only', is_flag=True, help='Only verify without deploying')
@click.option('--force', is_flag=True, help='Force deployment even with warnings')
def deploy(env: str, verify_only: bool, force: bool):
    """Deploy the prediction platform with enhanced verification"""
    try:
        setup_deployment_logging()
        logger = logging.getLogger(__name__)
        logger.info(f"Starting deployment process for environment: {env}")
        
        # Verify Python environment
        logger.info("Verifying Python environment...")
        if not _verify_python_environment():
            logger.error("Python environment verification failed")
            sys.exit(1)
        
        # Load configuration
        config = PlatformConfig(env)
        
        # Verify deployment
        verifier = DeploymentVerifier(config.config_file)
        
        # Enhanced verification
        verification_result = verifier.verify_all()
        if not verification_result.passed:
            if not force:
                logger.error("Critical checks failed:")
                for error in verification_result.messages:
                    logger.error(f"- {error}")
                sys.exit(1)
            logger.warning("Proceeding despite verification failures (force mode)")
        
        if verify_only:
            logger.info("Verification successful")
            return
            
        # Setup system components
        system_state = SystemStateManager()
        metrics_collector = MetricsCollector()
        
        # Initialize automated runner
        runner = AutomatedRunner(config.config_file)
        
        # Initialize monitoring
        health_check = SystemHealthCheck()
        perf_monitor = PerformanceMonitor()
        
        # Start services
        health_check.start_monitoring()
        system_state.start()
        
        # Start monitoring
        runner.start_monitoring()
        
        # Schedule jobs
        runner.schedule_jobs()
        
        logger.info("Deployment successful")
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)

def _verify_python_environment():
    """Verify Python environment and dependencies"""
    try:
        import pkg_resources
        requirements = pkg_resources.parse_requirements(open('requirements.txt'))
        for requirement in requirements:
            pkg_resources.require(str(requirement))
        return True
    except Exception as e:
        logging.error(f"Environment verification failed: {e}")
        return False

if __name__ == '__main__':
    deploy()
