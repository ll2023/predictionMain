import unittest
from scripts.verify_deployment import DeploymentVerifier
from scripts.automate_runs import AutomatedRunner
import tempfile
import os

class TestDeployment(unittest.TestCase):
    def setUp(self):
        self.test_config = {
            'data_source': 'test',
            'max_workers': 2,
            'batch_size': 10
        }
        self.config_file = self._create_test_config()
        
    def test_full_deployment_cycle(self):
        """Test complete deployment cycle"""
        verifier = DeploymentVerifier(self.config_file)
        self.assertTrue(verifier.verify_environment().passed)
        
        runner = AutomatedRunner(self.config_file)
        self.assertTrue(runner.verify_and_run(self.config_file))
