import unittest
from config.layer_config import LayerManager
from automation.layer_automator import LayerAutomator

class TestLayers(unittest.TestCase):
    def setUp(self):
        self.layer_manager = LayerManager('config/layers.yaml')
        
    def test_layer_independence(self):
        """Test each layer can operate independently"""
        layers = ['data', 'prediction', 'monitoring', 'reporting']
        
        for layer in layers:
            with self.subTest(layer=layer):
                self.assertTrue(
                    self.layer_manager.validate_layer(layer),
                    f"Layer {layer} failed independence test"
                )

    def test_layer_integration(self):
        """Test layers work together properly"""
        automator = LayerAutomator(self.layer_manager.config)
        results = automator.validate_layers()
        self.assertTrue(all(results.values()))
