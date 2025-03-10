import unittest
from Fuser import Fuser
from ReportManager import ReportManager
from dataman.DataManager import DataManager

class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup test environment once for all tests"""
        cls.test_data = load_test_data()
        cls.config = load_test_config()

    def setUp(self):
        self.dataManager = DataManager('test')
        self.fuser = Fuser(self.dataManager)
        self.reportManager = ReportManager(self.fuser)
        self.mock_data_source = MockDataSource(self.test_data)

    def test_full_prediction_pipeline(self):
        """Test the entire prediction pipeline"""
        forday = '2023-10-01'
        try:
            # Run predictions
            self.fuser.runseq()
            
            # Check reports
            self.assertTrue(os.path.exists(f'report_{forday}.csv'))
            
            # Verify predictions
            predictions = pd.read_csv(f'report_{forday}.csv')
            self.assertGreater(len(predictions), 0)
            
        except Exception as e:
            self.fail(f"Pipeline test failed: {e}")

    def test_full_pipeline_stress(self):
        """Test pipeline under heavy load"""
        large_ticker_list = generate_test_tickers(1000)
        with self.assertLogs(level='INFO') as logs:
            results = self.pipeline.run_pipeline(large_ticker_list)
            
        self.assertGreater(len(results['predictions']), 900)
        self.assertLess(len(results['errors']), 50)

    @pytest.mark.performance
    def test_prediction_performance(self):
        """Test prediction performance benchmarks"""
        start_time = time.time()
        predictions = self.predictor.getPrediction('2023-01-01', 'AAPL')
        duration = time.time() - start_time
        
        self.assertLess(duration, 0.1)  # Should complete within 100ms
