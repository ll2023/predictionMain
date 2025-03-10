import unittest
from Fuser import Fuser
from dataman.DataManager import DataManager

class TestFuser(unittest.TestCase):
    def setUp(self):
        self.dataManager = DataManager('test')
        self.fuser = Fuser(self.dataManager)

    def test_genTADF(self):
        """
        Test the genTADF method.
        """
        forday = '2023-10-01'
        st = 'AAPL'
        pred = [1, 2, 3]
        val = [4, 5, 6]
        try:
            self.fuser.genTADF(forday, st, pred, val)
        except Exception as e:
            self.fail(f"genTADF raised an exception: {e}")

    def test_chooseBests(self):
        """
        Test the chooseBests method.
        """
        forday = '2023-10-01'
        try:
            self.fuser.chooseBests(forday)
        except Exception as e:
            self.fail(f"chooseBests raised an exception: {e}")

    def test_bernoulliAccept(self):
        """
        Test the bernoulliAccept method.
        """
        forday = '2023-10-01'
        sticker_alg = 'AAPL_alg'
        try:
            result = self.fuser.bernoulliAccept(forday, sticker_alg)
            self.assertIsInstance(result, tuple)
        except Exception as e:
            self.fail(f"bernoulliAccept raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
