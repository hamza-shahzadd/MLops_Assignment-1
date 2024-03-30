import unittest
from app import app, model
import pandas as pd

class TestModel(unittest.TestCase):
    def test_prediction(self):
        # Create a sample input data
        sample_data = {'X': [1, 2, 3, 4, 5]}
        X_new = pd.DataFrame(sample_data['X'], columns=['X'])

        # Make predictions using the model
        y_pred = model.predict(X_new)

        # Check if predictions are of the expected length
        self.assertEqual(len(y_pred), len(sample_data['X']))

if __name__ == '__main__':
    unittest.main()
