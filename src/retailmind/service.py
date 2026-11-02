import joblib
import pandas as pd

from pathlib import Path


class ChurnPredictionService:

    def __init__(self, model_path):

        self.model_path = Path(model_path)
        self.model = None

    def load(self):

        self.model = joblib.load(self.model_path)

        return self

    def predict(self, customers):

        if self.model is None:
            self.load()

        df = pd.DataFrame(customers)

        probabilities = self.model.predict_proba(df)[:, 1]

        predictions = (probabilities >= 0.5).astype(int)

        df["churn_probability"] = probabilities
        df["predicted_churn"] = predictions

        df["prediction"] = [
            "Churn" if value == 1 else "No Churn"
            for value in predictions
        ]

        return df