import joblib
import numpy as np
import os

class ExoplanetModel:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(base_dir, "ml", "models", "exoplanet_model.pkl")

        self.model = joblib.load(model_path)

    def predict(self, row):
        features = np.array([[row["brightness"], row["temperature"]]])

        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0][1]

        return {
            "star_id": row.get("star_id", "unknown"),
            "prediction_score": round(float(probability), 2),
            "is_exoplanet": bool(prediction),
            "confidence": "HIGH" if probability > 0.85 else "MEDIUM"
        }