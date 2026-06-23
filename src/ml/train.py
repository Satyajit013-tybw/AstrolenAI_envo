import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# fake dataset (we simulate exoplanet data)
data = {
    "brightness": [0.8, 0.6, 0.9, 0.4, 0.7, 0.3],
    "temperature": [5500, 4900, 6000, 4500, 5200, 4000],
    "label": [1, 1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[["brightness", "temperature"]]
y = df["label"]

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

os.makedirs("src/ml/models", exist_ok=True)

joblib.dump(model, "src/ml/models/exoplanet_model.pkl")

print("Model trained and saved successfully 🚀")