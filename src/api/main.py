
from fastapi import FastAPI, UploadFile, File
import pandas as pd
import io
from src.ml.model import ExoplanetModel
print("UPLOAD ENDPOINT HIT")
app = FastAPI()

model = ExoplanetModel()


@app.get("/")
def home():
    return {"message": "AstroLen AI REAL ML Backend 🚀 (Day 4)"}

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    try:
        print("UPLOAD ENDPOINT HIT")

        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        print(df.head())  # debug

        return {
            "rows": len(df),
            "columns": list(df.columns)
        }

    except Exception as e:
        print("❌ ERROR OCCURRED:")
        print(e)
        raise e