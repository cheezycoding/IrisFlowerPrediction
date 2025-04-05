from sklearn.datasets import load_iris
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
import joblib
from pydantic import BaseModel
import numpy as np


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


templates = Jinja2Templates(directory="templates")

app = FastAPI(title="Iris Species Predictor API")


try:
    model = joblib.load("iris_model.pkl")
    from sklearn.datasets import load_iris
    iris = load_iris()
    target_names = iris.target_names
    print("Model and target names loaded successfully!")
except FileNotFoundError:
    print("Error: Model file 'iris_model.pkl' not found.")
    model = None
    target_names = None
except Exception as e:
    print(f"Error loading model or target names: {e}")
    model = None
    target_names = None

@app.post("/predict")
async def predict_species(features: IrisFeatures):
    """
    Receives iris features and predicts the species.
    """
    if model is None or target_names is None:
        raise HTTPException(status_code=500, detail="Model not loaded or target names unavailable.")


    input_data = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])

    try:
        prediction_index = model.predict(input_data) 
        predicted_species = target_names[prediction_index[0]] 


        return {"predicted_species": predicted_species}

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
