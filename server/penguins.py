from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import pickle
import pandas as pd

# Load the pretrained model using pickle
with open("/app/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

class PredictionInput(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    island: str
    sex: str

app = FastAPI()

@app.post("/predict")
def predict(input: PredictionInput):
    print(input)
    print(input.dict())
    # Prepare input data for prediction
    input_df = pd.DataFrame([input.dict()])
    input_df = pd.get_dummies(input_df)
    
    if "sex_Male" in input_df.columns:
        input_df['sex_Female'] = False
    elif "sex_Female" in input_df.columns:
        input_df['sex_Male'] = False
    
    if "island_Biscoe" in input_df.columns:
        input_df['island_Dream'] = False
        input_df['island_Torgersen'] = False
    elif "island_Dream" in input_df.columns:
        input_df['island_Biscoe'] = False
        input_df['island_Torgersen'] = False
    elif "island_Torgersen" in input_df.columns:
        input_df['island_Biscoe'] = False
        input_df['island_Dream'] = False
    
    new_column_order = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'island_Biscoe', "island_Dream", "island_Torgersen", "sex_Female", "sex_Male"]
    input_df = input_df[new_column_order]
    # Make prediction
    prediction = model.predict(input_df)[0]

    # Return the predicted species
    return {"prediction": prediction}