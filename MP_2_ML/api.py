import os
import pandas as pd
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Dictionary to hold the loaded model pipeline
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.pkl")
    
    print(f"Loading serialized model pipeline from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please run train.py first.")
    
    ml_models["pipeline"] = joblib.load(model_path)
    yield
    # Clean up on shutdown
    ml_models.clear()

# Initialize the FastAPI app with our lifespan handler and custom metadata
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="A FastAPI service that predicts Titanic passenger survival using a trained Scikit-Learn Logistic Regression pipeline.",
    version="1.0.0",
    lifespan=lifespan
)

# Define the Pydantic schema for input data validation
class PassengerInput(BaseModel):
    Pclass: int = Field(..., ge=1, le=3, description="Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)")
    Sex: str = Field(..., description="Gender/Sex ('male' or 'female')")
    Age: float = Field(..., ge=0.0, le=120.0, description="Age of the passenger in years")
    SibSp: int = Field(..., ge=0, description="Number of siblings/spouses aboard the Titanic")
    Parch: int = Field(..., ge=0, description="Number of parents/children aboard the Titanic")
    Fare: float = Field(..., ge=0.0, description="Passenger fare paid")
    Embarked: str = Field(..., description="Port of embarkation ('C' = Cherbourg, 'Q' = Queenstown, 'S' = Southampton)")
    HasCabin: int = Field(..., ge=0, le=1, description="Cabin flag (0 = No cabin, 1 = Has cabin)")

    # Example payload shown in the interactive Swagger UI docs
    model_config = {
        "json_schema_extra": {
            "example": {
                "Pclass": 3,
                "Sex": "male",
                "Age": 22.0,
                "SibSp": 1,
                "Parch": 0,
                "Fare": 7.25,
                "Embarked": "S",
                "HasCabin": 0
            }
        }
    }

# Define the output schema for response structure consistency
class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Survival prediction ('Survived' or 'Did not survive')")
    probability: float = Field(..., description="Probability of survival (0.0 to 1.0)")

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Titanic Survival Prediction API. Go to /docs for interactive Swagger UI documentation."
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(passenger: PassengerInput):
    # Ensure the model pipeline is loaded
    if "pipeline" not in ml_models:
        raise HTTPException(status_code=500, detail="Machine Learning model pipeline is not loaded.")
    
    try:
        # 1. Convert validated Pydantic object into a pandas DataFrame
        input_data = [passenger.model_dump()]
        input_df = pd.DataFrame(input_data)
        
        # 2. Extract pipeline
        pipeline = ml_models["pipeline"]
        
        # 3. Make predictions
        prediction = int(pipeline.predict(input_df)[0])
        probability = float(pipeline.predict_proba(input_df)[0][1])
        
        # 4. Generate semantic label
        prediction_label = "Survived" if prediction == 1 else "Did not survive"
        
        return PredictionResponse(
            prediction=prediction_label,
            probability=probability
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # When run directly, start the uvicorn development server
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
