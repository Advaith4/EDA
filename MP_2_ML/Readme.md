# Mini Project 2: Titanic Survival Prediction - Machine Learning Pipeline & Deployment

This project builds directly on the dataset preprocessed in **Mini Project 1** (`MP_1_EDA/data/preprocessed_titanic.csv`). It implements a clean, modular Machine Learning pipeline, evaluates multiple classifiers, saves the best model, and serves predictions through a validated FastAPI web service.

---

## Project Structure

```text
MP_2_ML/
├── train.py          # Script to load data, build the pipeline, compare models, and save the best model
├── api.py            # FastAPI service exposing a validated /predict endpoint
├── Dockerfile        # Containerization instructions for production deployment
├── Readme.md         # Documentation (this file)
└── requirements.txt  # Python package dependencies
```

---

## 1. Setup & Installation

### Step A: Activate Virtual Environment
Ensure your virtual environment from the project root is active. 

On Windows:
```powershell
..\.venv\Scripts\activate
```

On Mac/Linux:
```bash
source ../.venv/bin/activate
```

### Step B: Install Dependencies
Install the required packages for Mini Project 2:
```bash
pip install -r requirements.txt
```

---

## 2. Train and Evaluate Models

Run the training script to load the preprocessed data, evaluate models, and save the best one:
```bash
python train.py
```

### What happens under the hood?
1. **Feature Selection**: Unused identifiers (`PassengerId`, `Name`, `Ticket`) are dropped.
2. **Pipeline Preprocessing**: 
   - Numerical columns (`Age`, `Fare`, `SibSp`, `Parch`) are scaled using `StandardScaler` (essential for optimization-based models like **Logistic Regression** where feature scales affect weights; note that **Random Forest** is tree-based and scale-invariant, so it does not require scaled features but works fine with them).
   - Categorical columns (`Sex`, `Embarked`, `Pclass`, `HasCabin`) are encoded using `OneHotEncoder` since they remain in string format in the preprocessed dataset.
3. **Model Comparison**: Both **Logistic Regression** and **Random Forest** are trained on an 80% split and tested on a 20% stratified split.
4. **Multi-Metric Evaluation**: The models are compared using Accuracy, Precision, Recall, F1-score, and Confusion Matrices.
5. **Serialization**: The best performing model (Logistic Regression, F1-score: `0.7344`) is serialized to `model.pkl`.

---

## 3. Run the FastAPI Application

Serve the trained model locally using Uvicorn:
```bash
python api.py
```
Or run directly via uvicorn:
```bash
uvicorn api:app --reload
```

The server will start at: `http://127.0.0.1:8000`

---

## 4. Test the API

### Interactive API Docs (Swagger UI)
Open your browser and navigate to:
```text
http://127.0.0.1:8000/docs
```
Here, you can click on the `POST /predict` endpoint, click **"Try it out"**, modify the example JSON payload, and click **Execute** to see real-time predictions directly from the browser.

### Example Request JSON Payload
```json
{
  "Pclass": 1,
  "Sex": "female",
  "Age": 35.0,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 53.1,
  "Embarked": "S",
  "HasCabin": 1
}
```

### Example Response JSON
```json
{
  "prediction": "Survived",
  "probability": 0.9033201681782856
}
```

---

## 5. Containerize with Docker

*Note: Ensure Docker Desktop is open and running on your machine.*

### Step A: Build the Docker Image
From the root of the project, build the image and tag it as `titanic-predictor`:
```bash
docker build -t titanic-predictor MP_2_ML
```

### Step B: Run the Docker Container
Run the container and map port 8000 of your machine to port 8000 of the container:
```bash
docker run -p 8000:8000 titanic-predictor
```

Now, navigate to `http://127.0.0.1:8000/docs` to test the API running inside the isolated Docker container!
