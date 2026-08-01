# Titanic EDA and Survival Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/FastAPI-Inference-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn" />
</p>

An end-to-end Titanic analytics project that combines exploratory data analysis, interactive visualization, model training, and API-based machine learning inference. The repository is organized as two mini projects: a Streamlit dashboard for data exploration and a FastAPI service for survival prediction.

## Overview

This project demonstrates a compact machine learning workflow:

1. Explore and understand the Titanic passenger dataset.
2. Present key survival patterns through an interactive dashboard.
3. Train and evaluate classification models using scikit-learn pipelines.
4. Serialize the best model for reuse.
5. Serve predictions through a validated REST API.
6. Package the API with Docker for containerized deployment.

## Repository Structure

```text
EDA-1/
├── README.md
├── requirements.txt
├── MP_1_EDA/
│   ├── app.py
│   ├── analytics/
│   │   └── data_analytics.ipynb
│   └── data/
│       ├── Titanic-Dataset.csv
│       └── preprocessed_titanic.csv
└── MP_2_ML/
    ├── api.py
    ├── train.py
    ├── model.pkl
    ├── requirements.txt
    ├── Dockerfile
    └── Readme.md
```

## Components

### Exploratory Data Analysis

The EDA module is located in `MP_1_EDA`.

- `app.py` provides an interactive Streamlit dashboard.
- `analytics/data_analytics.ipynb` contains notebook-based analysis.
- `data/Titanic-Dataset.csv` stores the original dataset.
- `data/preprocessed_titanic.csv` stores the cleaned dataset used by the dashboard and ML pipeline.

Dashboard features include:

- passenger count, survival rate, average age, and average fare KPIs
- passenger class filtering
- survival analysis by gender
- survival analysis by ticket class
- fare distribution analysis
- contextual insight cards for the selected segment

### Machine Learning Pipeline

The ML module is located in `MP_2_ML`.

`train.py` builds a supervised classification workflow that:

- loads the preprocessed Titanic dataset
- removes identifier and metadata columns
- separates numerical and categorical features
- applies scaling and one-hot encoding through a `ColumnTransformer`
- trains Logistic Regression and Random Forest models
- evaluates models using accuracy, precision, recall, F1-score, confusion matrix, and classification report
- selects the best model by F1-score
- saves the final pipeline to `model.pkl`

### Prediction API

`api.py` exposes a FastAPI service for real-time Titanic survival predictions.

Available endpoints:

- `GET /` returns a basic API welcome message.
- `POST /predict` returns a survival label and survival probability.
- `GET /docs` opens the interactive Swagger UI documentation.

Example request:

```json
{
  "Pclass": 3,
  "Sex": "male",
  "Age": 22.0,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S",
  "HasCabin": 0
}
```

Example response:

```json
{
  "prediction": "Did not survive",
  "probability": 0.3240872045
}
```

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Advaith4/EDA.git
cd EDA
```

### 2. Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For API-only usage, dependencies can also be installed from the ML module:

```bash
pip install -r MP_2_ML/requirements.txt
```

## Usage

### Run the Streamlit Dashboard

```bash
cd MP_1_EDA
streamlit run app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

### Train the Model

From the repository root:

```bash
python MP_2_ML/train.py
```

This regenerates `MP_2_ML/model.pkl`.

### Run the FastAPI Service

```bash
cd MP_2_ML
python api.py
```

Alternatively, run it with Uvicorn:

```bash
cd MP_2_ML
uvicorn api:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Docker Deployment

The FastAPI service can be built and run as a Docker container.

Build the image from the repository root:

```bash
docker build -t titanic-survival-api MP_2_ML
```

Run the container:

```bash
docker run -p 8000:8000 titanic-survival-api
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Tech Stack

- Python
- Pandas
- NumPy
- Plotly
- Streamlit
- scikit-learn
- Joblib
- FastAPI
- Pydantic
- Uvicorn
- Docker

## Key Outcomes

This repository presents a complete learning and portfolio project for tabular data analysis and ML deployment. It shows how to move from exploratory analysis to model training, then from a saved model to a production-style API interface with request validation and container support.

## Future Improvements

- Add automated tests for the API and training pipeline.
- Track experiment results and model metrics in a dedicated artifact.
- Add dashboard screenshots to the README.
- Deploy the API to a cloud platform.
- Add CI checks for formatting, tests, and Docker builds.
