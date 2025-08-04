# 🧠 Insurance Premium Prediction (MLOps Project)

This project is built as part of the **#mlopszoomcamp** challenge. The goal is to predict insurance premiums based on features such as age, income, occupation, and credit score.

We follow a practical MLOps pipeline using tools like **MLflow**, **XGBoost**, **Prefect**, and **Docker**. The model was trained in a separate environment (e.g., Colab) and the trained model artifacts are reused here for fast inference.

This repository focuses on:
- Serving a trained model using **Flask** API
- Building a **Dockerized** microservice for prediction
- Running and testing the model inference workflow


## 🛠️ Tech Stack

- **Language:** Python 3.11  
- **ML Framework:** XGBoost  
- **Cloud:** AWS  
- **Orchestration:** Prefect  
- **Web Framework:** Flask + Gunicorn  
- **Containerization:** Docker  
- **Experiment Tracking:** MLflow  
- **Version Control:** Git & GitHub

## 📁 Project Structure

MLOPS-PROJECT-25
│
├── Data/
│ └── train.csv ← Training data 
│
├── Deployment/ 
│ ├── app.py 
│ ├── predict.py
│ ├── test.py 
│ └── Dockerfile 
│
├── MLFlow/ 
│ ├── colab_mlflow_tracking.ipynb
│ └── mlflow_tracking_script.py
│
├── Models/ 
│ ├── model.xgb
│ └── preprocessor.bin
│
├── Prefect/ 
│ ├── prefect_flow.py
│ └── prefect_setup.sh
│
├── .gitignore
├── .pylintrc
├── .prefectignore
├── Pipfile / Pipfile.lock 
├── prefect.yaml 
├── pyproject.toml 
├── req.txt 
└── README.md 


## ⚙️ Setup & Installation

### 1. 🚀 Clone the Repository

```bash
git clone https://github.com/your-username/mlops-25-project.git
cd mlops-25-project


### 2. 🐳 Build & Run with Docker

We are using a Dockerized Flask application served via Gunicorn.

#### ✅ Build the Docker Image

```bash
docker build -t premium-predictor -f Deployment/Dockerfile .


#### ✅ Run the Container

```bash
docker run -p 9696:9696 premium-predictor


Make sure Docker is running in the background.

### On Linux

Start Docker with the following command:

```bash
sudo service docker start

Once everything is set up, the service will be available at:

```bash
http://localhost:9696/predict

## 🧪 Running a Test Inference

Run the following command:

```bash
python Deployment/test.py

You should see a response like:

### 🔁 Inference API Details

**Endpoint**  
**URL**: `/predict`  
**Method**: `POST`  
**Content-Type**: `application/json`

---

**📥 Example Input**

```json
[
  {
    "Age": 30.0,
    "Annual Income": 32000.0,
    "Number of Dependents": 3.0,
    "Occupation": "Employed",
    "Credit Score": 690.0,
    "Property Type": "House"
  }
]

**📤 Example Output**

```json
{
  "prediction": [594.72]
}

## 🔄 Prefect Flow

To orchestrate the inference workflow using Prefect:

1. **Start the Prefect Server**  
   Make sure the Prefect server is running in a separate terminal:

   ```bash
   prefect server start


## ⚙️ Prefect Setup

Run the following command to set up Prefect:

```bash
bash Prefect/prefect_setup.sh

## ⚙️ Prefect Setup Details

Running the setup script will:

- ✅ Create a **work pool**
- ✅ Create a **deployment**
- ✅ Start a **worker** from the work pool

Once completed, you can navigate to the **Prefect UI** and start a flow using the **Quick Run** option from the created deployment.
