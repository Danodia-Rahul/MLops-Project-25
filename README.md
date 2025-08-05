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

### 📁 MLOPS-PROJECT-25 Directory Structure

<pre>
MLOPS-PROJECT-25/
├── Data/
│   └── train.csv
├── Deployment/
│   ├── app.py
│   ├── predict.py
│   ├── test.py
│   └── Dockerfile
├── MLFlow/
│   ├── colab_mlflow_tracking.ipynb
│   └── mlflow_tracking_script.py
├── Models/
│   ├── model.xgb
│   └── preprocessor.bin
├── Prefect/
│   ├── prefect_flow.py
│   └── prefect_setup.sh
├── .gitignore
├── .pylintrc
├── .prefectignore
├── Pipfile
├── Pipfile.lock
├── prefect.yaml
├── pyproject.toml
├── req.txt
└── README.md
</pre>



## ⚙️ Setup & Installation

<h2>📥 Clone the Repository</h2>
<pre><code>git clone https://github.com/your-username/mlops-25-project.git
cd mlops-25-project
</code></pre>

<h2>🐳 Build & Run with Docker</h2>
<p>We are using a Dockerized Flask application served via Gunicorn.</p>

<h3>✅ Build the Docker Image</h3>
<pre><code>docker build -t premium-predictor -f Deployment/Dockerfile .
</code></pre>

<h3>✅ Run the Container</h3>
<pre><code>docker run -p 9696:9696 premium-predictor
</code></pre>
<p>Make sure Docker is running in the background.</p>

<h3>On Linux</h3>
<p>Start Docker with the following command:</p>
<pre><code>sudo service docker start
</code></pre>

<p>Once everything is set up, the service will be available at:</p>
<pre><code>http://localhost:9696/predict
</code></pre>

<h2>🧪 Running a Test Inference</h2>
<p>Run the following command:</p>
<pre><code>python Deployment/test.py
</code></pre>

<p>You should see a response like:</p>

<h3>🔁 Inference API Details</h3>
<ul>
  <li><strong>Endpoint</strong></li>
  <li><strong>URL</strong>: <code>/predict</code></li>
  <li><strong>Method</strong>: <code>POST</code></li>
  <li><strong>Content-Type</strong>: <code>application/json</code></li>
</ul>

<hr>

<h4>📥 Example Input</h4>
<pre><code>{
  "Age": 30.0,
  "Annual Income": 32000.0,
  "Number of Dependents": 3.0,
  "Occupation": "Employed",
  "Credit Score": 690.0,
  "Property Type": "House"
}
</code></pre>

<h4>📤 Example Output</h4>
<pre><code>{
  "prediction": [594.72]
}
</code></pre>

<h2>🔄 Prefect Flow</h2>
<p>To orchestrate the inference workflow using Prefect:</p>

<ol>
  <li><strong>Start the Prefect Server</strong><br>
  Make sure the Prefect server is running in a separate terminal:</li>
</ol>

<pre><code>prefect server start
</code></pre>

<h2>⚙️ Prefect Setup</h2>
<p>Run the following command to set up Prefect:</p>
<pre><code>bash Prefect/prefect_setup.sh
</code></pre>

<h3>⚙️ Prefect Setup Details</h3>
<p>Running the setup script will:</p>
<ul>
  <li>✅ Create a <strong>work pool</strong></li>
  <li>✅ Create a <strong>deployment</strong></li>
  <li>✅ Start a <strong>worker</strong> from the work pool</li>
</ul>

<p>Once completed, you can navigate to the <strong>Prefect UI</strong> and start a flow using the <strong>Quick Run</strong> option from the created deployment.</p>
