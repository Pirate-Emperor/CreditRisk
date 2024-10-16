# CreditRisk

**Predict Bank Credit Risk using South German Credit Data**

## Motivation

Banks generate a significant portion of their revenue by providing loans, making effective credit risk assessment crucial. Poor risk management can lead to non-performing loans, impacting profitability. This project aims to minimize the risk of loan defaults by leveraging data mining techniques to analyze patterns in historical credit data. By building a predictive model, the goal is to determine whether an individual poses a safe (1) or risky (0) credit risk based on their attributes.

## Dataset

The dataset used in this project is the South German Credit dataset, available on the UCI Machine Learning Repository:

[South German Credit Dataset](https://archive.ics.uci.edu/dataset/522/south+german+credit)

## Overview

The "CreditRisk" project implements a machine learning pipeline to predict bank credit risk using the South German Credit Data. The project features:

- **Data Ingestion:** Load data and prepare it for analysis.
- **Data Transformation:** Handle missing values, apply scaling, and transform the data for model consumption.
- **Model Training:** Train and evaluate models to identify the most accurate classifier.
- **Prediction:** Deploy the model using a Flask-based web interface where users can test the model.

The project uses Python for backend processing, Flask for the web interface, and Cassandra as the database. Docker is used for easy containerization and deployment.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Workflow](#workflow)
5. [Customization](#customization)
6. [Contributing](#contributing)
7. [License](#license)

## Project Structure

```plaintext
├── .github\workflows
│   ├── main.yaml                  # Deployment configuration
├── artifacts                      # Stores saved models, preprocessed data, etc.
├── Documents                      # Project documentation
├── src/mlproject
│   ├── components                 # Key project components (data, transformation, model)
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_training.py
│   ├── pipelines                  # Flow for executing pipelines
│   │   ├── prediction_pipeline.py
│   │   ├── training_pipeline.py  
│   ├── exception.py               # Custom exceptions
│   ├── logger.py                  # Logging utility
│   ├── utils.py                   # Helper functions
├── templates/                     # HTML templates for Flask app
│   ├── history.html                       
│   ├── index.html
│   ├── train.html
│   ├── test.html
├── app.py                         # Main Flask application
├── Dockerfile                     # Docker configuration
├── LICENSE                        # License file
├── README.md                      # Project documentation (you are here)
├── requirements.txt               # Python dependencies
├── setup.py                       # Project packaging
```

## Installation

### 1. Clone the repository:

```bash
git clone https://github.com/Pirate-Emperor/CreditRisk.git
cd CreditRisk
```

### 2. Create a virtual environment and install dependencies:

```bash
conda create -p myenv python=3.9 -y
conda activate myenv
pip install -r requirements.txt
```

### 3. Set up environment variables:

Create a `.env` file in the root directory with your Cassandra credentials and necessary configurations:

```bash
CASSANDRA_USER = "your_username"
CASSANDRA_PASSWORD = "your_password"
CASSANDRA_KEYSPACE = "your_keyspace"
CASSANDRA_SECURE_BUNDLE ="path_to_secure_bundle.zip"

AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
AWS_REGION = ""
ECR_REPOSITORY_NAME = ""
```

## Usage

### Running the Flask Application

1. Start the application:

```bash
python app.py
```

2. Open your browser at [http://localhost:8080](http://localhost:8080) to access the application interface.

### Using Docker

1. Build the Docker image:

```bash
docker build -t CreditRisk .
```

2. Run the Docker container:

```bash
docker run -p 5000:5000 CreditRisk
```

3. Access the application at [http://localhost:5000](http://localhost:5000).

## Workflow

### 1. Data Ingestion

- The data is ingested from the Cassandra database.
- The dataset is processed, with renaming of columns, handling of null values, and splitting into `train.csv`, `test.csv`, and `raw.csv` in the `artifacts/` folder.

### 2. Data Transformation

- RobustScaler is used to scale the data to handle outliers.
- SimpleImputer deals with missing values.
- The transformed data is saved as `preprocessor.pkl`.

### 3. Model Training

- Various classification models are trained, and the best model is selected based on metrics such as accuracy.
- The trained model is saved as `model.pkl`.

### 4. Prediction

- The trained model predicts the risk class (safe or risky) for new inputs provided through the web interface.

## Customization

### 1. Data Ingestion

- Modify `data_ingestion.py` to accommodate different data sources and schemas. Connection settings can be configured in `utils.py`.

### 2. Data Transformation

- Customize `data_transformation.py` to apply different preprocessing techniques like feature selection, scaling methods, or imputation strategies.

### 3. Model Training

- Experiment with different models and hyperparameters by editing `model_training.py`. You can also use external libraries like TensorFlow or PyTorch.

### 4. Web Interface

- Update the HTML templates in the `templates/` folder to match your design preferences. You can modify form fields, outputs, and page styles.

## Contributing

Feel free to fork the repository, make changes, and submit pull requests. Contributions are welcome!

## License

This project is licensed under the Pirate-Emperor License. See the [LICENSE](LICENSE) file for details.

## Author

**Pirate-Emperor**

[![Twitter](https://skillicons.dev/icons?i=twitter)](https://twitter.com/PirateKingRahul)
[![Discord](https://skillicons.dev/icons?i=discord)](https://discord.com/users/1200728704981143634)
[![LinkedIn](https://skillicons.dev/icons?i=linkedin)](https://www.linkedin.com/in/piratekingrahul)

[![Reddit](https://img.shields.io/badge/Reddit-FF5700?style=for-the-badge&logo=reddit&logoColor=white)](https://www.reddit.com/u/PirateKingRahul)
[![Medium](https://img.shields.io/badge/Medium-42404E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@piratekingrahul)

- GitHub: [Pirate-Emperor](https://github.com/Pirate-Emperor)
- Reddit: [PirateKingRahul](https://www.reddit.com/u/PirateKingRahul/)
- Twitter: [PirateKingRahul](https://twitter.com/PirateKingRahul)
- Discord: [PirateKingRahul](https://discord.com/users/1200728704981143634)
- LinkedIn: [PirateKingRahul](https://www.linkedin.com/in/piratekingrahul)
- Skype: [Join Skype](https://join.skype.com/invite/yfjOJG3wv9Ki)
- Medium: [PirateKingRahul](https://medium.com/@piratekingrahul)

---
