# Fraud Transaction Analysis Project

## Overview
The **Fraud Transaction Analysis Project** analyzes credit card transactions to detect fraudulent activity.  
It includes **Exploratory Data Analysis (EDA)**, **Machine Learning model training**, and a **Streamlit dashboard** for interactive fraud detection.


## Features
- **Fraud vs Normal transaction analysis**: Compares patterns of fraudulent and legitimate transactions to identify anomalies.
- **Fraudulent Transaction Detection**: Uses machine learning models to classify transactions as fraud or legitimate.  
- **Interactive Dashboard**: Built with Streamlit, allowing users to visualize fraud trends and test predictions.  
- **Data Preprocessing**: Cleans, balances, and prepares the dataset for better model performance.  
- **Database Integration**: Uses SQLite to store transaction data and perform fraud analysis queries.  
- **Comprehensive Evaluation**: Provides ROC-AUC, precision, recall, and confusion matrix to validate models.  

## Technologies Used
- **Python**: Core programming language for building the project.  
- **Scikit-learn**: For training classification models and evaluating performance.  
- **Streamlit**: For creating an interactive web-based dashboard.  
- **SQLite**: For storing and analyzing transaction data.  
- **Pandas & Matplotlib**: For data processing and visualization.  

## Installation 

1. Clone the repository:
```bash
   git clone https://github.com/your-username/FraudDetection.git
 ```

2. Navigate to the project directory
```bash
  cd FraudDetection
```

3. Create a virtual environment
```bash
  python -m venv venv
  venv\Scripts\activate   # (Windows)
  source venv/bin/activate  # (Mac/Linux)
```

4. Install dependencies
```bash
pip install -r requirements.txt
```

5. Run the Streamlit application
```bash
streamlit run app/streamlit_app.py
```

## Usage
1. **Run Model Training:** Train the fraud detection model with preprocessed transaction data.

2. **Visualize Insights:** Use the Streamlit dashboard to explore fraud vs. normal transaction trends.

3. **Run SQL Analysis:** Query the transactions table inside fraud.db for deeper insights.





