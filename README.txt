========================================================================
CUSTOMER CHURN PREDICTION SYSTEM
========================================================================
Developed for: Teyzix Core Machine Learning Internship  
Domain: Machine Learning 
Project Status: Completed

PROJECT OVERVIEW
------------------------------------------------------------------------
This system utilizes a synthetically generated customer dataset to train 
a Machine Learning model capable of identifying historical patterns  and 
predicting whether a customer is likely to stop using a company's service. 
The final model is deployed via an interactive Streamlit web application.


REPOSITORY STRUCTURE
------------------------------------------------------------------------
* generate_data.py          - Script generating 1,550 realistic customer records.
* customer_churn_dataset.csv - The generated raw dataset file.
* eda_and_prep.py           - Handles categorical encoding and data preparation.
* prepared_customer_data.csv - Encoded, machine-ready matrix for modeling.
* train_model.py            - Trains the Random Forest and evaluates metrics.
* churn_model.pkl           - Serialized trained model file.
* model_features.pkl        - Saved list of training feature columns.
* app.py                    - Streamlit web interface for user predictions.


SYSTEM REQUIREMENTS & ENVIRONMENT SETUP
------------------------------------------------------------------------
1. Open your terminal in the project directory and create a virtual environment:
   python -m venv venv --without-pip

2. Activate the virtual environment:
   - Windows (CMD):       venv\Scripts\activate
   - Mac/Linux:           source venv/bin/activate

3. Bootstrap pip and install required lightweight dependencies:
   curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
   python get-pip.py
   pip install pandas numpy scikit-learn matplotlib seaborn streamlit


HOW TO RUN THE APPLICATION
------------------------------------------------------------------------
To launch the prediction web application interface, ensure your virtual 
environment is active and execute:

   streamlit run app.py


MODEL PERFORMANCE RESULTS
------------------------------------------------------------------------
The Random Forest Classifier was evaluated with the following test metrics[cite: 53, 60]:

* Accuracy:  71.29% 
* Precision: 69.35% 
* Recall:    62.77% 
* F1-Score:  65.90% 

Confusion Matrix:
[[135  38]   <- [True Negatives, False Positives]
 [ 51  86]]   <- [False Negatives, True Positives]

