Creditworthiness Prediction (Machine Learning Project)
OVERVIEW
This project predicts whether a loan applicant is creditworthy Good Credit or not Bad Credit using historical financial and demographic data.
It leverages multiple machine learning algorithms Logistic Regression, Decision Trees, Random Forest, and XGBoost to classify applicants based on their financial profiles.

OBJECTIVE
To build and evaluate machine learning models that can accurately predict credit risk using key applicant features such as income, credit history, debt, and employment status.

DATASET
Dataset Used: German Credit Data (UCI Machine Learning Repository)


Size: 1000 records


Features: 20 (mix of numerical and categorical)


Target Variable:


good → Creditworthy (1)


bad → Not creditworthy (0)




Alternatively, you can load it directly from scikit-learn:
from sklearn.datasets import fetch_openml
data = fetch_openml("credit-g", version=1, as_frame=True)

PROJECT WORKFLLOW


Data Loading & Inspection
Load and explore the dataset using pandas and visualize data distributions.


Data Preprocessing


Handle missing values


Encode categorical variables (OneHotEncoder)


Normalize numeric features (StandardScaler)




Model Training


Logistic Regression


Decision Tree


Random Forest


XGBoost




Model Evaluation


Accuracy


Precision


Recall


ROC-AUC


ROC Curve Visualization





Technologies & Libraries
CategoryToolsLanguagePython 3.xData Handlingpandas, numpyModelingscikit-learn, xgboostVisualizationmatplotlibEvaluationsklearn.metrics
Install dependencies:
pip install pandas numpy scikit-learn matplotlib xgboost


How to Run the Project


Clone the repository
git clone https//https://github.com/krishnapopuri185/Code_alpha2/edit/main/README.md
cd creditworthiness-prediction



Install dependencies
pip install -r requirements.txt



Run the model script
python credit_model.py



Run in Jupyter Notebook
jupyter notebook Creditworthiness_Prediction.ipynb


Visualizations


ROC Curves comparing all models


Feature importance (for Random Forest and XGBoost)


Confusion matrix for each classifier



Future Improvements


Add hyperparameter tuning using GridSearchCV or Optuna


Perform feature selection for better interpretability


Integrate SHAP or LIME for model explainability


Build a web dashboard using Streamlit
