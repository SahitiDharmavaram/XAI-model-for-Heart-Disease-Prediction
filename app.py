from flask import Flask, render_template, request
import joblib
import pandas as pd
import xgboost as xgb

app = Flask(__name__)


# Load your dataset for feature scaling (if needed)
data = pd.read_csv('C:\\Users\\Sahiti\\OneDrive\\Desktop\\xai_proj\\heart.csv')

# print("Dataset Columns:", data.columns.tolist())
# Define the feature names (modify according to your dataset)
feature_names = data.columns.tolist()
feature_names.remove("target")

# Load the XGBoost model
model = joblib.load('C:\\Users\\Sahiti\\OneDrive\\Desktop\\income\\best_model.pkl')

model.feature_names = feature_names

# # Print the loaded model and its feature names
# print("Loaded Model:", model)

# Ensure that the model is an XGBoost classifier
if not isinstance(model, xgb.XGBClassifier):
    raise ValueError("The loaded model is not an XGBoost classifier.")


@app.route('/')
def home():
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve user inputs from the form
    user_inputs = [int(request.form.get('age')),
                   int(request.form.get('sex')),
                   int(request.form.get('cp')),
                   int(request.form.get('trestbps')),
                   int(request.form.get('chol')),
                   int(request.form.get('fbs')),
                   int(request.form.get('restecg')),
                   int(request.form.get('thalach')),
                   int(request.form.get('exang')),
                   float(request.form.get('oldpeak')),
                   int(request.form.get('slope')),
                   int(request.form.get('ca')),
                   int(request.form.get('thal'))]

    print(user_inputs)

    # Perform predictions using your model
    probabilities = model.predict_proba([user_inputs])[0]

    # Extract the probability of class 1 (heart disease)
    probability_heart_disease = probabilities[1]

    return render_template('index.html', feature_names=feature_names, probability_heart_disease=probability_heart_disease)


if __name__ == '__main__':
    app.run(debug=True)
