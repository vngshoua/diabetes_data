import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load your dataset
data = pd.read_csv('diabetes.csv')

# Separate features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a model
model = LogisticRegression()
model.fit(X_scaled, y)

# Save the model and scaler
joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
