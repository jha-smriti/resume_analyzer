import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load dataset (Ensure you have placement_data.csv)
df = pd.read_csv("placement_data.csv")

# Selecting features and labels
X = df[['CGPA', 'Internships', 'Projects', 'ATS_Score']]
y = df['Placement_Status']  # 1 = Placed, 0 = Not Placed

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save trained model
pickle.dump(model, open("placement_model.pkl", "wb"))
print("Model saved as placement_model.pkl")
