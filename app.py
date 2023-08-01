import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
import sys 

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 50
random_state = int(sys.argv[2]) if len(sys.argv) > 2 else 50

model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Optionally, you can print the classification report for more detailed evaluation.
print(classification_report(y_test, y_pred))

# Start an MLflow run to track the model
with mlflow.start_run() as run:
    
    # Log the model parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)

    # Log the model metrics
    mlflow.log_metric("accuracy", accuracy)

    # Save the model as an artifact
    mlflow.sklearn.log_model(model, "iris_model")
