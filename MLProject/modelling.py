import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import os

def main():
    mlflow.set_experiment("Titanic_Classification_Basic")
    
    # Enable autolog
    mlflow.sklearn.autolog()
    
    # Load data
    df = pd.read_csv("titanic_preprocessing/titanic_cleaned.csv")
    
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    active_run = mlflow.active_run()
    if active_run:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print(f"Model training completed within active run: {active_run.info.run_id}")
    else:
        with mlflow.start_run(run_name="RandomForest_Basic"):
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            print("Model training completed with autologging.")

if __name__ == "__main__":
    main()

