import joblib
import pandas as pd
df=pd.read_csv("processed_student_data.csv")
x=df.drop(columns=["G3"])
x=pd.get_dummies(x)
model=joblib.load("model/deployment_model.pkl")
prediction=model.predict(x.iloc[[0]])
print("Prediction for the first student:", round(prediction[0],2))
print("Actual:", df.iloc[0]["G3"])