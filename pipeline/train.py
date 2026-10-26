import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "processed_student_data.csv"
model_dir = BASE_DIR / "model"
model_dir.mkdir(exist_ok=True)
df = pd.read_csv(data_path, sep=",", engine="python")
df.columns = df.columns.str.strip()
print("Number of columns:", len(df.columns))
print("Columns:", df.columns.tolist())
X = df.drop(columns=["G3"])
y = df["G3"]
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, model_dir / "deployment_model.pkl")
print("Model Training Completed")
