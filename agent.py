import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path

# Paths
base = Path(__file__).parent
data_path = base / "data" / "iris.csv"
output_path = base / "workdir" / "sol.csv"
output_path.parent.mkdir(exist_ok=True)

# Load and preprocess
df = pd.read_csv(data_path)
df = df.rename(columns={"Species": "species"})
df["species"] = df["species"].str.replace("Iris-", "", regex=False)

# Train/test split
split_idx = int(0.8 * len(df))
train, test = df[:split_idx], df[split_idx:]

X_train = train.drop(columns=["Id", "species"])
y_train = train["species"]
X_test = test.drop(columns=["Id", "species"])

# Model and prediction
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
preds = model.predict(X_test)

# Save predictions to workdir
pd.DataFrame({"prediction": preds}).to_csv(output_path, index=False)
