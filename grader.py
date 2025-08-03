import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score

def grade():
    base = Path(__file__).parent
    data_path = base / "data" / "iris.csv"
    sol_path = base / "workdir" / "sol.csv"

    if not sol_path.exists():
        print('{"score": 0.0, "accuracy": 0.0}')
        return

    df = pd.read_csv(data_path)
    preds = pd.read_csv(sol_path)["prediction"]

    df = df.rename(columns={"Species": "species"})
    df["species"] = df["species"].str.replace("Iris-", "", regex=False)

    split_idx = int(len(df) * 0.8)
    y_true = df["species"].iloc[split_idx:].reset_index(drop=True)

    if len(preds) != len(y_true):
        print('{"score": 0.0, "accuracy": 0.0}')
        return

    accuracy = accuracy_score(y_true, preds)
    print(f"score: {accuracy:4f}")
    print(f"accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    grade()
  
