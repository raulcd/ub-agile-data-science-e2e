import os
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main() -> None:

    # Create a tiny toy dataset (4 features)
    X, y = make_classification(
        n_samples=300,
        n_features=4,
        n_informative=3,
        n_redundant=1
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    # Simple, fast model
    model = LogisticRegression(max_iter=500, n_jobs=1)
    model.fit(X_train, y_train)

    # Quick metric
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy: {acc:.3f}")

    # Ensure output directory exists
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "model"))
    os.makedirs(out_dir, exist_ok=True)

    # Save as an MLflow model (sklearn flavor)
    # This creates `model/MLmodel` plus serialized model artifacts
    mlflow.sklearn.save_model(model, out_dir)
    print(f"Saved MLflow model to: {out_dir}")

    # Also write a tiny README about input expectations
    with open(os.path.join(out_dir, "INPUTS.txt"), "w") as f:
        f.write(
            "Model expects a list of 4 floats per sample (feature vector of length 4).\n"
            "Example request JSON: {\n  \"features\": [0.1, -1.2, 0.3, 2.4]\n}\n"
        )


if __name__ == "__main__":
    main()