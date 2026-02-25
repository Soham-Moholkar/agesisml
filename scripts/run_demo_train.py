"""Demo training script — runs a quick pipeline to verify everything works."""
import requests
import time
import os

BASE_URL = "http://localhost:8000"


def wait_for_server():
    print("Waiting for server...")
    for _ in range(30):
        try:
            r = requests.get(f"{BASE_URL}/health")
            if r.status_code == 200:
                print("✓ Server is running")
                return True
        except Exception:
            pass
        time.sleep(1)
    print("✗ Server not available")
    return False


def upload_dataset(filepath, name):
    print(f"\nUploading {name}...")
    with open(filepath, "rb") as f:
        r = requests.post(f"{BASE_URL}/data/upload", files={"file": (name, f, "text/csv")})
    data = r.json()
    print(f"  ✓ Dataset ID: {data['dataset_id']} ({data['n_rows']} rows)")
    return data["dataset_id"]


def train_tabular(dataset_id, target, model_type):
    print(f"\nTraining {model_type} on {dataset_id}...")
    r = requests.post(f"{BASE_URL}/train/tabular", json={
        "dataset_id": dataset_id,
        "target_column": target,
        "model_type": model_type,
    })
    data = r.json()
    print(f"  ✓ Run {data['run_id']}: Accuracy={data['metrics']['accuracy']:.3f} F1={data['metrics']['f1']:.3f}")
    return data


def train_text(dataset_id, model_type="nb"):
    print(f"\nTraining text {model_type} on {dataset_id}...")
    r = requests.post(f"{BASE_URL}/train/text", json={
        "dataset_id": dataset_id,
        "text_column": "text",
        "target_column": "label",
        "model_type": model_type,
    })
    data = r.json()
    print(f"  ✓ Run {data['run_id']}: Accuracy={data['metrics']['accuracy']:.3f}")
    return data


def train_rl():
    print("\nTraining RL TicTacToe agent (10k episodes)...")
    r = requests.post(f"{BASE_URL}/train/rl/tictactoe", json={"episodes": 10000})
    data = r.json()
    print(f"  ✓ Run {data['run_id']}: Wins={data['stats']['wins']} Q-table={data['q_table_size']}")
    return data


def test_fuzzy():
    print("\nTesting fuzzy grading...")
    r = requests.post(f"{BASE_URL}/fuzzy/grade", json={
        "attendance": 85, "assignment": 75, "quiz": 80, "project": 90
    })
    data = r.json()
    print(f"  ✓ Grade: {data['grade']} (Score: {data['numeric_score']})")
    return data


def main():
    if not wait_for_server():
        return

    datasets_dir = os.path.join(os.path.dirname(__file__), "..", "datasets")

    # Upload datasets
    tab_id = upload_dataset(os.path.join(datasets_dir, "tabular_sample.csv"), "tabular_sample.csv")
    text_id = upload_dataset(os.path.join(datasets_dir, "text_sample.csv"), "text_sample.csv")

    # Train tabular models
    for mt in ["dt", "nb", "svm", "knn"]:
        train_tabular(tab_id, "is_fraud", mt)

    # Train text model
    train_text(text_id, "nb")

    # Train RL
    train_rl()

    # Test fuzzy
    test_fuzzy()

    print("\n" + "=" * 50)
    print("✓ Demo training complete! Check the dashboard.")
    print("=" * 50)


if __name__ == "__main__":
    main()
