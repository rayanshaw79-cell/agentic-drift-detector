import sys
import os
import random

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drift.ml_detector import DriftModel

def synthesize_healthy_data(num_records=500):
    """
    Synthesizes a baseline of healthy telemetry executions.
    This replaces the need for a pre-warmed SQLite database for now.
    """
    data = []
    for _ in range(num_records):
        state = {
            "step_count": random.choice([3, 4, 4, 5]),
            "retry_count": random.choice([0, 0, 0, 1]),
            "execution_time_ms": max(100, int(random.gauss(300, 50))),
            "severity": random.choices(["low", "medium", "high"], weights=[0.6, 0.3, 0.1])[0]
        }
        
        # Healthy decision logic
        if state["severity"] == "high":
            state["decision"] = "escalate"
        else:
            state["decision"] = random.choices(["auto_resolve", "escalate"], weights=[0.9, 0.1])[0]
            
        data.append(state)
    return data

if __name__ == "__main__":
    print("Synthesizing baseline training data (500 records)...")
    historical_data = synthesize_healthy_data(500)
    
    print("Training ML Drift Model (IsolationForest)...")
    detector = DriftModel()
    detector.train(historical_data)
    
    print("Training complete. Model saved to config/model.joblib.")
