## gasflow_ml
# A machine learning pipeline for predicting Ethereum priority fees.

import os
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

# Config
DATA_DIR = "./data"
QUANTILES = [0.1, 0.2, 0.3, 0.5, 0.6]
RANDOM_STATE = 42


def load_and_process_data(data_dir=DATA_DIR):
    """Load block data and prepare features/targets."""
    features, targets = [], []
    
    for filename in sorted(os.listdir(data_dir)):
        if filename.startswith("block_") and filename.endswith(".json"):
            try:
                with open(os.path.join(data_dir, filename), 'r') as f:
                    data = json.load(f)
                    
                    if 'txs' not in data or data['txs'] is None:
                        continue
                    
                    # Extract features
                    network = data['network']
                    features.append([
                        network['mempool']['count'],
                        network['mempool']['p10'],
                        network['mempool']['p30'],
                        network['mempool']['p50'],
                        network['mempool']['p70'],
                        network['mempool']['p90'],
                        network['history']['gas_ratio_5'],
                        network['history']['gas_spikes_25'],
                        network['history']['fee_ewma_10'],
                        network['history']['fee_ewma_25']
                    ])
                    
                    # Calculate target quantiles
                    priority_fees = [int(tx['max_priority_fee']) / 1e9 for tx in data['txs'] if 'max_priority_fee' in tx]
                    if priority_fees:
                        targets.append(np.quantile(priority_fees, QUANTILES))
            
            except Exception as e:
                print(f"Skipping {filename} due to error: {str(e)}")
                continue
    
    return np.array(features), np.array(targets)


def train_model(X, y):
    """Train the ML model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=RANDOM_STATE)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    base_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    model = MultiOutputRegressor(base_model)
    
    print("Training model...")
    model.fit(X_train_scaled, y_train)
    
    # Save trained model
    model_artifacts = {
        'model': model,
        'scaler': scaler,
        'feature_names': [
            'mempool_count', 'mempool_p10', 'mempool_p30', 'mempool_p50', 
            'mempool_p70', 'mempool_p90', 'gas_ratio_5', 'gas_spikes_25', 
            'fee_ewma_10', 'fee_ewma_25'
        ],
        'quantiles': QUANTILES
    }
    joblib.dump(model_artifacts, 'gas_model.pkl')
    
    print("Model training complete. Saved as 'gas_model.pkl'")
    return model, scaler


def predict_priority_fees(network_context, model, scaler):
    """Predict fees for new network context."""
    features = np.array([[
        network_context['mempool']['count'],
        network_context['mempool']['p10'],
        network_context['mempool']['p30'],
        network_context['mempool']['p50'],
        network_context['mempool']['p70'],
        network_context['mempool']['p90'],
        network_context['history']['gas_ratio_5'],
        network_context['history']['gas_spikes_25'],
        network_context['history']['fee_ewma_10'],
        network_context['history']['fee_ewma_25']
    ]])
    
    features_scaled = scaler.transform(features)
    preds = model.predict(features_scaled)[0]
    
    return {
        '50': preds[0],
        '75': preds[1],
        '80': preds[2],
        '90': preds[3],
        '99': preds[4]
    }


if __name__ == "__main__":
    X, y = load_and_process_data()
    print(f"Loaded {len(X)} samples with {X.shape[1]} features each")
    
    model, scaler = train_model(X, y)
    
    example_context = {
        "mempool": {
            "count": 112,
            "p10": 0.0001,
            "p30": 0.1,
            "p50": 0.5,
            "p70": 1,
            "p90": 2
        },
        "history": {
            "gas_ratio_5": 0.5262975939337673,
            "gas_spikes_25": 4,
            "fee_ewma_10": 0.845250841409841,
            "fee_ewma_25": 0.7843088454933078
        }
    }
    
    predictions = predict_priority_fees(example_context, model, scaler)
    print("\nExample prediction:", predictions)
