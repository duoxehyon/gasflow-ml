# GasFlow ML

A machine learning pipeline for predicting Ethereum priority fees based on historical transaction data.

## How It Works
1. **Data Collection**: Reads transaction data from JSON files containing mempool and historical fee statistics.
2. **Feature Extraction**: Extracts relevant gas-related features like mempool fee distribution and historical gas spikes.
3. **Training the Model**: Uses a `RandomForestRegressor` wrapped in `MultiOutputRegressor` to predict different priority fee quantiles.
4. **Inference**: Given real-time network conditions, the model estimates optimal priority fees for different urgency levels.

## Installation

Requires Python 3.7+

```
pip install -r requirements.txt
```

## Usage

### Training the Model
Run the training script to process data and train the model:

```
python gasflow_ml.py
```

This will output a trained model saved as `gas_model.pkl`.

### Making Predictions
To use the trained model for predictions:

```python
import joblib
from gasflow_ml import predict_priority_fees

# Load trained model
model_artifacts = joblib.load("gas_model.pkl")
model = model_artifacts['model']
scaler = model_artifacts['scaler']

# Example input data
network_context = {
    "mempool": {
        "count": 112,
        "p10": 0.0001,
        "p30": 0.1,
        "p50": 0.5,
        "p70": 1,
        "p90": 2
    },
    "history": {
        "gas_ratio_5": 0.526,
        "gas_spikes_25": 4,
        "fee_ewma_10": 0.845,
        "fee_ewma_25": 0.784
    }
}

# Predict fees
predictions = predict_priority_fees(network_context, model, scaler)
print(predictions)
```

## Data Format

The training data should be stored as JSON files in a directory (`./data`). Each file should follow this structure:

```json
{
    "network": {
        "mempool": {
            "count": 100,
            "p10": 0.5,
            "p30": 1.2,
            "p50": 2.5,
            "p70": 3.8,
            "p90": 5.1
        },
        "history": {
            "gas_ratio_5": 0.52,
            "gas_spikes_25": 3,
            "fee_ewma_10": 0.82,
            "fee_ewma_25": 0.76
        }
    },
    "txs": [
        {"max_priority_fee": 2000000000},
        {"max_priority_fee": 3000000000}
    ]
}
```
Note: This is an initial version and may have areas for improvement. Contributions and feedback are welcome.

## License
MIT License
