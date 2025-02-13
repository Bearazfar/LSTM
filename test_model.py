import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy import stats
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle


class MultivariateLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, device):
        super().__init__()
        self.device = device
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True
        ).to(device)
        self.linear = torch.nn.Linear(hidden_size, 1).to(device)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        predictions = self.linear(last_time_step)
        return predictions


def create_dataset(X, y, lookback):
    """
    Create sequences of data for time series prediction
    """
    if len(X) <= lookback:
        raise ValueError("Lookback period longer than dataset")

    X_list, y_list = [], []
    for i in range(len(X) - lookback):
        X_list.append(X[i:i + lookback])
        y_list.append(y[i + lookback])

    return torch.from_numpy(np.array(X_list)).float(), torch.from_numpy(np.array(y_list)).float()


def load_model(fold, run_dir):
    """
    Load model, parameters, and scalers from the models folder
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = os.path.join(run_dir, 'models')

    model_path = os.path.join(models_dir, f'model_fold_{fold}.pth')
    params_path = os.path.join(models_dir, f'params_fold_{fold}.pt')
    scalers_path = os.path.join(models_dir, f'scalers_fold_{fold}.pkl')

    # Check if files exist
    if not all(os.path.exists(p) for p in [model_path, params_path, scalers_path]):
        raise FileNotFoundError(f"Model files for fold {fold} not found in {models_dir}")

    # Load parameters
    params = torch.load(params_path)
    model_params = params['model_params']
    training_params = params['training_params']

    # Create model and load weights
    model = MultivariateLSTM(
        input_size=model_params['input_size'],
        hidden_size=model_params['hidden_size'],
        num_layers=model_params['num_layers'],
        dropout_rate=model_params['dropout'],
        device=device
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Load scalers
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)

    return model, model_params, training_params, scalers


def load_and_preprocess_test_data(test_filepath, input_features, target_feature):
    """
    Load and preprocess external test data
    """
    # Load test data
    df = pd.read_csv(test_filepath, encoding="utf8")

    # Convert timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M', dayfirst=True)

    # Encode categorical variables
    df['SoilMoistureIndex'] = LabelEncoder().fit_transform(df['SoilMoistureIndex'])
    df['WeatherCondition'] = LabelEncoder().fit_transform(df['WeatherCondition'])

    # Convert to float
    df['Treatment'] = df['Treatment'].astype(float)
    df['SoilMoistureIndex'] = df['SoilMoistureIndex'].astype(float)
    df['WeatherCondition'] = df['WeatherCondition'].astype(float)

    # Select features and target
    X = df[input_features].values.astype('float32')
    y = df[target_feature].values.astype('float32')

    return X, y, df


def evaluate_model(run_dir, fold, test_filepath):
    """
    Evaluate trained model on external test dataset
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and related components
    print(f"Loading model from {run_dir}")
    model, model_params, _, scalers = load_model(fold, run_dir)
    model.eval()

    # Define features
    input_features = ['Temp', 'Humid', 'LightIntensity', 'Treatment',
                      'SoilMoistureIndex', 'WeatherCondition', 'HumidBefore']
    target_feature = 'SoilHumid'

    # Load and preprocess test data
    print(f"Loading test data from {test_filepath}")
    X_test, y_test, df_test = load_and_preprocess_test_data(
        test_filepath, input_features, target_feature)

    # Scale the test data using the saved scalers
    X_test_scaled = scalers['input_scaler'].transform(X_test)
    y_test_scaled = scalers['target_scaler'].transform(y_test.reshape(-1, 1)).flatten()

    # Create sequences
    X_test_tensor, y_test_actual = create_dataset(
        X_test_scaled, y_test_scaled, model_params['lookback'])
    X_test_tensor = X_test_tensor.to(device)

    # Get predictions
    print("Making predictions...")
    with torch.no_grad():
        test_pred_scaled = model(X_test_tensor).cpu().numpy()

    # Transform predictions back to original scale
    test_pred = scalers['target_scaler'].inverse_transform(test_pred_scaled)
    actual_test = scalers['target_scaler'].inverse_transform(
        y_test_actual.numpy().reshape(-1, 1))

    # Calculate metrics
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(actual_test, test_pred)),
        'MAE': mean_absolute_error(actual_test, test_pred),
        'R2': r2_score(actual_test, test_pred)
    }

    # Plot predictions
    plt.figure(figsize=(12, 6))
    plt.plot(actual_test, label='Actual', color='blue', alpha=0.7)
    plt.plot(test_pred, label='Predicted', color='red', alpha=0.7)
    plt.title('Model Predictions on External Test Dataset')
    plt.xlabel('Time')
    plt.ylabel('Soil Humidity')
    plt.legend()
    plt.grid(True)

    # Save plot
    plot_path = os.path.join(run_dir, 'plots', f'external_test_predictions_fold_{fold}.png')
    plt.savefig(plot_path)
    plt.close()

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(run_dir, 'metrics', f'external_test_metrics_fold_{fold}.csv')
    metrics_df.to_csv(metrics_path, index=False)

    return metrics, test_pred, actual_test


def main():
    # Specify paths
    run_dir = "output_lstm\\run_20250213_194251"
    test_filepath = "D:\\Project\\RelativeHumidity-LSTM-main\\datatest\\datasheet_for_test.csv"
    fold = 5  # เลือก fold ที่ต้องการทดสอบ

    try:
        # Evaluate model
        metrics, predictions, actuals = evaluate_model(run_dir, fold, test_filepath)

        # Print results
        print("\nExternal Test Dataset Evaluation Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # Print sample predictions
        print("\nSample Predictions:")
        for i in range(min(20, len(predictions))):
            print(f"Actual: {actuals[i][0]:.2f}, Predicted: {predictions[i][0]:.2f}, "
                  f"Diff: {abs(actuals[i][0] - predictions[i][0]):.2f}")

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
