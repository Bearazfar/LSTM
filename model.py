import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
import os
import datetime

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create main output directory
MAIN_OUTPUT_DIR = 'output_lstm'
os.makedirs(MAIN_OUTPUT_DIR, exist_ok=True)


def create_run_directory():
    """
    Create a new run directory within output_lstm
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(MAIN_OUTPUT_DIR, f'run_{timestamp}')

    # Create subdirectories
    subfolders = ['models', 'plots', 'metrics']
    for folder in subfolders:
        os.makedirs(os.path.join(run_dir, folder), exist_ok=True)

    return run_dir


def load_data(filepath):
    """
    Load and preprocess data from CSV file
    """
    df = pd.read_csv(filepath, encoding="utf8")

    # Convert timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M', dayfirst=True)

    # Encode categorical variables
    df['SoilMoistureIndex'] = LabelEncoder().fit_transform(df['SoilMoistureIndex'])
    df['WeatherCondition'] = LabelEncoder().fit_transform(df['WeatherCondition'])

    # Convert to float
    df['Treatment'] = df['Treatment'].astype(float)
    df['SoilMoistureIndex'] = df['SoilMoistureIndex'].astype(float)
    df['WeatherCondition'] = df['WeatherCondition'].astype(float)

    return df


def prepare_data(df, input_features, target_feature):
    """
    Prepare data for training by removing outliers and scaling
    """
    # Remove outliers
    df_selected = df[input_features + [target_feature]]
    z_scores = stats.zscore(df_selected)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    df_filtered = df_selected[filtered_entries]

    # Scale data
    X = df_filtered[input_features].values.astype('float32')
    y = df_filtered[target_feature].values.astype('float32')

    input_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_scaled = input_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    return X_scaled, y_scaled, input_scaler, target_scaler


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


class MultivariateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, device):
        super().__init__()
        self.device = device
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True
        ).to(device)
        self.linear = nn.Linear(hidden_size, 1).to(device)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        predictions = self.linear(last_time_step)
        return predictions


def train_model(X_train, y_train, model_params, training_params):
    """
    Train the LSTM model
    """
    # Create datasets
    X_tensor, y_tensor = create_dataset(X_train, y_train, model_params['lookback'])
    X_tensor, y_tensor = X_tensor.to(device), y_tensor.to(device)

    # Initialize model
    model = MultivariateLSTM(
        input_size=model_params['input_size'],
        hidden_size=model_params['hidden_size'],
        num_layers=model_params['num_layers'],
        dropout_rate=model_params['dropout'],
        device=device
    )

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=training_params['learning_rate'])
    loss_fn = nn.MSELoss()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    loader = data.DataLoader(
        data.TensorDataset(X_tensor, y_tensor),
        batch_size=training_params['batch_size'],
        shuffle=True
    )

    # Training loop
    losses = []
    for epoch in range(training_params['epochs']):
        model.train()
        epoch_losses = []

        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{training_params['epochs']}, Loss: {avg_loss:.4f}")

    return model, losses


def plot_loss(losses, save_path):
    """
    Plot and save training loss curve
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_predictions(actual_train, train_pred, actual_test, test_pred, lookback, save_path):
    """
    Plot and save model predictions
    """
    plt.figure(figsize=(16, 10))
    plt.suptitle("Multivariate LSTM Prediction")

    train_indices = np.arange(lookback, len(actual_train) + lookback)
    test_indices = np.arange(len(actual_train) + lookback,
                             len(actual_train) + len(actual_test) + lookback)

    plt.plot(train_indices, actual_train, label='Actual Train', color='blue', alpha=0.7)
    plt.plot(train_indices, train_pred, label='Train Predictions', color='red', alpha=0.7)
    plt.plot(test_indices, actual_test, label='Actual Test', color='green', alpha=0.7)
    plt.plot(test_indices, test_pred, label='Test Predictions', color='orange', alpha=0.7)

    plt.title(f"LSTM Predictions")
    plt.xlabel("Time Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_model(model, model_params, training_params, scalers, fold, run_dir):
    """
    Save model, parameters, and scalers in the models folder
    """
    models_dir = os.path.join(run_dir, 'models')

    model_path = os.path.join(models_dir, f'model_fold_{fold}.pth')
    params_path = os.path.join(models_dir, f'params_fold_{fold}.pt')
    scalers_path = os.path.join(models_dir, f'scalers_fold_{fold}.pkl')

    # Save model
    torch.save(model.state_dict(), model_path)

    # Save parameters
    torch.save({
        'model_params': model_params,
        'training_params': training_params
    }, params_path)

    # Save scalers
    import pickle
    with open(scalers_path, 'wb') as f:
        pickle.dump({
            'input_scaler': scalers['input_scaler'],
            'target_scaler': scalers['target_scaler']
        }, f)


def load_model(fold, run_dir):
    """
    Load model, parameters, and scalers from the models folder
    """
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

    # Load scalers
    import pickle
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)

    return model, model_params, training_params, scalers


def save_metrics(metrics_dict, fold, run_dir):
    """
    Save metrics to CSV file
    """
    metrics_dir = os.path.join(run_dir, 'metrics')
    metrics_path = os.path.join(metrics_dir, f'metrics_fold_{fold}.csv')
    pd.DataFrame([metrics_dict]).to_csv(metrics_path, index=False)


def main():
    # Create new run directory inside output_lstm
    run_dir = create_run_directory()
    print(f"Saving results to: {run_dir}")

    # Model parameters
    model_params = {
        'input_size': 6,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'lookback': 6
    }

    # Training parameters
    training_params = {
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 1000
    }

    # Load and prepare data
    input_features = ['Temp', 'Humid', 'LightIntensity', 'Treatment', 'SoilMoistureIndex', 'WeatherCondition']
    target_feature = 'SoilHumid'

    try:
        df = load_data(
            "D:\\Project\\RelativeHumidity-LSTM-main\\greenhouse\\datasheet4_resampling_with_condition_d4.csv")
    except FileNotFoundError:
        print("Error: Data file not found. Please check the file path.")
        return

    X_scaled, y_scaled, input_scaler, target_scaler = prepare_data(df, input_features, target_feature)

    scalers = {
        'input_scaler': input_scaler,
        'target_scaler': target_scaler
    }

    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)

    for fold, (train_index, test_index) in enumerate(tscv.split(X_scaled), 1):
        print(f"\nProcessing fold {fold}")

        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y_scaled[train_index], y_scaled[test_index]

        # Train model
        model, losses = train_model(X_train, y_train, model_params, training_params)

        # Save model and related files
        save_model(model, model_params, training_params, scalers, fold, run_dir)

        # Plot and save training loss
        plot_loss(losses, os.path.join(run_dir, 'plots', f'loss_fold_{fold}.png'))

        # Generate and save predictions
        model.eval()
        with torch.no_grad():
            X_train_tensor, _ = create_dataset(X_train, y_train, model_params['lookback'])
            X_test_tensor, _ = create_dataset(X_test, y_test, model_params['lookback'])

            train_pred = model(X_train_tensor.to(device)).cpu().numpy()
            test_pred = model(X_test_tensor.to(device)).cpu().numpy()

        # Transform predictions back to original scale
        train_pred = target_scaler.inverse_transform(train_pred)
        test_pred = target_scaler.inverse_transform(test_pred)

        actual_train = target_scaler.inverse_transform(y_train[model_params['lookback']:].reshape(-1, 1))
        actual_test = target_scaler.inverse_transform(y_test[model_params['lookback']:].reshape(-1, 1))

        # Calculate and save metrics
        metrics = {
            'fold': fold,
            'train_rmse': np.sqrt(np.mean((train_pred - actual_train) ** 2)),
            'test_rmse': np.sqrt(np.mean((test_pred - actual_test) ** 2)),
            'test_mae': mean_absolute_error(actual_test, test_pred),
            'test_r2': r2_score(actual_test, test_pred)
        }

        print(f"\nFold {fold} Results:")
        for metric, value in metrics.items():
            if metric != 'fold':
                print(f"{metric}: {value:.4f}")

        save_metrics(metrics, fold, run_dir)

        # Plot and save predictions
        plot_predictions(actual_train, train_pred, actual_test, test_pred,
                         model_params['lookback'],
                         os.path.join(run_dir, 'plots', f'predictions_fold_{fold}.png'))

        # Test model loading
        try:
            print(f"\nTesting model loading for fold {fold}")
            loaded_model, _, _, _ = load_model(fold, run_dir)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")


if __name__ == "__main__":
    main()
