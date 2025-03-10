import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skopt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from skopt.space import Real, Integer
import time
import os
from skopt import gp_minimize
from sklearn.model_selection import TimeSeriesSplit

# Contribution: how to reduce the h-index variance to less than 1 (the optimal point that affects human thermal comfort sensation)

# Set the directory and round number for this optimization
dir_name = 'output_baye_filter'
round = "1"

# Consider GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create dir_name and round inside dir_name directory if it doesn't exist
os.makedirs(dir_name, exist_ok=True)
os.makedirs(f'{dir_name}/{round}', exist_ok=True)

# Load the dataset
df = pd.read_csv("D:\\65109076\\LSTM_Model\\greenhouse\\datasheet4_resampling_4885.csv", encoding="utf8")

print('original data:')
print(df.describe().to_string(), "\n")

# Set index to datetime using column 'Timestamp'
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M', dayfirst=True)

# df['SoilMoistureIndex'] = LabelEncoder().fit_transform(df['SoilMoistureIndex'])
# df['WeatherCondition'] = LabelEncoder().fit_transform(df['WeatherCondition'])
#
# df['Treatment'] = df['Treatment'].astype(float)
# df['SoilMoistureIndex'] = df['SoilMoistureIndex'].astype(float)
# df['WeatherCondition'] = df['WeatherCondition'].astype(float)

# Data Summary
print('Format data:')
print(df.describe().to_string(), "\n")

# Select features (now multiple input features)
input_features = [
            'Temp', 'Humid', 'LightIntensity', 'Treatment'
]

# input_features = [
#             'Temp', 'Humid', 'LightIntensity', 'Treatment', 'SoilHumid', 'WeatherCondition'
# ]
target_feature = 'SoilHumid'

# Remove outliers
df = df[input_features + [target_feature]]
z_scores = stats.zscore(df)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
df = df[filtered_entries]


# Prepare the dataset
X = df[input_features].values.astype('float32')
y = df[target_feature].values.astype('float32')

# Separate scalers for input features and target
input_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_scaled = input_scaler.fit_transform(X)
y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

########################################################################################################################
# Time series split, n_splits is the number of splits (dependent on the dataset), test_size = n_samples // (n_splits + 1)
tscv = TimeSeriesSplit(n_splits=5)
########################################################################################################################

def create_dataset(X, y, lookback):
    if len(X) <= lookback:
        raise ValueError("Lookback period longer than dataset")

    X_list, y_list = [], []
    for i in range(len(X) - lookback):
        X_list.append(X[i:i + lookback])
        y_list.append(y[i + lookback])

    X_array = np.array(X_list)
    y_array = np.array(y_list)

    return torch.from_numpy(X_array).float(), torch.from_numpy(y_array).float()

class MultivariateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, device):
        super().__init__()
        self.device = device
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True,
            device=device
        ).to(device)
        self.linear = nn.Linear(hidden_size, 1).to(device)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        predictions = self.linear(last_time_step)
        return predictions

class LSTMWrapper:
    def __init__(self):
        self.model = None
        self.input_size = len(input_features)
        self.current_params = None

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def fit(self, X, y, lookback=4, hidden_size=50, num_layers=2, learning_rate=0.01,
            dropout=0.1, batch_size=64, epochs=200):
        start_time = time.time()
        self.current_params = locals()
        del self.current_params['self']
        del self.current_params['X']
        del self.current_params['y']

        try:
            # Create datasets
            X_tensor, y_tensor = create_dataset(X, y, lookback)
            X_tensor, y_tensor = X_tensor.to(device), y_tensor.to(device)

            # Initialize model
            self.model = MultivariateLSTM(
                self.input_size,
                hidden_size,
                num_layers=num_layers,
                dropout_rate=dropout,
                device=device
            )
            self.model = self.model.to(device)

            # Training setup
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            loss_fn = nn.MSELoss()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            loader = data.DataLoader(
                data.TensorDataset(X_tensor, y_tensor),
                batch_size=batch_size,
                shuffle=True
            )

            # Training loop
            for epoch in range(epochs):
                self.model.train()
                epoch_losses = []

                for X_batch, y_batch in loader:
                    optimizer.zero_grad()
                    y_pred = self.model(X_batch)
                    loss = loss_fn(y_pred.squeeze(), y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(loss.item())

                if epoch % 50 == 0:
                    print(f"Epoch {epoch}/{epochs}, Loss: {np.mean(epoch_losses):.4f}")

            return self

        except Exception as e:
            print(f"Error in fit: {e}")
            return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet")

        X_tensor, _ = create_dataset(X, np.zeros(len(X)), self.current_params['lookback'])
        X_tensor = X_tensor.to(device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions

    def score(self, X, y):
        """Return negative MSE score for optimization"""
        try:
            predictions = self.predict(X)
            y_seq = y[self.current_params['lookback']:]
            mse = np.mean((predictions.squeeze() - y_seq) ** 2)
            return -mse
        except Exception as e:
            print(f"Error in score: {e}")
            return float('-inf')

def run_optimization(tuning_method='bayes'):
    results_list = []
    start_time = time.time()
    trial_counter = 0

    if tuning_method == 'bayes':
        search_space = [
            Integer(1, 7, name='lookback'),
            Integer(25, 512, name='hidden_size'),
            Integer(1, 5, name='num_layers'),
            Real(0.001, 0.05, prior='log-uniform', name='learning_rate'),
            Real(0.1, 0.5, name='dropout'),
            Integer(32, 128, name='batch_size'),
            Integer(100, 1500, name='epochs')
        ]

        def objective(params):
            nonlocal trial_counter
            trial_counter += 1

            lookback, hidden_size, num_layers, learning_rate, dropout, batch_size, epochs = params
            # lookback, hidden_size, num_layers, learning_rate, batch_size, epochs = params
            current_params = {
                'lookback': int(lookback),
                'hidden_size': int(hidden_size),
                'num_layers': int(num_layers),
                'learning_rate': float(learning_rate),
                'dropout': float(dropout),
                'batch_size': int(batch_size),
                'epochs': int(epochs),
            }

            try:
                print(f"Trial {trial_counter} - Training with params: {current_params}")

                # Train model with current parameters
                model = LSTMWrapper()

                train_size, test_size = 0, 0
                actual_train, actual_test = [], []
                train_pred, test_pred = [], []

                # Get split indices
                splits = list(tscv.split(X_scaled))
                # Track the end of the previous training data
                previous_train_end = 0

                for split_idx, (train_index, test_index) in enumerate(splits):

                    print(f"\nProcessing split {split_idx + 1}/{len(splits)}")

                    # Get cumulative training data up to this point
                    if split_idx > 0:
                        # Include all previous training data plus new training data
                        train_index = np.concatenate([np.arange(previous_train_end), train_index])

                    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
                    y_train, y_test = y_scaled[train_index], y_scaled[test_index]

                    print(f"Train index: {train_index}, size: {len(X_train)}")
                    print(f"Test index: {test_index}, size: {len(X_test)}")

                    # Update the previous train end marker
                    previous_train_end = max(train_index) + 1

                    # If not first split, reduce epochs to avoid overtraining
                    current_epochs = current_params['epochs'] if split_idx == 0 else current_params['epochs'] // 2

                    # Fit model with current cumulative data
                    model.fit(X_train, y_train,
                              **{**current_params, 'epochs': current_epochs})

                    # Get predictions and reshape them to 2D arrays
                    train_predictions = model.predict(X_train).reshape(-1, 1)
                    test_predictions = model.predict(X_test).reshape(-1, 1)

                    train_pred.append(train_predictions)
                    test_pred.append(test_predictions)

                    # Reshape actual values to 2D arrays
                    actual_train.append(y_train[current_params['lookback']:].reshape(-1, 1))
                    actual_test.append(y_test[current_params['lookback']:].reshape(-1, 1))

                    train_size += len(X_train)
                    test_size += len(X_test)

                # Concatenate predictions and actual values
                train_pred = np.concatenate(train_pred)
                test_pred = np.concatenate(test_pred)

                actual_train = np.concatenate(actual_train)
                actual_test = np.concatenate(actual_test)

                # Inverse transform to get actual values
                train_pred_actual = target_scaler.inverse_transform(train_pred)
                test_pred_actual = target_scaler.inverse_transform(test_pred)
                actual_train = target_scaler.inverse_transform(actual_train)
                actual_test = target_scaler.inverse_transform(actual_test)

                train_rmse = np.sqrt(np.mean((train_pred_actual - actual_train) ** 2))
                test_rmse = np.sqrt(np.mean((test_pred_actual - actual_test) ** 2))
                r2 = r2_score(actual_test, test_pred_actual)
                mae = mean_absolute_error(actual_test, test_pred_actual)

                # Save results for this trial
                trial_results = {
                    **current_params,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'test_mae': mae,
                    'test_r2': r2,
                    'training_time': time.time() - start_time,
                    'trial_number': trial_counter
                }
                results_list.append(trial_results)

                # Save trial results to CSV after each trial
                pd.DataFrame(results_list).to_csv(
                    f'{dir_name}/{round}/optimization_results_{tuning_method}_trial_{trial_counter}.csv',
                    index=False
                )

                # Plot results
                plt.figure(figsize=(16, 10))
                plt.suptitle(f"Trial {trial_counter} - Multivariate LSTM Prediction")

                train_indices = np.arange(current_params['lookback'], len(actual_train) + current_params['lookback'])
                test_indices = np.arange(len(actual_train) + current_params['lookback'],
                                         len(actual_train) + len(actual_test) + current_params['lookback'])

                plt.plot(train_indices, actual_train, label='Actual Train', color='blue', alpha=0.7)
                plt.plot(train_indices, train_pred_actual, label='Train Predictions', color='red', alpha=0.7)
                plt.plot(test_indices, actual_test, label='Actual Test', color='green', alpha=0.7)
                plt.plot(test_indices, test_pred_actual, label='Test Predictions', color='orange', alpha=0.7)
                plt.axvline(x=train_size, color='purple', linestyle='--', label='Train/Test Split')

                plt.title(f"Trial {trial_counter} Predictions (Test RMSE: {test_rmse:.4f})")
                plt.xlabel("Time Index")
                plt.ylabel("Heat Index")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

                plt.savefig(f'{dir_name}/{round}/trial_{trial_counter}_results.png')
                plt.close()

                print(
                    f"Trial {trial_counter} completed in {trial_results['training_time']:.2f} seconds with test RMSE: {test_rmse:.4f} \n\n")

                return test_rmse

            except Exception as e:
                print(f"Error in trial {trial_counter}: {e}")
                return float('inf')

        # Run Bayesian optimization
        print("Starting Bayesian optimization...")
        result = gp_minimize(
            objective,
            search_space,
            n_calls=40,  # Increase exploration
            n_random_starts=10,  # More initial random configurations
            random_state=42
        )

        print("Bayesian optimization completed.")

        best_params = {
            'lookback': int(result.x[0]),
            'hidden_size': int(result.x[1]),
            'num_layers': int(result.x[2]),
            'learning_rate': float(result.x[3]),
            'batch_size': int(result.x[4]),
            'epochs': int(result.x[5])
        }

        results_list.append({
            **best_params,
            'train_rmse': None,
            'test_rmse': result.fun,
            'training_time': time.time() - start_time,
            'trial_number': trial_counter
        })

        print(f"Best parameters found: {best_params}")

    # Save final combined results
    final_results_df = pd.DataFrame(results_list)
    final_results_df.to_csv(f'{dir_name}/{round}/final_optimization_results_{tuning_method}.csv', index=False)

    # Print best performing configuration
    if len(results_list) > 0:
        best_config = min(results_list, key=lambda x: x['test_rmse'])
        print("\nBest Configuration:")
        for key, value in best_config.items():
            print(f"{key}: {value}")

    return results_list


# Run optimization with chosen method
tuning_method = 'bayes'  # Choose from 'grid', 'random', or 'bayes'
results_list = run_optimization(tuning_method)

# Save results
results_df = pd.DataFrame(results_list)
# results_df.to_csv(f'{dir_name}/{round}/optimization_results_{tuning_method}_without_dropout.csv', index=False)
results_df.to_csv(f'{dir_name}/{round}/optimization_results_{tuning_method}_with_dropout.csv', index=False)

# Print best performing configuration
if len(results_list) > 0:
    best_config = min(results_list, key=lambda x: x['test_rmse'])
    print("\nBest Configuration:")
    for key, value in best_config.items():
        print(f"{key}: {value}")
