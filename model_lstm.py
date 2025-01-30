import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skopt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from skopt.space import Real, Integer
import time
import os
from skopt import gp_minimize
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# Set the round number for this optimization
round = "5"

# Consider GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Load the dataset
df = pd.read_csv("D:\\Project\\MicroClimate-LSTM-main\\greenhouse\\datasheet3_resampling.csv", encoding="utf8")

# df = df.drop('Timestamp', axis=1)

# df['SoilMoistureIndex'] = LabelEncoder().fit_transform(df['SoilMoistureIndex'])
# df['WeatherCondition'] = LabelEncoder().fit_transform(df['WeatherCondition'])

# Format data type
df['Treatment'] = df['Treatment'].astype(float)
# df['SoilMoistureIndex'] = df['SoilMoistureIndex'].astype(float)
# df['WeatherCondition'] = df['WeatherCondition'].astype(float)

df['Temp'] = df['Temp'].astype('float32')
df['Humid'] = df['Humid'].astype('float32')
df['SoilHumid'] = df['SoilHumid'].astype('float32')
df['LightIntensity'] = df['LightIntensity'].astype('float32')
df['Treatment'] = df['Treatment'].astype('float32')
# df['SoilMoistureIndex'] = df['SoilMoistureIndex'].astype('float32')
# df['WeatherCondition'] = df['WeatherCondition'].astype('float32')


print(df.head())
print(df.info(20))
print(df.dtypes)

# Select features (now multiple input features)
# input_features = ['Temp', 'Humid', 'LightIntensity', 'Treatment', 'SoilMoistureIndex', 'WeatherCondition']
input_features = ['Temp', 'Humid', 'LightIntensity', 'Treatment' ]

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


# Separate scaler for input features and target
input_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_scaled = input_scaler.fit_transform(X)
y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

# train-test split for time series
train_size = int(len(X_scaled) * 0.70) # 67
test_size = len(X_scaled) - train_size

X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

def create_dataset(X, y, lookback):
    """Transform a time series into a prediction dataset"""
    X_list, y_list = [], []
    for i in range(len(X) - lookback):
        X_list.append(X[i:i + lookback])
        y_list.append(y[i + lookback])
    return torch.tensor(X_list).float(), torch.tensor(y_list).float()

class MultivariateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate, device):
        super().__init__()
        self.device = device
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            # dropout=dropout_rate if num_layers > 1 else 0,
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

    def fit(self, X, y, lookback=3, hidden_size=50, num_layers=2, learning_rate=0.01,
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

def calculate_r2(y_true, y_pred):
    """Calculate R² score"""
    return r2_score(y_true, y_pred)

def run_optimization(tuning_method='bayes'):
    results_list = []
    start_time = time.time()
    trial_counter = 0

    if tuning_method == 'bayes':
        search_space = [
            Integer(1, 6, name='lookback'), # 1-7
            Integer(10, 1000, name='hidden_size'),
            Integer(1, 5, name='num_layers'),
            Real(0.001, 0.05, prior='log-uniform', name='learning_rate'),
            # Real(0.0, 0.2, name='dropout'),
            Integer(16, 128, name='batch_size'),
            Integer(50, 2000, name='epochs')
        ]

        def objective(params):
            nonlocal trial_counter
            trial_counter += 1

            # lookback, hidden_size, num_layers, learning_rate, dropout, batch_size, epochs = params
            lookback, hidden_size, num_layers, learning_rate, batch_size, epochs = params
            current_params = {
                'lookback': int(lookback),
                'hidden_size': int(hidden_size),
                'num_layers': int(num_layers),
                'learning_rate': float(learning_rate),
                # 'dropout': float(dropout),
                'batch_size': int(batch_size),
                'epochs': int(epochs),
            }

            try:
                print(f"Trial {trial_counter} - Training with params: {current_params}")

                # Train model with current parameters
                model = LSTMWrapper()
                model.fit(X_train, y_train, **current_params)

                # Make predictions
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)

                # Convert predictions back to original scale
                train_pred_actual = target_scaler.inverse_transform(train_pred)
                test_pred_actual = target_scaler.inverse_transform(test_pred)
                actual_train = target_scaler.inverse_transform(y_train[current_params['lookback']:].reshape(-1, 1))
                actual_test = target_scaler.inverse_transform(y_test[current_params['lookback']:].reshape(-1, 1))

                # Calculate metrics
                train_rmse = np.sqrt(np.mean((train_pred_actual - actual_train) ** 2))
                test_rmse = np.sqrt(np.mean((test_pred_actual - actual_test) ** 2))
                train_r2 = calculate_r2(actual_train, train_pred_actual)
                test_r2 = calculate_r2(actual_test, test_pred_actual)

                # Save results for this trial
                trial_results = {
                    **current_params,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'training_time': time.time() - start_time,
                    'trial_number': trial_counter
                }
                results_list.append(trial_results)

                # Save trial results to CSV after each trial
                pd.DataFrame(results_list).to_csv(
                    f'output_baye/{round}/optimization_results_{tuning_method}_trial_{trial_counter}.csv',
                    index=False
                )

                # Create subplot figure
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 15))
                plt.suptitle(f"Trial {trial_counter} - Multivariate LSTM Prediction")

                # Time series plot
                train_indices = np.arange(current_params['lookback'], train_size)
                test_indices = np.arange(train_size + current_params['lookback'], train_size + test_size)

                ax1.plot(train_indices, actual_train, label='Actual Train', color='blue', alpha=0.7)
                ax1.plot(train_indices, train_pred_actual, label='Train Predictions', color='red', alpha=0.7)
                ax1.plot(test_indices, actual_test, label='Actual Test', color='green', alpha=0.7)
                ax1.plot(test_indices, test_pred_actual, label='Test Predictions', color='orange', alpha=0.7)
                ax1.axvline(x=train_size, color='purple', linestyle='--', label='Train/Test Split')
                ax1.set_title(f"Time Series Prediction (Test RMSE: {test_rmse:.4f})")
                ax1.set_xlabel("Time Index")
                ax1.set_ylabel("Heat Index")
                ax1.legend()
                ax1.grid(True)

                # R² scatter plot
                ax2.scatter(actual_train, train_pred_actual, label=f'Train (R² = {train_r2:.4f})', alpha=0.5)
                ax2.scatter(actual_test, test_pred_actual, label=f'Test (R² = {test_r2:.4f})', alpha=0.5)

                # Add perfect prediction line
                min_val = min(min(actual_train), min(actual_test))
                max_val = max(max(actual_train), max(actual_test))
                ax2.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')

                ax2.set_title("Actual vs Predicted Values")
                ax2.set_xlabel("Actual Values")
                ax2.set_ylabel("Predicted Values")
                ax2.legend()
                ax2.grid(True)

                plt.tight_layout()
                plt.savefig(f'output_baye/{round}/trial_{trial_counter}_results.png')
                plt.close()

                print(f"Trial {trial_counter} completed in {trial_results['training_time']:.2f} seconds")
                print(f"Test RMSE: {test_rmse:.4f}")
                print(f"Train R²: {train_r2:.4f}")
                print(f"Test R²: {test_r2:.4f}\n")

                return test_rmse  # Return test RMSE as the objective to minimize

                # # Plot and save results for this trial
                # plt.figure(figsize=(16, 10))
                # plt.suptitle(f"Trial {trial_counter} - Multivariate LSTM Prediction")
                #
                # train_indices = np.arange(current_params['lookback'], train_size)
                # test_indices = np.arange(train_size + current_params['lookback'], train_size + test_size)
                #
                # plt.plot(train_indices, actual_train, label='Actual Train', color='blue', alpha=0.7)
                # plt.plot(train_indices, train_pred_actual, label='Train Predictions', color='red', alpha=0.7)
                # plt.plot(test_indices, actual_test, label='Actual Test', color='green', alpha=0.7)
                # plt.plot(test_indices, test_pred_actual, label='Test Predictions', color='orange', alpha=0.7)
                # plt.axvline(x=train_size, color='purple', linestyle='--', label='Train/Test Split')
                #
                # plt.title(f"Trial {trial_counter} Predictions (Test RMSE: {test_rmse:.4f})")
                # plt.xlabel("Time Index")
                # plt.ylabel("Heat Index")
                # plt.legend()
                # plt.grid(True)
                # plt.tight_layout()
                #
                # # Save plot for this trial
                # plt.savefig(f'output_baye/{round}/trial_{trial_counter}_results.png')
                # plt.close()
                #
                # print(f"Trial {trial_counter} completed in {trial_results['training_time']:.2f} seconds with test RMSE: {test_rmse:.4f} \n\n")
                #
                # return test_rmse  # Return test RMSE as the objective to minimize

            except Exception as e:
                print(f"Error in trial {trial_counter}: {e}")
                return float('inf')

        # Run Bayesian optimization
        print("Starting Bayesian optimization...")
        result = gp_minimize(
            objective,
            search_space,
            n_calls=10,  # Increase exploration 150
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
    final_results_df.to_csv(f'output_baye/{round}/final_optimization_results_{tuning_method}.csv', index=False)

    # Print best performing configuration
    if len(results_list) > 0:
        best_config = min(results_list, key=lambda x: x['test_rmse'])
        print("\nBest Configuration:")
        for key, value in best_config.items():
            print(f"{key}: {value}")

    return results_list

# Create output directory if it doesn't exist
os.makedirs('output_baye', exist_ok=True)

# Run optimization with chosen method
tuning_method = 'bayes'  # Choose from 'grid', 'random', or 'bayes'
results_list = run_optimization(tuning_method)

# Save results
results_df = pd.DataFrame(results_list)
results_df.to_csv(f'output_baye/{round}/optimization_results_{tuning_method}_without_dropout.csv', index=False)

# Print best performing configuration
if len(results_list) > 0:
    best_config = min(results_list, key=lambda x: x['test_rmse'])
    print("\nBest Configuration:")
    for key, value in best_config.items():
        print(f"{key}: {value}")

