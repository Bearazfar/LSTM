import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")


# Custom Dataset class for time series
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        relu_out = self.relu(last_hidden)
        output = self.linear(relu_out)
        return output


# Create sequences function
def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)


# Training function
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)


# Evaluation function with R² score
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    r2 = r2_score(actuals, predictions)
    return total_loss / len(val_loader), predictions, actuals, r2


# Main execution
if __name__ == "__main__":
    # Load data
    data = pd.read_csv("D:\\Project\MicroClimate-LSTM-main_2\\metting room\\datasheet3_resampling_4606.csv",
                       encoding="utf8")
    values = data["SoilHumid"].values.reshape(-1, 1)

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(values)

    # Create sequences
    time_steps = 6
    X, y = create_sequences(scaled_data, time_steps)

    # Time Series Cross Validation
    tscv = TimeSeriesSplit(n_splits=5)
    mse_scores = []
    r2_scores = []

    # Model parameters
    input_size = 1
    hidden_size = 50
    num_layers = 1
    learning_rate = 0.001
    num_epochs = 500
    batch_size = 16

    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        # Split data
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Create data loaders
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        model = LSTMModel(input_size, hidden_size, num_layers).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            if (epoch + 1) % 100 == 0:
                print(f'Fold {fold + 1}, Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}')

        # Evaluation
        val_loss, predictions, actuals, r2 = evaluate_model(model, val_loader, criterion, device)
        mse = mean_squared_error(actuals, predictions)
        mse_scores.append(mse)
        r2_scores.append(r2)
        print(f"Fold {fold + 1} - MSE: {mse:.4f}, R²: {r2:.4f}")

    # Final results
    print("\nFinal Results:")
    print(f"Mean MSE: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
    print(f"Mean R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

    # Optional: Save the best model
    best_fold = np.argmax(r2_scores)
    print(f"\nBest performance was in fold {best_fold + 1}")
    print(f"Best R²: {r2_scores[best_fold]:.4f}")
    print(f"Corresponding MSE: {mse_scores[best_fold]:.4f}")
