import os
import pickle
import numpy as np
import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2
    ):
        super().__init__()

        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Final reconstruction layer
        self.output_layer = nn.Linear(
            hidden_size,
            input_size
        )

    def forward(self, x):

        # Encode input sequence
        _, (h, c) = self.encoder(x)

        # Use last hidden state
        latent = h[-1]

        # Repeat latent vector for every timestep
        repeated = latent.unsqueeze(1).repeat(
            1,
            x.shape[1],
            1
        )

        # Decode sequence
        decoded, _ = self.decoder(repeated)

        # Map back to original feature size
        reconstructed = self.output_layer(decoded)

        return reconstructed


# ==========================================
# Train Autoencoder
# ==========================================

def train_autoencoder(
    X_seq,
    epochs=50,
    lr=1e-3,
    hidden_size=64,
    num_layers=2,
    save_path="models/lstm_autoencoder.pth"
):

    os.makedirs("models", exist_ok=True)

    # Create model
    model = LSTMAutoencoder(
        input_size=X_seq.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    loss_fn = nn.MSELoss()

    # Convert to tensor
    X_tensor = torch.FloatTensor(X_seq)

    print("\nTraining LSTM Autoencoder...\n")

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        output = model(X_tensor)

        loss = loss_fn(output, X_tensor)

        loss.backward()

        optimizer.step()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss.item():.6f}"
            )

    # Save model weights
    torch.save(model.state_dict(), save_path)

    print(f"\nModel saved to: {save_path}")

    return model


# ==========================================
# Load Model
# ==========================================

def load_autoencoder(
    input_size,
    hidden_size=64,
    num_layers=2,
    model_path="models/lstm_autoencoder.pth"
):

    model = LSTMAutoencoder(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers
    )

    model.load_state_dict(
        torch.load(model_path)
    )

    model.eval()

    return model


# ==========================================
# Reconstruction Errors
# ==========================================

def get_reconstruction_errors(
    model,
    X_seq
):

    model.eval()

    X_tensor = torch.FloatTensor(X_seq)

    with torch.no_grad():

        reconstructed = model(X_tensor)

    # Mean Squared Error per sequence
    errors = torch.mean(
        (reconstructed - X_tensor) ** 2,
        dim=(1, 2)
    ).numpy()

    return errors


# ==========================================
# Detect Anomalies
# ==========================================

def detect_anomalies(
    errors,
    threshold=None
):

    # Default threshold:
    # mean + 3 standard deviations
    if threshold is None:

        threshold = (
            np.mean(errors)
            + 3 * np.std(errors)
        )

    anomalies = errors > threshold

    return anomalies, threshold

