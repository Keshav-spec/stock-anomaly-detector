import pandas as pd

from sklearn.preprocessing import StandardScaler

from features import (
    add_price_features,
    add_rolling_features,
    FEATURES
)

from models.isolation_forest import (
    train_isolation_forest,
    predict_anomalies
)

from models.lstm_autoencoder import (
    train_autoencoder,
    get_reconstruction_errors,
    detect_anomalies
)

from utils.sequence import create_sequences


def process_features(df):

    df = add_price_features(df)

    df = add_rolling_features(df)

    df = df.dropna().reset_index(drop=True)

    return df


def run_iforest(df):

    X = df[FEATURES]

    model, scaler = train_isolation_forest(X)

    preds, scores = predict_anomalies(
        X,
        model,
        scaler
    )

    return preds, scores


def run_lstm(df):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        df[FEATURES]
    )

    X_seq = create_sequences(
        X_scaled,
        seq_length=20
    )

    model = train_autoencoder(
        X_seq,
        epochs=20
    )

    errors = get_reconstruction_errors(
        model,
        X_seq
    )

    anomalies, threshold = detect_anomalies(
        errors
    )

    preds = [
        -1 if a else 1
        for a in anomalies
    ]

    padded_preds = [1] * 20 + preds

    padded_scores = [0] * 20 + list(errors)

    return padded_preds, padded_scores