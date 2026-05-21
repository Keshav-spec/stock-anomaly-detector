from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle

def train_isolation_forest(X, contamination=0.02):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,  # expected % of anomalies
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    # Save model + scaler
    with open("models/iforest.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    return model, scaler

def predict_anomalies(X, model, scaler):
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)       # -1 = anomaly, 1 = normal
    scores = model.score_samples(X_scaled)  # lower = more anomalous
    return preds, scores
