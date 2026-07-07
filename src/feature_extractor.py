# Build Network Flow Features & State Vector
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_loader import load_all_csvs

df = load_all_csvs()

def build_network_flow_features(df):
    df = df.copy()
    df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace

    labels = df["Label"].copy()
    features = df.drop(columns=["Label"])

    # Clean infinities
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna()

    # Align labels with cleaned features
    labels = labels.loc[features.index]
    binary_labels = labels.apply(lambda x: 0 if x == "BENIGN" else 1)

    # Scale features
    scaler = StandardScaler()
    states = scaler.fit_transform(features)

    return states, binary_labels.to_numpy(), scaler

if __name__ == "__main__":
    states, labels, scaler = build_network_flow_features(df)

    print("States:", states.shape)
    print("Labels:", labels.shape)
    print("Scaler mean:", scaler.mean_)
