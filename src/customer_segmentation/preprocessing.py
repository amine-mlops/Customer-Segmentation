from sklearn.preprocessing import StandardScaler


def scale_features(X):
    """Standardize customer features."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler