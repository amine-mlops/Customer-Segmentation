from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def elbow_method(X_scaled, k_range=range(1,11)):
    wcss = []

    for k in k_range:
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    return wcss


def find_optimal_k(X_scaled, k_range=range(2,11)):

    results = []

    for k in k_range:
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        labels = kmeans.fit_predict(X_scaled)
        results.append(silhouette_score(X_scaled, labels))

    return results

def train_kmeans(X_scaled, k = 5):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    return model, labels
