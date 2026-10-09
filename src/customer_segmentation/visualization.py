
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_elbow(k_values, wcss):
    """Plot WCSS values to help choose the number of clusters."""
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_values), wcss, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS / Inertia")
    plt.title("Elbow Method")
    plt.xticks(list(k_values))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_silhouette(k_values, silhouette_scores):
    """Plot silhouette scores for different numbers of clusters."""
    plt.figure(figsize=(8, 5))

    plt.plot(
        list(k_values),
        silhouette_scores,
        marker="o",
        color="teal"
    )

    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis")
    plt.xticks(list(k_values))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_clusters(X, labels, centroids=None):
    """Plot customer clusters with centroids and a complete legend."""
    plt.figure(figsize=(10, 6))

    cluster_ids = sorted(set(labels))

    scatter = plt.scatter(
        X.iloc[:, 0],
        X.iloc[:, 1],
        c=labels,
        cmap="viridis",
        s=60,
        alpha=0.8
    )

    # Create a legend entry for every cluster
    colors = [
        scatter.cmap(scatter.norm(cluster_id))
        for cluster_id in cluster_ids
    ]

    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=9,
            label=str(cluster_id)
        )
        for cluster_id, color in zip(cluster_ids, colors)
    ]

    # Plot centroids as black downward-pointing triangles
    if centroids is not None:
        plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="v",
            s=130,
            c="black",
            label="Centroids",
            zorder=3
        )

        legend_handles.append(
            Line2D(
                [0], [0],
                marker="v",
                linestyle="None",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=10,
                label="Centroids"
            )
        )

    plt.legend(
        handles=legend_handles,
        loc="center right",
        frameon=True
    )

    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Customer Segments Using K-Means")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()