import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP
import seaborn as sns


def plot_embeddings(X, y, method='tsne', title=None, save_path=None):

    n_samples = X.shape[0]

    if method == 'tsne':
        perplexity = min(30, (n_samples - 1) // 3)
        print(f"[TSNE] Using perplexity={perplexity}")
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    elif method == 'umap':
        n_neighbors = min(15, n_samples - 1)
        print(f"[UMAP] Using n_neighbors={n_neighbors}")
        reducer = UMAP(n_neighbors=n_neighbors, min_dist=0.3, random_state=42)
    else:
        raise ValueError("method must be 'tsne' or 'umap'")

    X_reduced = reducer.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='coolwarm', s=40, edgecolor='k')
    plt.title(title or f"{method.upper()} Projection", fontsize=14)
    plt.colorbar(scatter, label='Class')
    if save_path:
        plt.savefig(save_path)
        print(f"Saved {method.upper()} plot to: {save_path}")
    plt.close()