import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP


def plot_embeddings(X, y, method='tsne', title='Embedding', save_path=None):
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'umap':
        reducer = UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'tsne' or 'umap'")

    emb = reducer.fit_transform(X)
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(emb[:, 0], emb[:, 1], c=y, cmap='coolwarm', s=12, alpha=0.8)
    plt.title(f"{method.upper()} - {title}")
    plt.xticks([])
    plt.yticks([])
    plt.legend(*scatter.legend_elements(), title="Label")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
