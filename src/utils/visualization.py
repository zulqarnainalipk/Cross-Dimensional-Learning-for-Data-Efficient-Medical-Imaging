"""
Visualization utilities for cross-dimensional transfer learning.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np
import torch
from typing import Optional, List, Dict
from sklearn.manifold import TSNE
import os

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def save_figure(
    fig: plt.Figure,
    filename: str,
    directory: str = 'results/figures'
) -> str:
    """
    Save figure to file.

    Args:
        fig: Matplotlib figure object
        filename: Output filename
        directory: Output directory

    Returns:
        Full path to saved figure
    """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight', format='png')
    plt.close(fig)
    return filepath


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Training History"
) -> plt.Figure:
    """
    Plot training history with loss and accuracy curves.

    Args:
        history: Dictionary containing training metrics
        save_path: Optional path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss curves
    ax1 = axes[0, 0]
    if 'train_loss' in history:
        ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2 = axes[0, 1]
    if 'train_acc' in history:
        ax2.plot(history['train_acc'], label='Train Acc', linewidth=2)
    if 'val_acc' in history:
        ax2.plot(history['val_acc'], label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # AUC curves
    ax3 = axes[1, 0]
    if 'train_auc' in history:
        ax3.plot(history['train_auc'], label='Train AUC', linewidth=2)
    if 'val_auc' in history:
        ax3.plot(history['val_auc'], label='Val AUC', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('AUC-ROC')
    ax3.set_title('AUC-ROC Curves')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Domain alignment loss
    ax4 = axes[1, 1]
    if 'domain_loss' in history:
        ax4.plot(history['domain_loss'], label='Domain Loss', linewidth=2,
                 color='purple')
    if 'semantic_loss' in history:
        ax4.plot(history['semantic_loss'], label='Semantic Loss', linewidth=2,
                 color='orange')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('Alignment Losses')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """
    Plot confusion matrix heatmap.

    Args:
        cm: Confusion matrix array (C, C)
        class_names: List of class names
        save_path: Optional path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Count'}
    )

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_feature_space(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    title: str = "Feature Space Visualization",
    method: str = 'tsne',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize embedding feature space using t-SNE or UMAP.

    Args:
        embeddings: Feature embeddings (N, D)
        labels: Class labels (N,)
        title: Plot title
        method: Dimensionality reduction method ('tsne', 'umap')
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Convert to numpy
    emb_np = embeddings.detach().cpu().numpy()
    lbl_np = labels.detach().cpu().numpy()

    # Dimensionality reduction
    if emb_np.shape[1] > 2:
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb_np)-1))
            emb_2d = reducer.fit_transform(emb_np)
        else:
            # UMAP-like using sklearn (simplified)
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(50, emb_np.shape[1]))
            emb_reduced = pca.fit_transform(emb_np)
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb_reduced)-1))
            emb_2d = reducer.fit_transform(emb_reduced)
    else:
        emb_2d = emb_np

    # Plot
    unique_labels = np.unique(lbl_np)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = lbl_np == label
        ax.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            c=[colors[i]],
            label=f'Class {label}',
            alpha=0.7,
            s=50
        )

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_attention_maps(
    attention_weights: torch.Tensor,
    save_path: Optional[str] = None,
    title: str = "Cross-Dimensional Attention"
) -> plt.Figure:
    """
    Plot cross-dimensional attention weight maps.

    Args:
        attention_weights: Attention weights (H, W) or (H, W, D)
        save_path: Optional path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    if attention_weights.ndim == 3:
        # Multi-head attention - show first head
        attn = attention_weights[0].detach().cpu().numpy()
    else:
        attn = attention_weights.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(attn, cmap='viridis', aspect='auto')
    ax.set_xlabel('Query Position')
    ax.set_ylabel('Key Position')
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label='Attention Weight')

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_domain_alignment(
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    source_labels: Optional[torch.Tensor] = None,
    target_labels: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    title: str = "Domain Alignment Visualization"
) -> plt.Figure:
    """
    Visualize domain alignment between source and target embeddings.

    Args:
        source_embeddings: Source domain embeddings (N, D)
        target_embeddings: Target domain embeddings (M, D)
        source_labels: Source domain labels (optional)
        target_labels: Target domain labels (optional)
        save_path: Optional path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Combine embeddings for t-SNE
    all_emb = torch.cat([source_embeddings, target_embeddings], dim=0)
    domain_labels = torch.cat([
        torch.zeros(len(source_embeddings)),
        torch.ones(len(target_embeddings))
    ])

    # Plot domain alignment
    ax1 = axes[0]
    plot_feature_space(
        all_emb,
        domain_labels,
        title=f'{title} (t-SNE)',
        method='tsne'
    )

    # Plot class alignment if labels available
    ax2 = axes[1]
    if source_labels is not None and target_labels is not None:
        combined_labels = torch.cat([source_labels, target_labels])
        plot_feature_space(
            all_emb,
            combined_labels,
            title=f'{title} by Class (t-SNE)',
            method='tsne'
        )
    else:
        ax2.axis('off')

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_proto_visualization(
    support_embeddings: torch.Tensor,
    support_labels: torch.Tensor,
    query_embeddings: torch.Tensor,
    query_predictions: torch.Tensor,
    prototypes: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    title: str = "Prototypical Network Visualization"
) -> plt.Figure:
    """
    Visualize prototypical network decision boundaries.

    Args:
        support_embeddings: Support set embeddings (S, D)
        support_labels: Support set labels (S,)
        query_embeddings: Query set embeddings (Q, D)
        query_predictions: Query set predictions (Q,)
        prototypes: Prototype embeddings (C, D) if computed
        save_path: Optional path to save figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Reduce to 2D
    all_emb = torch.cat([support_embeddings, query_embeddings], dim=0)
    all_labels = torch.cat([support_labels, query_predictions])

    # Use t-SNE for visualization
    emb_np = all_emb.detach().cpu().numpy()
    lbl_np = all_labels.detach().cpu().numpy()

    if emb_np.shape[1] > 2:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb_np)-1))
        emb_2d = reducer.fit_transform(emb_np)
    else:
        emb_2d = emb_np

    # Plot support points with different markers
    n_support = len(support_embeddings)
    support_2d = emb_2d[:n_support]
    query_2d = emb_2d[n_support:]

    unique_classes = np.unique(lbl_np)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))

    # Plot prototypes if available
    if prototypes is not None:
        proto_np = prototypes.detach().cpu().numpy()
        if proto_np.shape[1] > 2:
            proto_reducer = TSNE(n_components=2, random_state=42, perplexity=min(5, len(proto_np)))
            proto_2d = proto_reducer.fit_transform(proto_np)
        else:
            proto_2d = proto_np

        for i, c in enumerate(unique_classes):
            ax.scatter(
                proto_2d[i, 0], proto_2d[i, 1],
                marker='*', s=300, c=[colors[i]],
                edgecolors='black', linewidth=2,
                label=f'Prototype {c}', zorder=5
            )

    # Plot support points
    support_labels_np = support_labels.detach().cpu().numpy()
    for i, c in enumerate(unique_classes):
        mask = support_labels_np == c
        if mask.sum() > 0:
            ax.scatter(
                support_2d[mask, 0], support_2d[mask, 1],
                marker='o', s=100, c=[colors[i]],
                edgecolors='black', linewidth=1.5,
                label=f'Support {c}'
            )

    # Plot query points with different markers based on prediction
    for i, c in enumerate(unique_classes):
        mask = query_predictions.detach().cpu().numpy() == c
        if mask.sum() > 0:
            ax.scatter(
                query_2d[mask, 0], query_2d[mask, 1],
                marker='x', s=80, c=[colors[i]],
                label=f'Query (pred {c})'
            )

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig
