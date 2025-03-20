import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_kde_comparison(x_samples, y_samples, p_y=None, cmap_x='plasma', cmap_y='coolwarm', left_title=None):
    """
    Plots KDE for x_samples and y_samples. If p_y is provided, it is used instead of KDE for y_samples.
    
    Parameters:
        x_samples (np.ndarray): Samples for X.
        y_samples (np.ndarray): Samples for Y.
        p_y (callable or None): Function taking (X, Y) and returning density values.
        cmap_x (str): Colormap for X samples KDE.
        cmap_y (str): Colormap for Y samples KDE.
    """
    
    def plot_kde(samples, ax, cmap, density_fn=None):
        x, y = samples[:, 0], samples[:, 1]
        
        # Define grid
        xmin, xmax = x.min() - 1, x.max() + 1
        ymin, ymax = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
        
        # Compute density
        if density_fn:
            zz = density_fn(xx, yy)
        else:
            kde = gaussian_kde(np.vstack([x, y]))
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        
        # Plot
        ax.contourf(xx, yy, zz, levels=30, cmap=cmap, alpha=0.8)
        ax.scatter(x, y, s=5, color='black', alpha=0.3)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot KDE for x_samples
    plot_kde(x_samples, axes[0], cmap_x)
    axes[0].set_title("KDE of X Samples") if left_title is None else axes[0].set_title(left_title)

    # Plot KDE or given density for y_samples
    density_fn = (lambda X, Y: p_y(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)) if p_y else None
    plot_kde(y_samples, axes[1], cmap_y, density_fn)
    axes[1].set_title("KDE of Y Samples" if p_y is None else "Given Density for Y")

    plt.tight_layout()
    plt.show()

