import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copulas.datasets import sample_trivariate_xyz
from copulas.multivariate import GaussianMultivariate
from sklearn.mixture import GaussianMixture

# Uncomment below to render in LateX.

# plt.rcParams.update({
#        "text.usetex": True,
#        "font.family": "serif",
#        "font.serif": ["Computer Modern Roman"],
#    })

# Constants
IMAGE_DIR = 'images/'
N_COMPONENTS_RANGE = range(1, 30)
N_SAMPLES = 1000


# Utility Functions

def plot_2d_comparison(data, samples_copulas, samples_gmm, x_idx, y_idx, title, filename):
    """Plot and save 2D comparison."""
    plt.scatter(samples_copulas[:, x_idx], samples_copulas[:, y_idx], label='Copulas', alpha=0.6)
    plt.scatter(samples_gmm[:, x_idx], samples_gmm[:, y_idx], label='GMM', alpha=0.6)
    plt.scatter(data[:, x_idx], data[:, y_idx], label='Original Data', alpha=0.6)
    plt.legend()
    plt.title(title)
    plt.savefig(f"{IMAGE_DIR}{filename}")
    plt.show()
    plt.close()


def plot_3d_comparison(data, samples_copulas, samples_gmm, title, filename):
    """Plot and save 3D comparison."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(samples_copulas[:, 0], samples_copulas[:, 1], samples_copulas[:, 2], label='Copulas', alpha=0.6)
    ax.scatter(samples_gmm[:, 0], samples_gmm[:, 1], samples_gmm[:, 2], label='GMM', alpha=0.6)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], label='Original Data', alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title(title)
    plt.savefig(f"{IMAGE_DIR}{filename}")
    plt.show()
    plt.close()


# Main Process
if __name__ == "__main__":
    # Load and transform data
    data = (-2) * sample_trivariate_xyz()
    data = data.to_numpy()

    # Fit models
    copula = GaussianMultivariate()
    copula.fit(data)

    aic_scores = []
    for n in N_COMPONENTS_RANGE:
        gmm = GaussianMixture(n_components=n, random_state=0)
        gmm.fit(data)
        aic_scores.append(gmm.aic(data))

    n_components_optimal = N_COMPONENTS_RANGE[aic_scores.index(min(aic_scores))]
    gmm = GaussianMixture(n_components=n_components_optimal, random_state=0)
    gmm.fit(data)

    # Generate samples
    samples_copulas = copula.sample(N_SAMPLES).to_numpy()
    samples_gmm = gmm.sample(N_SAMPLES)[0]

    # 2D Comparisons
    plot_2d_comparison(data, samples_copulas, samples_gmm, 0, 1, '2D Comparison: X-Y', '2D_Comparison_X-Y.png')
    plot_2d_comparison(data, samples_copulas, samples_gmm, 1, 2, '2D Comparison: Y-Z', '2D_Comparison_Y-Z.png')
    plot_2d_comparison(data, samples_copulas, samples_gmm, 0, 2, '2D Comparison: X-Z', '2D_Comparison_X-Z.png')

    # 3D Comparison
    plot_3d_comparison(data, samples_copulas, samples_gmm, '3D Comparison: X-Y-Z', '3D_Comparison_X-Y-Z.png')

    # Save DataFrames
    df_original = pd.DataFrame(data)
    df_copula = pd.DataFrame(samples_copulas)
    df_gmm = pd.DataFrame(samples_gmm)
