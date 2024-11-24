#%%

from copulas.datasets import sample_trivariate_xyz
from copulas.multivariate import GaussianMultivariate
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Constants
IMAGE_DIR = 'images/'
N_COMPONENTS_RANGE = range(1, 30)
N_SAMPLES = 1000

#%%

data = (-2)*sample_trivariate_xyz()
copula = GaussianMultivariate()

# Computation of the AIC for fiding the optimal number of components of the GMM

n_components_range = range(1, 30)
aic_scores = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(data)
    aic_scores.append(gmm.aic(data))

# %%

n_components_optimal = n_components_range[aic_scores.index(min(aic_scores))]

# %%
gmm = GaussianMixture(n_components=n_components_optimal, random_state=0)

# %%
gmm.fit(data)
copula.fit(data)
# %%

samples_copulas = copula.sample(1000)
samples_gmm = gmm.sample(1000)

# %%

samples_copulas = samples_copulas.to_numpy()
data = data.to_numpy()

#%%

plt.scatter(samples_copulas[:, 0], samples_copulas[:, 1], label='Copulas')
plt.scatter(samples_gmm[0][:, 0], samples_gmm[0][:, 1], label='GMM')
plt.scatter(data[:, 0], data[:, 1], label='Data')
plt.legend()
plt.title('2D Comparison of Original Data, Copulas, and GMM, X-Y')
# save the figure
plt.savefig('images/2D_Comparison_X-Y.png')
plt.show()

#%%

plt.scatter(samples_copulas[:, 1], samples_copulas[:, 2], label='Copulas')
plt.scatter(samples_gmm[0][:, 1], samples_gmm[0][:, 2], label='GMM')
plt.scatter(data[:, 1], data[:, 2], label='Data')
plt.legend()
plt.title('2D Comparison of Original Data, Copulas, and GMM, Y-Z')
plt.savefig('images/2D_Comparison_Y-Z.png')
plt.show()

#%%

plt.scatter(samples_copulas[:, 0], samples_copulas[:, 2], label='Copulas')
plt.scatter(samples_gmm[0][:, 0], samples_gmm[0][:, 2], label='GMM')
plt.scatter(data[:, 0], data[:, 2], label='Data')
plt.legend()
plt.title('2D Comparison of Original Data, Copulas, and GMM, X-Z')
plt.savefig('images/2D_Comparison_X-Z.png')
plt.show()

#%%

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the three datasets
ax.scatter(samples_copulas[:, 0], samples_copulas[:, 1], samples_copulas[:, 2], 
          label='Copulas', alpha=0.6)
ax.scatter(samples_gmm[0][:, 0], samples_gmm[0][:, 1], samples_gmm[0][:, 2], 
          label='GMM', alpha=0.6)
ax.scatter(data[:, 0], data[:, 1], data[:, 2], 
          label='Data', alpha=0.6)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('3D Comparison of Original Data, Copulas, and GMM, X-Y-Z')
plt.savefig('images/3D_Comparison_X-Y-Z.png')
plt.show()

# %%

df_original = pd.DataFrame(data)
df_copula = pd.DataFrame(samples_copulas)
df_gmm = pd.DataFrame(samples_gmm[0])
