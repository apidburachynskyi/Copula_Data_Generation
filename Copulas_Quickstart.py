#%%

from copulas.datasets import sample_trivariate_xyz
from copulas.multivariate import GaussianMultivariate
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
n_components_optimal = 5

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
plt.show()
# %%

df_original = pd.DataFrame(data)
df_copula = pd.DataFrame(samples_copulas)
df_gmm = pd.DataFrame(samples_gmm[0])

# %%

kendall_original = df_original.corr(method='kendall').to_numpy()
kendall_copula = df_copula.corr(method='kendall').to_numpy()
kendall_gmm = df_gmm.corr(method='kendall').to_numpy()

spearman_original = df_original.corr(method='spearman').to_numpy()
spearman_copula = df_copula.corr(method='spearman').to_numpy()
spearman_gmm = df_gmm.corr(method='spearman').to_numpy()

pearson_original = df_original.corr(method='pearson').to_numpy()
pearson_copula = df_copula.corr(method='pearson').to_numpy()
pearson_gmm = df_gmm.corr(method='pearson').to_numpy()

# Compute average over non-diagonal elements
N = kendall_original.shape[0]*(kendall_original.shape[0] - 1)
#%%

kendall_error_copula = np.abs(kendall_original - kendall_copula).sum()/N
kendall_error_gmm = np.abs(kendall_original - kendall_gmm).sum()/N

spearman_error_copula = np.abs(spearman_original - spearman_copula).sum()/N
spearman_error_gmm = np.abs(spearman_original - spearman_gmm).sum()/N

pearson_error_copula = np.abs(pearson_original - pearson_copula).sum()/N
pearson_error_gmm = np.abs(pearson_original - pearson_gmm).sum()/N

print(f"Kendall Mean Error (Copula vs Original): {kendall_error_copula}")
print(f"Kendall Mean Error (GMM vs Original): {kendall_error_gmm}")
print(f"Spearman Mean Error (Copula vs Original): {spearman_error_copula}")
print(f"Spearman Mean Error (GMM vs Original): {spearman_error_gmm}")
print(f"Pearson Mean Error (Copula vs Original): {pearson_error_copula}")
print(f"Pearson Mean Error (GMM vs Original): {pearson_error_gmm}")

# %%
