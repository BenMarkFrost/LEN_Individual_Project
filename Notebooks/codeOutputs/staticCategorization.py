# %% [markdown]
# # Static Mortality Categorisation
# 
# Benjamin Frost 2022
# 

# %%
import pandas as pd
import numpy as np
import torch.multiprocessing as mp
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, MinMaxScaler
from Categorization import Categorizer
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import copy
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, TensorDataset, random_split
import imblearn
from collections import Counter
from importlib import reload


# %% [markdown]
# ### Loading in the data

# %%
mimicDF = pd.read_csv('../LEN_Test/data/StaticData.csv')
targetDF = mimicDF[['deathperiod']]

ids = mimicDF['PatientID']

mimicDF = mimicDF.drop(columns=['PatientID', 'deathperiod'])

categorisationTypes = {}

# %%
mimicDF.head()

# %% [markdown]
# ### Finding values in need of cleaning

# %%
mimicDF.describe()

# %%
rowsWithNaN = sum(mimicDF.isnull().any(axis=1))
print(f"{mimicDF.shape[0]} rows in df, {rowsWithNaN} containing NaN values")

# %% [markdown]
# ### Missing values dealt with by filling with the mode.

# %%
mimicDF['age'] = mimicDF['age'].fillna(mimicDF['age'].mean())

for col in mimicDF:
    mimicDF[col] = mimicDF[col].fillna(mimicDF[col].mode()[0])

mimicDF['comorbidity'][mimicDF['comorbidity'] < 0] = mimicDF['comorbidity'][mimicDF['comorbidity'] < 0] * -1

# %%
mimicDF.describe()

# %% [markdown]
# ### All missing values filled

# %%
rowsWithNaN = sum(mimicDF.isnull().any(axis=1))
print(f"{mimicDF.shape[0]} rows in df, {rowsWithNaN} containing NaN values")

# %%
targetDF.describe()

# %%
targetDF

# %% [markdown]
# ### Exploring raw target data

# %%
plt.scatter(targetDF['deathperiod'], targetDF['deathperiod'].map(targetDF['deathperiod'].value_counts()))
plt.xlabel("Days after recording data")
plt.ylabel("Frequency")
plt.title("Mortality After Recording Data")
plt.show()

# %%
targetDF['deathperiod'] = targetDF['deathperiod'].apply(lambda x: x if x > -1 else -1)

targetDiedDF = targetDF[targetDF['deathperiod'] > -1]
targetNoDeathDF = targetDF[targetDF['deathperiod'] == -1].apply(lambda x: x+1.0)

# %% [markdown]
# ### Sanity Checks

# %%
targetDiedDF.shape

# %%
targetNoDeathDF.shape

# %% [markdown]
# ### Categorising target data

# %%
bins = 3

cat = Categorizer(targetDiedDF)
targetCategorisedDF = cat.kBins(bins=bins, strategy='quantile')

targetCategorisedDF['deathperiod'] = targetCategorisedDF['deathperiod'].apply(lambda x: x + 1)

targetCategorisedDF.set_index(targetDiedDF.index, inplace=True)

targetCategorisedDF.head()

# %%
targetNoDeathDF.head()

# %% [markdown]
# ### Combining death and no death target data

# %%
combinedTargetDF = targetCategorisedDF.merge(targetNoDeathDF, how='outer', left_index=True, right_index=True).rename(columns={'deathperiod_x': 'deathperiod'})

withDeath = combinedTargetDF.iloc[:,0]
noDeath = combinedTargetDF.iloc[:,1]

newTargetDF = withDeath.fillna(noDeath)

newTargetDF = newTargetDF.astype(np.int64)

newTargetDF

# %% [markdown]
# ### Simple version of target data (Used in the final dataset)

# %%
simpleNewTargetDF = targetDF['deathperiod'].apply(lambda x: 0 if x < 0 else 1)

simpleNewTargetDF

# %%
plt.scatter(targetDF['deathperiod'], targetDF['deathperiod'].map(targetDF['deathperiod'].value_counts()), c=simpleNewTargetDF)
plt.xlabel("Days after recording data")
plt.ylabel("Frequency")
plt.title("Mortality Categorised Simple")
plt.show()

# %%
simpleNewTargetDF.value_counts()

# %% [markdown]
# ### Starting data categorization

# %%
dataNeedingEncodingDF = mimicDF[['los', 'age', 'comorbidity', 'sofa']]

# %% [markdown]
# ### Fixing high age range

# %%
ageWithoutOutliers = dataNeedingEncodingDF['age'][dataNeedingEncodingDF['age'] < 200]

dataNeedingEncodingDF['age'] = dataNeedingEncodingDF['age'].apply(lambda x: x if x < 200 else int(ageWithoutOutliers.sample()))

dataNeedingEncodingDF.head()

# %% [markdown]
# ### Exploring clustering in 2d here, wasn't used in the final dataset

# %%
comorbidity = dataNeedingEncodingDF[['comorbidity']]

comorbidity = pd.DataFrame(data=list(zip(comorbidity.value_counts().index, comorbidity.value_counts().values)), columns=['comorbidity', 'count'])

comorbidity['comorbidity'] = comorbidity['comorbidity'].astype(str).apply(lambda x: x[1:-2]).astype(np.int64)

comorbidityDF = AgglomerativeClustering(n_clusters=5).fit_predict(np.asarray(comorbidity))

sil_x = np.dstack((comorbidity['comorbidity'], comorbidity['count']))[0]

score = silhouette_score(sil_x, comorbidityDF)

fig = plt.figure(figsize=(10,6), dpi=100)
fig.suptitle("2d Cluster Plot", fontsize=20)

plt.scatter(comorbidity['comorbidity'], comorbidity['count'], c=comorbidityDF)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title(f"comorbidity, score: {score}")
plt.show()

# %%
dataNeedingEncodingDF

# %% [markdown]
# # Graphically representing the categorisation

# %%
cat = Categorizer(dataNeedingEncodingDF)

clusters = 3

kBinsDF = cat.kBins(bins=clusters, strategy='quantile')
aggDF = cat.agglomerative(n_clusters=clusters)
kMeansDF = cat.kMeans(n_clusters=clusters)

cat.display(target=simpleNewTargetDF)

# %% [markdown]
# ### Computing the boundaries of the categorised data.

# %%
boundaries = cat.getBoundaries()
for type in boundaries:
    print(f"{type}: {boundaries[type]}")

# %% [markdown]
# ### Labelling categorised data

# %%
categories = {0: 'low', 1: 'medium', 2: 'high'}

mapped = cat.map_types(data={'agg': cat.categorizationTypes['agglomerative']}, mapping=categories)['agg']

mapped

# %%
mapped['Mortality14Days'] = simpleNewTargetDF

mapped['PatientID'] = ids

mapped = mapped.set_index('PatientID')

mapped

# %%
dataname = "staticData.csv"

mapped.to_csv(f"./categorisedData/{dataname}")


