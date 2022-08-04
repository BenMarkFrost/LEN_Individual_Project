# %% [markdown]
# # Mortalty Metric Extraction
# 
# Benjamin Frost 2022
# 

# %%
import pandas as pd
import numpy as np
import torch.multiprocessing as mp
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.interpolate import interp1d
from Categorization import Categorizer
import torch
import copy
from torch.nn.functional import one_hot
import imblearn
from collections import Counter
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.metrics import silhouette_score
from tsfresh import extract_features, select_features, extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute
from dask.dataframe import from_pandas
from tsfresh.utilities.distribution import MultiprocessingDistributor
import hashlib 
from sklearn.metrics import precision_recall_fscore_support
from importlib import reload
from temporalHelper import TemporalHelper as TH

# %% [markdown]
# ### Loading in the time-series dataset

# %%
th = TH()

mimicDF = th.get_mimic()

mimicDF

# %%
mimicDF.describe()

# %%
print(f"There are {mimicDF['PatientID'].nunique()} unique patients in the dataset")

# %% [markdown]
# ### Defining the hashing function

# %%
def find_cached(df=None, my_hash=None):

    if my_hash is None:
        my_hash = hashlib.sha256(bytes(str(df), 'utf-8')).hexdigest()

    display(my_hash)


    try:
        cachedDF = pd.read_csv("./processingCache/" + my_hash + ".csv").set_index("Unnamed: 0")

        print("Using cached df")

        return cachedDF, my_hash

    except:

        return False, my_hash
    

# %%
# Fixing 'arterial pH', 'ionized calcium' since they contain erroneous values.

mimicDF['Arterial pH'][mimicDF['Arterial pH'] > 8] = mimicDF['Arterial pH'].mean()
mimicDF['Ionized Calcium'][mimicDF['Ionized Calcium'] > 8] = mimicDF['Ionized Calcium'].mean()


# %% [markdown]
# ### Calculating the 'missingness' of columns

# %%

patients = th.get_patients(mimicDF)

patientsKept, columnsExplored = th.count_null(patients)

patients = th.get_top_columns(patients, 12)

# %%
patients[0].topColumns

# %%
# tsfresh can't handle missing values. So we'll interpolate them.

for patient in tqdm(patients):
    patient.cleanedDF = copy.copy(patient.topColumns)
    for col in patient.cleanedDF:
        nonNullCount = patient.cleanedDF[col].count()

        if (nonNullCount >= 3):
            # Polynomial if there are >=3 non-null values
            patient.cleanedDF[col] = patient.cleanedDF[col].interpolate(method='polynomial', order=2)
        elif nonNullCount == 1:
            # Pad with three values either side if only 1 data point exists.
            patient.cleanedDF[col] = patient.cleanedDF[col].interpolate(method='linear', limit_direction='both', limit=3)
        else:
            # Otherwise, we can linearly interpolate.
            patient.cleanedDF[col] = patient.cleanedDF[col].interpolate(method='linear', limit_direction='both', limit_area='inside')

        # Fill in the start and end with the closest data point.
        patient.cleanedDF[col] = patient.cleanedDF[col].interpolate(method='linear', limit_direction='both', limit_area='outside')

    patient.cleanedDF["PatientID"] = patient.patientID
    

formattedMimicDF = pd.concat([patient.cleanedDF for patient in patients])


display(formattedMimicDF)

# %% [markdown]
# Display interpolation and fillna results here

# %%
formattedMimicDF.columns

# %% [markdown]
# ### Visualise the interpolation

# %%
for patient in patients[:2]:


    print(patient.patientID)


    step = 6

    columns = list(patient.cleanedDF.columns)[:-1]

    for idx in range(0, len(columns), step):


        fig = plt.figure(figsize = (30, 6), dpi=100)

        cols = columns[idx:idx+step]


        for k, col in enumerate(cols):

            plt.subplot(2, step, k+1)

            plt.plot(patient.cleanedDF.index, patient.cleanedDF[col])
            plt.scatter(patient.cleanedDF.index, patient.cleanedDF[col])
            plt.scatter(patient.cleanedDF.index, patient.data[col])
            plt.title(f"{col}")

    plt.show()

# %%

targetSeries = pd.Series(data=[patient.label for patient in patients], index=[patient.patientID for patient in patients])


targetSeries


# %%
formattedMimicDF

# %% [markdown]
# ### Caching to get he extracted features since processing can take a while

# %%
## Checking if this DF already has cached time series features. Will save approx 15 mins if so.

# my_hash = hashlib.sha256(pd.util.hash_pandas_object(formattedMimicDF, index=True).values).hexdigest()

my_hash = "26128f35758669a6e2001768a98df93bff7b0b8ac8fdf241d94641c49a83b901"

filtered_features, my_hash = find_cached(formattedMimicDF, my_hash = my_hash)


if filtered_features is False:

    filtered_features = extract_relevant_features(formattedMimicDF, y=targetSeries, column_id="PatientID", n_jobs=4)

    filtered_features.to_csv("./processingCache/" + my_hash + ".csv")

# %% [markdown]
# ### Adding back the patient ID column, visualising the data

# %%
filtered_features = filtered_features.reindex(formattedMimicDF['PatientID'].unique())

filtered_features['PatientID'] = formattedMimicDF['PatientID'].unique()

filtered_features = filtered_features.set_index('PatientID')

display(filtered_features)

# %%
# These are the columns deemed most impactful on the prediction.

for col in filtered_features.columns:
    print(col)

# %% [markdown]
# ### Agglomerative clustering and labelling as seen in static dataset

# %%
cat = Categorizer(filtered_features)

cat.agglomerative(n_clusters=3)

cat.display(target=targetSeries)

categories = {0: 'very_low', 1: 'low', 2: 'medium', 3: 'high', 4: 'very_high'}

mapped = cat.map_types(mapping=categories)['agglomerative']

# %%
mapped['Mortality14Days'] = targetSeries

mapped.to_csv("./categorisedData/metricExtractedData.csv")


