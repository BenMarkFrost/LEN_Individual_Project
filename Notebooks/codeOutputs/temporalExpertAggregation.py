# %% [markdown]
# # Mortality Aggregation
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
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from dask.dataframe import from_pandas
from tsfresh.utilities.distribution import MultiprocessingDistributor
import hashlib 
from sklearn.metrics import precision_recall_fscore_support
from importlib import reload
from temporalHelper import TemporalHelper as TH

# %% [markdown]
# ### Loading in the mimic dataset

# %%
th = TH()

mimicDF = th.get_mimic()

mimicDF

# %%
mimicDF.describe()

# %%
print(f"There are {mimicDF['PatientID'].nunique()} unique patients in the dataset")

# %%
patients = th.get_patients()

print(len(patients))

# %% [markdown]
# ### Aggregating the dataset using expert values

# %%
# Aggregating the dataset

staticPatients = []

target = []

for patient in patients:

    curr = {}

    df = patient.data

    curr['PatientID'] = patient.patientID
    target.append(patient.label)
    curr['ALT'] = df['ALT'].max()
    curr['AST'] = df['AST'].max()
    curr['Admit_Ht'] = df['Admit Ht'].max()
    curr['Albumin'] = df['Albumin'].min()
    curr['Arterial_BP_Mean'] = df['Arterial BP Mean'].min()
    curr['Arterial_BP_Diastolic'] = df['Arterial BP [Diastolic]'].min()
    curr['Arterial_BP_Systolic'] = df['Arterial BP [Systolic]'].min()
    curr['Arterial_PaCO2'] = df['Arterial PaCO2'].min()
    curr['Arterial_PaO2'] = df['Arterial PaO2'].min()
    curr['Arterial_pH_Max'] = df['Arterial pH'].max()
    curr['Arterial_pH_Min'] = df['Arterial pH'].min()
    curr['Urea'] = df['BUN'].min() * 0.357
    curr['CVP_Min'] = df['CVP'].min()
    curr['CVP_Max'] = df['CVP'].max()
    curr['CaO2'] = df['CaO2'].min()
    curr['Chloride'] = df['Chloride'].min()
    curr['Creatinine'] = df['Creatinine'].min()
    curr['Daily_Weight'] = df['Daily Weight'].loc[df['Daily Weight'].first_valid_index()] / 2.205 if df['Daily Weight'].first_valid_index() is not None else None
    curr['Fibrinogen'] = df['Fibrinogen'].max()
    curr['Glucose_Max'] = df['Glucose'].max()
    curr['Glucose_Min'] = df['Glucose'].min()
    curr['Heart_Rate_Min'] = df['Heart Rate'].min()
    curr['Heart_Rate_Max'] = df['Heart Rate'].max()
    curr['Hamoglobin'] = df['Hemoglobin'].min()
    curr['INR'] = df['INR'].max()
    curr['Ionized_Calcium'] = df['Ionized Calcium'].min()
    curr['LDH'] = df['LDH'].max()
    curr['Magnesium'] = df['Magnesium'].min()
    curr['NBP_Mean'] = df['NBP Mean'].min()
    curr['NBP_Diastolic'] = df['NBP [Diastolic]'].min()
    curr['NBP_Systolic'] = df['NBP [Systolic]'].min()
    curr['PTT'] = df['PTT'].max()
    curr['Platelets'] = df['Platelets'].min()
    curr['Potassium_Max'] = df['Potassium'].max()
    curr['Potassium_Min'] = df['Potassium'].min()
    curr['Resp_Rate_(Spont)_Min'] = df['Resp Rate (Spont)'].min()
    curr['Resp_Rate_(Spont)_Max'] = df['Resp Rate (Spont)'].max()
    curr['SVI'] = df['SVI'].min()
    curr['SVRI_Max'] = df['SVRI'].max()
    curr['SVRI_Min'] = df['SVRI'].min()
    curr['SaO2'] = df['SaO2'].min()
    curr['Sodium_Max'] = df['Sodium'].max()
    curr['Sodium_Min'] = df['Sodium'].min()
    curr['SpO2'] = df['SpO2'].min()
    curr['SvO2_Max'] = df['SvO2'].max()
    curr['SvO2_Min'] = df['SvO2'].min()
    curr['Temperature_C_Max'] = df['Temperature C'].max()
    curr['Temperature_C_Min'] = df['Temperature C'].min()
    curr['Bilirubin'] = df['Total Bili'].max()
    curr['White_Blood_Cells_Max'] = df['WBC'].max()
    curr['White_Blood_Cells_Min'] = df['WBC'].min()

    staticPatients.append(curr)


staticPatientsDF = pd.DataFrame([x.values() for x in staticPatients], columns=curr.keys())

staticPatientsDF = staticPatientsDF.set_index('PatientID')

targetSeries = pd.Series(data=target)

# %%
staticPatientsDF.describe()

# %%
rowsWithNaN = sum(staticPatientsDF.isnull().any(axis=1))
print(f"{staticPatientsDF.shape[0]} rows in df, {rowsWithNaN} containing NaN values")

# %% [markdown]
# ### Filling the missing values from aggregated data

# %%
fillNaModeDF = staticPatientsDF.copy()

for col in fillNaModeDF:
    fillNaModeDF[col] = fillNaModeDF[col].fillna(fillNaModeDF[col].mean())

fillNaModeDF['Admit_Ht'][fillNaModeDF["Admit_Ht"] > 100] = fillNaModeDF["Admit_Ht"].mean()

display(fillNaModeDF)

# %% [markdown]
# ### Binning with K-Bins

# %%
import Categorization

reload(Categorization)

cat = Categorization.Categorizer(fillNaModeDF)

cat.kBins(bins = 3)

boundaries = cat.getBoundaries()

display(boundaries['kBins'])

cat.display(num=6)

# %% [markdown]
# ### Labelling high to low

# %%
categories = {0: 'very_low', 1: 'low', 2: 'medium', 3: 'high', 4: 'very_high'}

cat.map_types(mapping=categories)

mapped = cat.mappedTypes['kBins']

mapped

# %%
targetSeries.value_counts()

# %%
mapped['Mortality14Days'] = targetSeries.values

mapped.to_csv("./categorisedData/expertLabelledData.csv")


