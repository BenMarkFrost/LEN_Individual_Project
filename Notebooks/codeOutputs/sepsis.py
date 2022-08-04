# %% [markdown]
# # Sepsis data - Clustering
# 
# Benjamin Frost 2022

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
import random
import Categorization
import torch
import copy
from torch.nn.functional import one_hot
import imblearn
from collections import Counter
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from dask.dataframe import from_pandas
from tsfresh.utilities.distribution import MultiprocessingDistributor
import hashlib 
from sklearn.metrics import precision_recall_fscore_support
from importlib import reload
from temporalHelper import TemporalHelper as TH
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import os



# %% [markdown]
# ### Loading in the data

# %%
sepsisDF = pd.read_csv('../LEN_Test/data/sepsis_data.csv')

sepsisDF

# %% [markdown]
# ### Investigating data

# %%
# Too many columns to display all in one cell.

step = 10

for idx in range(0, len(sepsisDF.columns), step):
    tempCols = sepsisDF[sepsisDF.columns[idx:idx+step]]
    display(tempCols.describe())

# %%
print(f"There are {sepsisDF['Patient_id'].nunique()} unique patients in the dataset")

# %%
print(sepsisDF['SepsisLabel'].value_counts())

# %% [markdown]
# ### Splitting data by patient

# %%
th = TH()

staticColumns = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]

patients = th.get_patients(sepsisDF, by="Patient_id", label="SepsisLabel", static = staticColumns)

# %% [markdown]
# ### Sanity check

# %%
patients[0].data.head()

# %%
patients[0].static.head()

# %%
totalNullColumns = 0

for patient in tqdm(patients):
    totalNullColumns += patient.data.isnull().all().sum()

totalColumns = len(patient.data.columns) * len(patients)

print(totalColumns, totalNullColumns)

print(f"{np.round(totalNullColumns / totalColumns * 100, 2)}% of columns are null")

# %% [markdown]
# ### Counting missingness

# %%
patientsKept, columnsExplored = th.count_null(patients)

# %% [markdown]
# #### Sharp drop off after 15 columns so will keep around 34000 patients with at least some data in the top 15 columns

# %%
clusteringPatients = th.get_top_columns(patients, 15)

print(len(clusteringPatients))

clusteringPatients[3].topColumns.head()

# %% [markdown]
# ### Sanity Check

# %%
totalNullColumns = 0

for patient in tqdm(clusteringPatients):
    totalNullColumns += patient.topColumns.isnull().all().sum()

totalColumns = len(patient.topColumns.columns) * len(clusteringPatients)

print(totalColumns, totalNullColumns)

print(f"{np.round(totalNullColumns / totalColumns * 100, 2)}% of columns are null")

# %% [markdown]
# ### Interpolating missing data

# %%
noInterpolation = 0
failureExample = (0,0)

maxPatientLen = 0

tempCP = copy.deepcopy(clusteringPatients)


# Finding the longest patient's time series. Other patients will be extended to match this length.

for idx, patient in (enumerate(tempCP)):

    if patient.data.shape[0] > maxPatientLen:
        maxPatientLen = patient.data.shape[0]

print("Max time series length: " + str(maxPatientLen))


for patient in tqdm(tempCP):

    # Padding patients to match the longest length.
    if patient.topColumns.shape[0] < maxPatientLen:

        fixedData = []
        for col in patient.topColumns:
            fixedData.append(np.pad(patient.topColumns[col], (0, maxPatientLen - patient.topColumns[col].shape[0]), 'constant', constant_values=np.nan))

        tempDF = pd.DataFrame(data = fixedData).T
        tempDF.columns = patient.topColumns.columns
        patient.topColumns = tempDF

    patientNonNullCount = patient.topColumns.count()


    # Interpolate with polynomial regression
    for column in patient.topColumns.columns:

        try:
            patient.interpolatedData[column] = patient.topColumns[column].interpolate(method='polynomial', order=2, limit_direction='both', limit_area='inside')

    
        except ValueError:

            try:

                ## Use linear interpolation if polynomial fails
                if patientNonNullCount[column] == 1:
                    patient.interpolatedData[column] = patient.topColumns[column].interpolate(method='linear', limit_direction='both', limit=3)
                else: 
                    patient.interpolatedData[column] = patient.topColumns[column].interpolate(method='linear', limit_direction='both', limit_area='inside')
            
            except ValueError:

                patient.interpolatedData[column] = patient.topColumns[column].fillna(patient.topColumns[column].mean())
                noInterpolation += 1
    

print(f"{noInterpolation}/{len(clusteringPatients)} patients failed to interpolate")


# %%
clusteringPatients = tempCP

# %%
clusteringPatients[4].interpolatedData

# %% [markdown]
# ### Checkpointing the processing here

# %%
withIDs = []

for patient in clusteringPatients:
    tempP = copy.deepcopy(patient)
    tempP.interpolatedData['PatientID'] = tempP.patientID
    tempP.interpolatedData['Mortality14Days'] = tempP.label
    withIDs.append(tempP)


cleanedTimeSeriesDF = pd.concat([patient.interpolatedData for patient in withIDs])

cleanedTimeSeriesDF = cleanedTimeSeriesDF.set_index('PatientID')

cleanedTimeSeriesDF.to_csv("cleanedTemporalSepsisData.csv")

# %% [markdown]
# # Reload cached interpolated data from here 
# 
# #### Saves about 20 mins of processing

# %%
# Reading the original dataframe
sepsisDF = pd.read_csv('../LEN_Test/data/sepsis_data.csv')

th = TH()

staticColumns = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]

# Split the original dataframe into patients
patients = th.get_patients(sepsisDF, by="Patient_id", label="SepsisLabel", static = staticColumns)

patientsDict = {patient.patientID : patient for patient in patients}

# Read the cached preprocessing data
cleanedTimeSeriesDF = pd.read_csv("./processingCache/cleanedTemporalSepsisData.csv")


loadedIDs = set(cleanedTimeSeriesDF['PatientID'].unique())

# Selecting only the patients that were selected during feature selection
for id, group in tqdm(cleanedTimeSeriesDF.groupby("PatientID")):
    label = group.iloc[0]['Mortality14Days']
    group = group.drop(columns=['PatientID', 'Mortality14Days'], axis=1)
    patientsDict[id].interpolatedData = group

clusteringPatients = [patientsDict[id] for id in patientsDict if id in loadedIDs]

# %% [markdown]
# ### Sanity Check

# %%
print(len(clusteringPatients))

# %%
# The below graphing can only be run if caching has not been enabled.

# patient = clusteringPatients[1]

# fig = plt.figure(figsize = (30, 8), dpi=200)


# fig.suptitle(f"Patient: {patient.patientID}", fontsize=30)

# for idx, col in enumerate(patient.topColumns.columns):
#     plt.subplot(2, (len(patient.topColumns.columns)//2)+1, idx+1)

#     plt.scatter(patient.topColumns.index, patient.topColumns[col], c='Orange')
#     plt.title(f"{col}", fontsize=20)

# plt.tight_layout()
# plt.show()

# %%
for patient in clusteringPatients[:1]:

    # display(clusteringPatients[i].interpolatedData.head())

    fig = plt.figure(figsize = (30, 8),dpi=200)


    fig.suptitle(f"Patient: {patient.patientID}", fontsize=30)

    for idx, col in enumerate(patient.interpolatedData.columns):
        plt.subplot(2, (len(patient.interpolatedData.columns)//2)+1, idx+1)

        plt.plot(patient.interpolatedData.index, patient.interpolatedData[col])
        plt.scatter(patient.interpolatedData.index, patient.interpolatedData[col])
        # plt.scatter(patient.topColumns.index, patient.topColumns[col], c="Orange")
        plt.title(f"{col}", fontsize=20)

    plt.tight_layout()
    plt.show()

# %%
minorityClass = [patient for patient in clusteringPatients if patient.label == 1]
majorityClass = [patient for patient in clusteringPatients if patient.label == 0]

print(len(minorityClass))
print(len(majorityClass))

# %% [markdown]
# ### Helper functions for clustering

# %%

def formatForTimeSeries(column, sampleSize=None):

    # Getting an even split of target classes
    if sampleSize is None:
        sampleList = clusteringPatients
    else:
        minorityClass = [patient for patient in clusteringPatients if patient.label == 1]
        majorityClass = [patient for patient in clusteringPatients if patient.label == 0]
        
        minorityList = random.choices(minorityClass, k=sampleSize//2)
        majorityList = random.choices(majorityClass, k=sampleSize//2)

        sampleList = minorityList + majorityList

        

    print("Creating stacked DF...")
    stackedDF = pd.DataFrame([patient.interpolatedData[column].values for patient in sampleList])


    stackedNumpy = stackedDF.to_numpy()

    cleanedNumpy = []

    print("Cleaning")
    for row in stackedNumpy:
        cleanedNumpy.append(row[~np.isnan(row)])


    dataFormatted = to_time_series_dataset([*cleanedNumpy])

    return dataFormatted



def timeSeriesCluster(clusters, dataFormatted):

    print("Clustering")

    model = TimeSeriesKMeans(n_clusters=clusters, tol=1e-1, metric="dtw", max_iter=1, random_state=0, n_jobs=4)
    y_pred = model.fit_predict(dataFormatted)


    return y_pred, model


# %% [markdown]
# ### Caching for clustering

# %%
def find_cached(df=None, hash=None):

    if hash is None:

        print("Hashing...")
        hash = hashlib.sha256(bytes(str(df), 'utf-8')).hexdigest()

    display(hash)


    try:
        cachedDF = pd.read_csv("./processingCache/" + hash + ".csv").set_index("PatientID")

        print("Using cached df")

        return cachedDF, hash

    except:

        print("No cached df found")

        return False, hash
    

# %%
myHash = "Chosen_clusters_sepsis"


clusteredDF, myHash = find_cached(clusteringPatients, hash=myHash)

if clusteredDF is False:

    clusteredDF = pd.DataFrame()

    for column in tqdm(clusteringPatients[0].interpolatedData.columns):
        dataFormatted = formatForTimeSeries(column, 1000)
        y_pred, model = timeSeriesCluster(2, dataFormatted)

        print("Finished fitting. Predicting... ")

        dataFormattedAll = formatForTimeSeries(column)
        y_pred = model.predict(dataFormattedAll)
        clusteredDF[column] = y_pred

    ids = [patient.patientID for patient in clusteringPatients]

    clusteredDF["PatientID"] = ids

    clusteredDF = clusteredDF.set_index("PatientID")


    clusteredDF.to_csv("./processingCache/" + myHash + ".csv")



# %% [markdown]
# # Reload clustered cached data from here
# 
# ### Saves about 1.5 hours of processing

# %%
# sepsisDF = pd.read_csv('../LEN_Test/data/sepsis_data.csv')

# th = TH()

# staticColumns = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]

# patients = th.get_patients(sepsisDF, by="Patient_id", label="SepsisLabel", static = staticColumns)

# clusteredDF = pd.read_csv("./processingCache/Chosen_clusters_sepsis.csv").set_index("PatientID")

# clusteredDF = clusteredDF.set_index("PatientID")


# %%
# fig = plt.figure(figsize = (15, 12), dpi=200)

# for idx, col in enumerate(colScores):
#     scores = [x[1] for x in colScores[col]]
#     plt.subplot(4, len(colScores)//4, idx+1)
#     plt.title(col)
#     plt.ylabel("Silhouette Score")
#     plt.xlabel("Num clusters")
#     plt.plot(list(range(2,2+len(scores))), scores)
    
# fig.suptitle(f"Silhouette scores for clusters 2 to {1+len(scores)}", fontsize=30)
# plt.tight_layout()
# plt.show()

# %%
clusteredDF.describe()

# %% [markdown]
# ### Get an estimation of the true silhouette score with a sample of the entire dataset

# %%
def silhouetteScoreCalc(data, y_pred, test_size=0.1):

    sample_idx = np.random.choice(data.shape[0], int(test_size * len(data)), replace=False)
    test_sample_x = data[sample_idx]
    test_sample_y = [y_pred[i] for i in sample_idx]
    
    patience = 0

    while len(np.unique(test_sample_y)) < 2:
        patience += 1
        if patience > 3:
            return 0
        print("Recalculating sample due to too few clusters")
        sample_idx = np.random.choice(data.shape[0], int(test_size * len(data)), replace=False)
        test_sample_x = data[sample_idx]
        test_sample_y = [y_pred[i] for i in sample_idx]

    score = silhouette_score(test_sample_x, test_sample_y, metric='dtw')
    print("Calculating sil score...")

    score = silhouette_score(test_sample_x, test_sample_y, metric='dtw', n_jobs=4)

    return score


# %%


scores = {}

# Using sampling for the silhouette score since calculating the score on the entire dataset takes a long time
# The time series in this dataset are long so a smaller test size helps.
for column in tqdm(clusteredDF.columns):
    y_pred = list(clusteredDF[column])
    dataFormatted = formatForTimeSeries(column)
    
    score = silhouetteScoreCalc(dataFormatted, y_pred, test_size=0.001)


    scores[column] = score



# %% [markdown]
# ### Remove outliers from the clustering graphs

# %%
def removeOutliers(data, threshold):
    stdDev = np.nanstd(data)
    mean = np.nanmean(data)
    normalised = [np.nanmean(np.abs(d - mean)) for d in data]
    mask = normalised < threshold * stdDev
    return data[mask], data[np.logical_not(mask)]

# %% [markdown]
# ### Creating the graphs takes a long time to run, commented out for speed

# %%
colours = {0:'r', 1:'g', 2:'b', 3:'c', 4:'m', 5:'y', 6:'k', 7:'w', 8:'orange', 9:'purple', 10:'pink'}

clusterMetricsList = []

for col in clusteringPatients[0].interpolatedData.columns:

    clusters = 2

    # fig = plt.figure(figsize=(clusters*3,2.5), dpi=200)

    # fig.suptitle(f"{col}, Sil score: {np.round(scores[col], 2)}", fontsize=20)

    colData = [j.interpolatedData[col].values for j in clusteringPatients]

    minVal, maxVal = np.nanmin([np.nanmin(j) for j in colData]), np.nanmax([np.nanmax(j) for j in colData])

    formattedData = formatForTimeSeries(col)

    for i in range(clusters):
        # plt.subplot(1, clusters, i+1)

        y_pred = clusteredDF[col]

        # dataCluster = np.array(colData)[y_pred == i]

        dataCluster = formattedData[y_pred == i]

        withoutOutliers, outliers = removeOutliers(dataCluster, 1.5)
        
        # print(f"Num removed: {len(dataCluster) - len(withoutOutliers)}")

        # print(len(dataCluster))
        # print(len(withoutOutliers))
        
        # for sample in outliers:
        #     plt.plot(sample, c='black', alpha=0.05, linewidth=1)
        
        # for sample in withoutOutliers:
        #     plt.plot(sample, c=colours[i], alpha=0.1, linewidth=1)

        stdDev = np.nanstd(withoutOutliers)
        mean = np.nanmean(withoutOutliers)

        clusterMetricsList.append([col, stdDev, mean])

        # plt.title(f"C {i+1}, std: {np.round(stdDev, 2)}, mean: {np.round(mean, 2)}")
        # plt.xlabel("Time")
        # plt.ylabel("Value")
        # # print(dataCluster)
        # # print(np.nanstd(dataCluster))
        # plt.ylim(minVal, maxVal)

        
    # plt.tight_layout()
    # plt.savefig(f"./figures/sepsis/{col}.png")
    # plt.show()
    


clusterMetricsDF = pd.DataFrame(data = clusterMetricsList, columns=['Feature', 'StdDev', 'Mean'])
display(clusterMetricsDF)


# %% [markdown]
# ### Helper code to combine the above graphs into one image for the dissertation

# %%
# figdir = "./figures/sepsis/"

# images = [Image.open(figdir + x) for x in list(next(os.walk(figdir))[2:])[0]]

# widths, heights = zip(*(i.size for i in images))

# widthMax = max(widths)
# widthMin = min(widths)
# heightTotal = sum(heights)

# combined = Image.new('RGBA', (widthMax, heightTotal))

# offset = 0
# for im in images:
#   xOffset = 0
#   if im.size[0] != widthMax:
#     xOffset = (widthMax - im.size[0]) // 2
#   combined.paste(im, (xOffset, offset))
#   offset += im.size[1]


# fig = plt.figure(figsize=(widthMax/100, heightTotal/100), dpi=100)
# plt.title("Result of Sepsis DTW clustering", fontsize=50)
# plt.axis('off')
# # plt.tight_layout()
# plt.imshow(combined)
# plt.show()


# # new_im.save('test.jpg')

# %% [markdown]
# ### Order by std dev to find the clusters that vary the most, order by mean to find the highest/lowest values.

# %%

display(clusteredDF.head())

def getMapping(metric, subset):
    
    ordered = subset.reset_index().sort_values(by=metric, ascending=True)

    before = ordered.index
    after = ordered.reset_index().index

    mapping = {before[i]: after[i] for i in range(len(before))}

    return mapping



orderedDF = pd.DataFrame()

for name, subset in clusterMetricsDF.groupby('Feature'):

    # clusterData = [np.pad(j.interpolatedData[col].values, (0, 48 - len(j.interpolatedData[col].values)), 'constant', constant_values = (np.NaN, np.NaN)) for j in clusteringPatients]

    for metric in list(clusterMetricsDF.columns)[1:]:

        mapping = getMapping(metric, subset)

        newCol = str(name + "_" + metric)

        orderedDF[newCol] = clusteredDF[name].map(mapping)


# orderedDF = orderedDF.set_index(clusteredDF.index)


display(orderedDF.head())


# %%
staticVals = [p.static.max().values for p in clusteringPatients]


staticDF = pd.DataFrame(data = staticVals, columns=staticColumns)

ids = [p.patientID for p in clusteringPatients]

staticDF['PatientID'] = ids

staticDF = staticDF.set_index("PatientID")

staticDF = staticDF.apply(lambda x: x.fillna(x.mean()))

staticDF


# %%
cat = Categorization.Categorizer(staticDF)

binnedDF = cat.kBins(2, 'uniform')

boundaries = cat.getBoundaries()

display(boundaries)


binnedDF['PatientID'] = ids

binnedDF = binnedDF.set_index("PatientID")

binnedDF = binnedDF.astype(np.int64)


binnedDF

# %%
orderedDF[staticColumns] = binnedDF[staticColumns]

orderedDF

# %%
cat = Categorization.Categorizer()

mapping = {0: 'very_low', 1: 'low', 2: 'medium', 3: 'high', 4: 'very_high'}

mapped = cat.map_types(data = {"ordered":orderedDF}, mapping=mapping)['ordered']

display(mapped)

# %%
targetSeries = [patient.label for patient in clusteringPatients]

# targetSeries

# %%
mapped['Mortality14Days'] = targetSeries

display(mapped)

mapped.to_csv("./categorisedData/clusteredDataSepsis.csv")

# %%
mapped['Mortality14Days'].value_counts()


