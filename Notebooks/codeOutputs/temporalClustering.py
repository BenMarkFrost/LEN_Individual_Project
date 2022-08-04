# %% [markdown]
# # Mortality Clustering
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
import random


# %% [markdown]
# ### Loading in the MIMIC dataset
# 

# %%
th = TH()

mimicDF = th.get_mimic()

mimicDF

# %% [markdown]
# ### Exploring values

# %%

mimicDF[mimicDF['Arterial pH'] > 8]

# %%
mimicDF[~mimicDF['Admit Ht'].isnull()]

# %% [markdown]
# ### Describing all columns

# %%
# Too many columns to display all in one cell.

step = 10

for idx in range(0, len(mimicDF.columns), step):
    tempCols = mimicDF.columns[idx:idx+step]
    display(mimicDF[tempCols].describe())

# %%
# Fixing 'arterial pH', 'ionized calcium' since they contain erroneous values.

mimicDF['Arterial pH'][mimicDF['Arterial pH'] > 8] = mimicDF['Arterial pH'].mean()
mimicDF['Ionized Calcium'][mimicDF['Ionized Calcium'] > 8] = mimicDF['Ionized Calcium'].mean()

# %%
print(f"There are {mimicDF['PatientID'].nunique()} unique patients in the dataset")

# %%
patients = th.get_patients(mimicDF)

print(len(patients))

# %% [markdown]
# ### Counting total null columns across all patients

# %%
totalNullColumns = 0

for patient in patients:
    totalNullColumns += patient.data.isnull().all().sum()

totalColumns = len(patient.data.columns) * len(patients)

print(totalColumns, totalNullColumns)

print(f"{np.round(totalNullColumns / totalColumns * 100, 2)}% of columns are null")

# %% [markdown]
# ### Counting how many columns have no data in 

# %%
patientsKept, columnsExplored = th.count_null(patients)

# %% [markdown]
# #### Sharp drop off after 12 columns so will keep around 1000 patients with at least some data in the top 12 columns

# %%
clusteringPatients = th.get_top_columns(patients, 12)

# clusteringPatients = patients

# for patient in clusteringPatients:
#     patient.topColumns = patient.data

print(len(clusteringPatients))

clusteringPatients[0].topColumns

# %%
allDataTemp = pd.concat([patient.data for patient in clusteringPatients])
allDataTemp = allDataTemp.describe().T['mean'].T
allDataTemp['ALT']

# %%
clusteringPatients[0].topColumns.count()

# %% [markdown]
# ### Interpolating missing data

# %%

noInterpolation = 0
failureExample = (0,0)



for idx, patient in tqdm(enumerate(clusteringPatients)):


    patient.interpolatedData = copy.copy(patient.topColumns)

    patientNonNullCount = patient.topColumns.count()

    for column in patient.topColumns.columns:

        # If the column has no data in then fill with the mean from the other patients
        if patient.topColumns[column].isnull().all():
            print("replacing ", column)
            print(allDataTemp[column])
            patient.interpolatedData[column] = allDataTemp[column]
            continue

        try:
            # Try interpolating with a polynomial model
            patient.interpolatedData[column] = patient.topColumns[column].interpolate(method='polynomial', order=2, limit_direction='both', limit_area='inside')

    
        except ValueError:

            try:

                if patientNonNullCount[column] == 1:
                    # If the column has only one data point in, pad with three values either side.
                    patient.interpolatedData[column] = patient.topColumns[column].interpolate(method='linear', limit_direction='both', limit=3)
                elif patientNonNullCount[column] == 0:
                    print("no data in ", column)
                    break
                else: 
                    # Otherwise, interpolate with a linear model
                    patient.interpolatedData[column] = patient.topColumns[column].interpolate(method='linear', limit_direction='both', limit_area='inside')
            
            except ValueError:
                
                # If the interpolation fails, then fill with the mean from the other patients
                patient.interpolatedData[column] = patient.topColumns[column].fillna(allDataTemp[column])
                noInterpolation += 1
                failureExample = (idx, column)



    if patient.interpolatedData.shape[0] != 48:
        # If hte column length is shorter than 48, pad with nan values.
        fixedData = []
        for col in patient.interpolatedData:
            fixedData.append(np.pad(patient.interpolatedData[col], (0, 48 - patient.interpolatedData[col].shape[0]), 'constant', constant_values=np.nan))

        tempDF = pd.DataFrame(data = fixedData).T
        tempDF.columns = patient.interpolatedData.columns
        patient.interpolatedData = tempDF


print(f"{noInterpolation}/{len(clusteringPatients)} patients failed to interpolate all columns")
print(f"{failureExample}")

# %%
for patient in clusteringPatients[:5]:
    display(patient.interpolatedData.head())

# %% [markdown]
# ### Visualising the data before interpolation

# %%
patient = clusteringPatients[0]

fig = plt.figure(figsize = (30, 8), dpi=200)


fig.suptitle(f"Patient: {patient.label}", fontsize=30)

for idx, col in enumerate(patient.topColumns.columns):
    plt.subplot(2, len(patient.topColumns.columns)/2, idx+1)

    plt.scatter(patient.topColumns.index, patient.topColumns[col], c='Orange')
    plt.title(f"{col}", fontsize=20)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Visualising the data after interpolation

# %%

for patient in clusteringPatients[:1]:


    fig = plt.figure(figsize = (30, 8),dpi=200)


    fig.suptitle(f"Patient: {patient.label}", fontsize=30)

    for idx, col in enumerate(patient.interpolatedData.columns):
        plt.subplot(2, len(patient.interpolatedData.columns)//2, idx+1)

        plt.plot(patient.interpolatedData.index, patient.interpolatedData[col])
        plt.scatter(patient.interpolatedData.index, patient.interpolatedData[col])
        plt.scatter(patient.topColumns.index, patient.topColumns[col], c="Orange")
        plt.title(f"{col}", fontsize=20)

    plt.tight_layout()
    plt.show()

# %%
# Format the given column for clustering
def formatForTimeSeries(column, sampleSize=None):

    if sampleSize is None:
        sampleList = clusteringPatients
    else:
        minorityClass = [patient for patient in clusteringPatients if patient.label == 1]
        majorityClass = [patient for patient in clusteringPatients if patient.label == 0]
        
        minorityList = random.choices(minorityClass, k=sampleSize//2)
        majorityList = random.choices(majorityClass, k=sampleSize//2)

        sampleList = minorityList + majorityList

        print(minorityList)

        # sampleList = random.shuffle(sampleList)

        

    print("Creating stacked DF...")
    stackedDF = pd.DataFrame([patient.interpolatedData[column].values for patient in sampleList])


    stackedNumpy = stackedDF.to_numpy()

    cleanedNumpy = []

    print("Cleaning")
    for row in stackedNumpy:
        cRow = row[~np.isnan(row)]
        if len(cRow) > 0:
            cleanedNumpy.append(cRow)

            
    dataFormatted = to_time_series_dataset([*cleanedNumpy])

    return dataFormatted


# Cluster the given column
def timeSeriesCluster(clusters, dataFormatted):

    print("Clustering")

    model = TimeSeriesKMeans(n_clusters=clusters, tol=1e-1, metric="dtw", max_iter=1, random_state=0, n_jobs=4)
    model.fit(dataFormatted)


    return model


# %% [markdown]
# ### Establishing the caching function

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
    

# %% [markdown]
# ### Clustering all columns

# %%
myHash = "Chosen_clusters"


clusteredDF, myHash = find_cached(clusteringPatients, hash=myHash)


#Originally each column was a separate cluster, but this was changed to 2 for all since it gave best performance.
clusterNums = {"Platelets": 3, "Arterial BP [Diastolic]": 2, "Arterial BP [Systolic]": 2, "Arterial BP Mean": 2, "CVP": 2, "Arterial pH": 2, "Hemoglobin": 2, "Arterial PaCO2": 2, "Arterial PaO2": 2, "SVR": 2, "Ionized Calcium": 2, "SVRI": 2}

# Caching disabled
if clusteredDF is False:

    clusteredDF = pd.DataFrame()

    for column in tqdm(clusteringPatients[0].interpolatedData.columns):
        dataFormatted = formatForTimeSeries(column)
        model = timeSeriesCluster(clusterNums[column], dataFormatted)

        print("Finished fitting. Predicting... ")


        print(column)
        dataFormattedAll = formatForTimeSeries(column)
        y_pred = model.predict(dataFormattedAll)
        clusteredDF[column] = y_pred

    ids = [patient.label for patient in clusteringPatients]

    clusteredDF["PatientID"] = ids

    clusteredDF = clusteredDF.set_index("PatientID")


    clusteredDF.to_csv("./processingCache/" + myHash + ".csv")


# %%
clusteringPatients[2].interpolatedData


# %%
clusteredDF.describe()

# %% [markdown]
# ### Custom random sampling silhouette function

# %%
def silhouetteScoreCalc(data, y_pred):

    test_size = 0.1
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

    return score


# %%


scores = {}

# Using sampling for the silhouette score since calculating the score on the entire dataset takes a long time
for column in tqdm(clusteredDF.columns):
    y_pred = list(clusteredDF[column])
    dataFormatted = formatForTimeSeries(column)
    
    score = silhouetteScoreCalc(dataFormatted, y_pred)
    scores[column] = score

print(scores)


# %% [markdown]
# ### Removing outliers from the clustering

# %%
def removeOutliers(data, threshold):
    stdDev = np.nanstd(data)
    
    mean = np.nanmean(data)
    
    normalised = [np.nanmean(np.abs(d - mean)) for d in data]
    mask = normalised < threshold * stdDev
    return data[mask], data[np.logical_not(mask)]

# %% [markdown]
# ### Plotting the results of clustering

# %%
colours = {0:'r', 1:'g', 2:'b', 3:'c', 4:'m', 5:'y', 6:'k', 7:'w', 8:'orange', 9:'purple', 10:'pink'}

clusterMetricsList = []

for col in clusteringPatients[0].interpolatedData.columns:

    clusters = 2

    fig = plt.figure(figsize=(clusters*3,2.5), dpi=200)

    fig.suptitle(f"{col}, Sil score: {np.round(scores[col], 2)}", fontsize=20)
    

    colData = [np.pad(j.interpolatedData[col].values, (0, 48 - len(j.interpolatedData[col].values)), 'constant', constant_values = (np.NaN, np.NaN)) for j in clusteringPatients]

    minVal, maxVal = np.nanmin([np.nanmin(j) for j in colData]), np.nanmax([np.nanmax(j) for j in colData])

    for i in range(clusters):
        plt.subplot(1, clusters, i+1)

        y_pred = clusteredDF[col]

        dataCluster = formatForTimeSeries(col)[y_pred == i]

        withoutOutliers, outliers = removeOutliers(dataCluster, 1.5)
        
        for sample in outliers:
            plt.plot(sample, c='black', alpha=0.05, linewidth=1)
        
        for sample in withoutOutliers:
            plt.plot(sample, c=colours[i], alpha=0.1, linewidth=1)

        stdDev = np.nanstd(withoutOutliers)
        mean = np.nanmean(withoutOutliers)

        clusterMetricsList.append([col, stdDev, mean])

        plt.title(f"C {i+1}, std: {np.round(stdDev, 2)}, mean: {np.round(mean, 2)}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.ylim(minVal, maxVal)

        
    plt.tight_layout()
    plt.savefig(f"./figures/{col}.png")
    plt.show()
    


clusterMetricsDF = pd.DataFrame(data = clusterMetricsList, columns=['Feature', 'StdDev', 'Mean'])
display(clusterMetricsDF)


# %% [markdown]
# ### Helper code to combine the clustering graphs into one big graph for the report

# %%
# figdir = "./figures/"

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
# plt.title("Result of DTW clustering", fontsize=50)
# plt.axis('off')
# # plt.tight_layout()
# plt.imshow(combined)
# plt.show()


# new_im.save('test.jpg')

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

    for metric in list(clusterMetricsDF.columns)[1:]:

        mapping = getMapping(metric, subset)

        newCol = str(name + "_" + metric)

        orderedDF[newCol] = clusteredDF[name].map(mapping)



display(orderedDF.head())


# %% [markdown]
# ### Mapping low to high

# %%

cat = Categorization.Categorizer()

mapping = {0: 'very_low', 1: 'low', 2: 'medium', 3: 'high', 4: 'very_high'}

mapped = cat.map_types(data = {"ordered":orderedDF}, mapping={0: 'very_low', 1: 'low', 2: 'medium', 3: 'high', 4: 'very_high'})['ordered']

display(mapped)

# %%
targetSeries = [patient.label for patient in clusteringPatients]

# %%
pd.Series(targetSeries).value_counts()

# %%
mapped['Mortality14Days'] = targetSeries

display(mapped)

mapped.to_csv("./categorisedData/clusteredData.csv")

# %% [markdown]
# ### Exploring the differences in each class' features

# %%
minority = mapped[mapped['Mortality14Days'] == 1]

majority = mapped[mapped['Mortality14Days'] == 0]

display(mapped.describe())
display(minority.describe())
display(majority.describe())


