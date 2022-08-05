from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy


class Categorizer:

    """ This class provides helper methods to be able to more easily categorize the data. """
    def __init__(self, data={}):
        self.data = data
        self.categorizationTypes = {}
        self.mappedTypes = {}


    # Maps the ordered values to labels.
    def map_types(self, data = None, mapping = {0: 'very_low', 1: 'low', 2: 'medium', 3: 'high', 4: 'very_high'}):

        if data is None:
            data = self.categorizationTypes
            
        for type in data:

            df = copy.copy(data[type])

            for col in df.columns:
                
                # Updating the mapping dictionary if there are fewer than 5 categories.
                newMapping = mapping
                
                if len(np.unique(df[col])) == 2:

                    newMapping = {0: 'low', 1: 'high'}

                elif len(np.unique(df[col])) < len(list(mapping.keys())):
                   
                    newMapping = dict(list(mapping.items())[len(np.unique(df[col]))//2:(len(np.unique(df[col])) + len(np.unique(df[col])) // 2)])

                    fixedMapping = {}

                    count = 0
                    for i in newMapping:
                        fixedMapping[count] = newMapping[i]
                        count += 1

                    newMapping = fixedMapping

                df[col] = df[col].map(newMapping)

            self.mappedTypes[type] = pd.get_dummies(df)

        return self.mappedTypes


    def reorder(self, data, bins):

        # This reorders the categories to the ascending order of the values in the original DF. Took bloody ages to figure this out.

        for col in data:
            # Getting the REVERSE index of the ordered values in the original DF. Double index thing took ages.
            tempDF = pd.DataFrame()
            # Sort by the values
            tempDF['idx'] = list(self.data[col].sort_values().index)
            # Then sort by the result's index
            tempDF = tempDF.sort_values(by = 'idx')

            # Using the created index to reorder the categories.
            tempDF2 = data.copy()
            tempDF2['idx'] = tempDF.index
            tempDF2 = tempDF2.set_index('idx').sort_index()
            # Mapping from previous categories to the new ordered categories.
            mapping = dict(zip(tempDF2[col].unique(), list(range(bins))))
            data[col] = data[col].map(mapping)

        return data


    # K-Bins discretization interface.
    def kBins(self, bins, strategy='uniform'):

        est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=strategy)
        kBinsDF = pd.DataFrame(data=est.fit_transform(self.data), columns=self.data.columns)

        self.categorizationTypes['kBins'] = kBinsDF

        return kBinsDF

    # K-Means clustering interface.
    def kMeans(self, n_clusters):


        kMeansDF = self.data.apply(lambda x: KMeans(n_clusters=n_clusters, random_state=0).fit_predict(np.asarray(x).reshape(-1,1)))

        reorderedKMeansDF = self.reorder(kMeansDF, bins = n_clusters)

        self.categorizationTypes['kMeans'] = reorderedKMeansDF

        return kMeansDF
    
    # Agglomerative clustering interface.
    def agglomerative(self, n_clusters):

        agglomerativeDF = self.data.apply(lambda x: AgglomerativeClustering(n_clusters=n_clusters).fit_predict(np.asarray(x).reshape(-1,1)))

        reorderedAgglomerativeDF = self.reorder(agglomerativeDF, bins = n_clusters)

        self.categorizationTypes['agglomerative'] = reorderedAgglomerativeDF

        return agglomerativeDF
    
    # Plotting each column's categorization.
    def plotData(self, col, df):
        
        score = np.round(silhouette_score(np.asarray(self.data[col]).reshape(-1,1), df[col]), 2)

        tempData = copy.copy(self.data[[col]])

        tempData['cluster'] = df[col].values
        tempData['target'] = df['target'].values

        for _, data in tempData.groupby("target"):

            plt.scatter(data[col], data[col].map(self.data[col].value_counts()), c=data['cluster'], marker=data['target'].map({0: 'o', 1: 'x'}).iloc[0])
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.title(f"{col}, score: {score}")


    # Controlling the display of each column's categorization.
    def display(self, num=None, target=None):

        if target is None:
            target = pd.Series(data=np.zeros((len(self.data))))

        for type in self.categorizationTypes:

            df = copy.copy(self.categorizationTypes[type])

            df['target'] = target

            max = num if num is not None else len(df.columns)

            if max < 6:

                fig = plt.figure(figsize=(10,6), dpi=100)

                fig.suptitle(f"{type}", fontsize=15)


                for idx, col in enumerate(list(self.data.columns)[:max]):

                    plt.subplot(int(len(self.data.columns)/2), int(len(self.data.columns)/2), idx+1)
                    self.plotData(col, df)
            
                plt.tight_layout()
                plt.show()

            else:

                for idx, col in enumerate(list(self.data.columns)[:max]):

                    fig = plt.figure()

                    fig.suptitle(f"{type}")

                    self.plotData(col, df)

                    plt.tight_layout()
                    plt.show()



    # Finds the boundaries between each category.
    def getBoundaries(self):

        boundaries = {}

        for type in self.categorizationTypes:

            df = self.categorizationTypes[type]

            boundaries[type] = {}

            for col in df.columns:

                tempDF = pd.DataFrame(data=[df[col].values, self.data[col].values]).T

                tempDF = tempDF.sort_values(by=1)

                tempDF = tempDF.reset_index()

                boundaries[type][col] = []

                for i in list(tempDF[0].unique())[1:]:
                    boundaries[type][col].append(tempDF[1].iloc[tempDF[0][tempDF[0] == i].index[0]])

        return boundaries


            