import pandas as pd
from functools import lru_cache
import copy
from tqdm import tqdm
from matplotlib import pyplot as plt

class Patient:

    """ This class was created to store data relating to an individual patient."""

    def __init__(self, patientID, label, data, static=None):
        self.patientID = patientID
        self.data = data
        self.interpolatedData = pd.DataFrame()
        self.label = label
        if static is not None:
            self.static=static


    def __repr__(self):
        return f"PatientID: {self.patientID}\n Readings: {self.data.to_string()}"




class TemporalHelper:

    """ This class contains methods for loading and preprocessing the MIMIC data. """

    def __init__(self):
        pass

    @lru_cache()
    def get_mimic(self):

        """ Loads in the MIMIC time series data"""

        self.DF = pd.read_csv("../LEN_Test/data/TimeSeries.csv")

        return self.DF

    def get_patients(self, dataFrame=None, by="PatientID", label="Mortality14Days", static=None):
        
        """ Formats the mimic data so that it is separated into patients. """

        patients = []

        if dataFrame is None:
            dataFrame = self.get_mimic()

        for id, df in tqdm(dataFrame.groupby(by)):
            if static is not None:
                staticDF = df[static]
                df = df.drop(static, axis=1)
            else:
                staticDF = None
            patientDF = df.reset_index().drop(columns=[by, 'index', label])
            mortality = df[label].max()
            patient = Patient(id, mortality, patientDF, staticDF)
            patients.append(patient)

        self.patients = patients

        return patients

    def get_null_count(self, patients=None):

        """ Returns a dictionary of the number of null values in each column. """

        if patients is None:
            patients = self.patients

        nullCount = {}

        print("[1/4] Counting null values...")

        for patient in tqdm(patients):

            nullCols = patient.data.isnull().all()

            for column in patient.data.columns:

                if nullCols[column]:

                    if column not in nullCount:
                        nullCount[column] = 1
                    else:
                        nullCount[column] += 1

        nullCount = dict(sorted(nullCount.items(), key=lambda item: item[1]))

        return nullCount


    def count_null(self, patients=None):

        """ Creates a graph of missingness and a list of patientIDs for each n-populated columns"""

        if patients is None:
            patients = self.patients


        patientsKept = []
        numPatientsKept = []

        nullCount = self.get_null_count(patients)


        columnsExplored = list(nullCount.keys())

        print("[2/3] Copying dataset...")

        tempPatients = [copy.copy(patient) for patient in patients]

        print("[3/4] Dropping null columns...")

        for i in tqdm(range(len(columnsExplored))):

            col = columnsExplored[i]

            nonNullPatients = []

            toRemove = []

            for idx in range(len(tempPatients)):


                if tempPatients[idx].data[col].isnull().all():
                    toRemove.append(idx)
                else:
                    nonNullPatients.append(tempPatients[idx].patientID)

            toRemove = sorted(toRemove, reverse=True)


            for idx in toRemove:
                tempPatients.pop(idx)


            patientsKept.append(nonNullPatients)
            numPatientsKept.append(len(nonNullPatients))


        print("[4/4] Graphing...")

        patientsKeptDF = pd.DataFrame(data=numPatientsKept)

        patientsKeptDF['n_col'] =list(range(1, len(patientsKeptDF)+1))

        patientsKeptDF = patientsKeptDF.set_index('n_col')

        fig = plt.figure(figsize=(10, 6), dpi=100)

        plt.xticks(patientsKeptDF.iloc[:,0].index)
        plt.xlabel("n-columns to keep")
        plt.ylabel("Patients with non-null values in all n-columns")
        plt.title("Patients with non-null values in all n-most populated columns")
        plt.plot(patientsKeptDF.iloc[:,0])
        plt.tight_layout()
        plt.show()

        self.patientsKept = patientsKept
        self.columnsExplored = columnsExplored

        return patientsKept, columnsExplored


    def get_top_columns(self, patients=None, columns=12):

        """ Returns a list of patients with data in the n most populated columns. """

        if patients is None:
            patients = self.patients

        clusteringPatients = []
        ids = set(self.patientsKept[columns-1])

        for patient in tqdm(patients):
            if patient.patientID in ids:
                patient.topColumns = patient.data[self.columnsExplored[:columns]]
                clusteringPatients.append(patient)

        return clusteringPatients