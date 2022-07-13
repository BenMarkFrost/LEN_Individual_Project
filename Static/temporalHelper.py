import pandas as pd
from functools import lru_cache
import copy
from tqdm import tqdm
from matplotlib import pyplot as plt

class Patient:

    def __init__(self, patientID, mortality, data):
        self.patientID = patientID
        self.data = data
        self.interpolatedData = pd.DataFrame()
        self.mortality = mortality


    def __repr__(self):
        return f"PatientID: {self.patientID}\n Readings: {self.data.to_string()}"




class TemporalHelper:

    def __init__(self):
        pass

    @lru_cache()
    def get_mimic(self):

        self.mimicDF = pd.read_csv("../LEN_Test/data/TimeSeries.csv")

        return self.mimicDF

    def get_patients(self, dataFrame=None):

        patients = []

        if dataFrame is None:
            dataFrame = self.get_mimic()

        for id in dataFrame['PatientID'].unique():
            patientDF = dataFrame[dataFrame['PatientID'] == id].reset_index().drop(columns=['PatientID', 'index', 'Mortality14Days'])
            mortality = dataFrame[dataFrame['PatientID'] == id]['Mortality14Days'].max()
            patient = Patient(id, mortality, patientDF)
            patients.append(patient)

        self.patients = patients

        return patients

    def get_null_count(self, patients=None):

        if patients is None:
            patients = self.patients

        nullCount = {}

        for patient in patients:

            for column in patient.data.columns:

                if patient.data[column].isnull().all():

                    if column not in nullCount:
                        nullCount[column] = int(patient.data[column].isnull().all())
                    else:
                        nullCount[column] += int(patient.data[column].isnull().all())

        nullCount = dict(sorted(nullCount.items(), key=lambda item: item[1]))

        return nullCount


    def count_null(self, patients=None):

        if patients is None:
            patients = self.patients


        patientsKept = []
        numPatientsKept = []

        nullCount = self.get_null_count(patients)


        columnsExplored = list(nullCount.keys())

        tempPatients = [copy.copy(patient) for patient in patients]

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


        patientsKeptDF = pd.DataFrame(data=numPatientsKept)

        patientsKeptDF['n_col'] =list(range(1, len(patientsKeptDF)+1))

        patientsKeptDF = patientsKeptDF.set_index('n_col')

        fig = plt.figure(figsize=(10, 6))

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

        if patients is None:
            patients = self.patients

        clusteringPatients = []
        ids = set(self.patientsKept[columns-1])

        for patient in patients:
            if patient.patientID in ids:
                patient.topColumns = patient.data[self.columnsExplored[:columns]]
                clusteringPatients.append(patient)

        return clusteringPatients