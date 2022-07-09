import pandas as pd
from functools import lru_cache

class Patient:

    def __init__(self, patientID, data):
        self.patientID = patientID
        self.data = data
        self.interpolatedData = pd.DataFrame()


    def __repr__(self):
        return f"PatientID: {self.patientID}\n Readings: {self.data}"


class TemporalHelper:

    def __init__(self):
        pass

    @lru_cache()
    def get_mimic(self):

        self.mimicDF = pd.read_csv("../LEN_Test/data/TimeSeries.csv")

        return self.mimicDF

    def get_patients(self):

        patients = []

        for id in self.mimicDF['PatientID'].unique():
            patientDF = self.mimicDF[self.mimicDF['PatientID'] == id].reset_index().drop(columns=['PatientID', 'index'])
            patient = Patient(id, patientDF)
            patients.append(patient)

        return patients

        
