# %% [markdown]
# # Model Training and Testing
# 
# Benjamin Frost 2022
# 

# %%
import os
import pandas as pd
import numpy as np
import time
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data.sampler import WeightedRandomSampler
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer, seed_everything
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
import seaborn as sns
import os
import sys
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from torch_explain.models.explainer import Explainer
from torch_explain.logic.metrics import formula_consistency
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTEN
from imblearn.combine import SMOTEENN
from torch.nn.functional import one_hot
from func_timeout import func_set_timeout, func_timeout, FunctionTimedOut
import datetime
import time

seed_everything(42)
base_dir = f'./runs'

# %% [markdown]
# ### Loading in the datasets

# %%
files = os.listdir("./categorisedData/")

datasets = {file : pd.read_csv("./categorisedData/" + file) for file in files}

print(files)

results_dict = {}

# %% [markdown]
# ### Defining the timeout wrapper around the explainer API

# %%
@func_set_timeout(90)
def explain_with_timeout(model, val_data, train_data, test_data, topk_expl, concepts):

    return model.explain_class(val_dataloaders=val_data, train_dataloaders=train_data, test_dataloaders=test_data, topk_explanations=topk_expl, concept_names=concepts, max_minterm_complexity=5)

# %%
# Nodes in each hidden layer, learning rate

hiddenLayers = {
    'breastCancer.csv' : [[20], 0.01],
    'clusteredData.csv' : [[20], 0.01], 
    'clusteredDataSepsis.csv' : [[20, 40, 20], 0.0001],
    'expertLabelledData.csv' : [[20], 0.01],
    'metricExtractedData.csv' : [[20, 20], 0.01],
    'staticData.csv': [[20], 0.01]
}

# %% [markdown]
# ### K-Fold Validation

# %%
for file in files:

    if file in hiddenLayers:
        layers = hiddenLayers[file]
    else:
        print("Set layers for " + file)
        layers = [[20], 0.01]

    print(f"Training {file}\n")

    data = datasets[file]

    if "PatientID" in data.columns:
        data = data.drop(columns=["PatientID"])


    targetName = "Mortality14Days"

    targetSeries = data[targetName]
    print(data[targetName].value_counts())
    data = data.drop(columns=[targetName])

    n_concepts = data.shape[1]
    print("There are " + str(n_concepts) + " concepts")
    n_classes = 2   

    splitResults_list = []

    """ The following lines were taken from the MIMIC example code by Pietro Barbiero"""
    
    dataTensor = torch.FloatTensor(data.to_numpy())
    targetTensor = one_hot(torch.tensor(targetSeries.values).to(torch.long)).to(torch.float)   

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    x = dataTensor
    y = targetTensor

    for split, (trainval_index, test_index) in enumerate(skf.split(x.cpu().detach().numpy(),
                                                                y.argmax(dim=1).cpu().detach().numpy())):
        print(f'Split [{split + 1}/{n_splits}]')


        x_trainval, x_test = torch.FloatTensor(x[trainval_index]), torch.FloatTensor(x[test_index])
        y_trainval, y_test = torch.FloatTensor(y[trainval_index]), torch.FloatTensor(y[test_index])
        x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.2, random_state=42)
        print(f'{len(y_train)}/{len(y_val)}/{len(y_test)}')


        """ End of reference code """

        print(pd.Series(np.argmax(y_train.numpy(), axis=1)).value_counts().values)

        # For oversampling... 
        clf = SMOTEN(random_state=0)

        x_train, y_train = clf.fit_resample(x_train.numpy(), np.argmax(y_train.numpy(), axis=1))

        x_train = torch.FloatTensor(x_train)
        y_train = one_hot(torch.tensor(y_train).to(torch.long)).to(torch.float)

        print(pd.Series(np.argmax(y_train.numpy(), axis=1)).value_counts().values)

        batch_size = 64

        train_data = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True)

        # For random sampling...
        # class_count = pd.Series(targetSeries).value_counts()
        # print(class_count)
        # weights = 1. / torch.FloatTensor(class_count.values)
        # print(weights)
        # train_weights = np.array([weights[t] for t in torch.argmax(y_train, axis=1).numpy()]).astype(np.float64)
        # sampler = WeightedRandomSampler(train_weights, train_size)
        # train_data = TensorDataset(x_train, y_train)
        # train_loader = DataLoader(train_data, batch_size=train_size, sampler=sampler)

        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20, verbose=True, mode='min')

        logger = TensorBoardLogger("./runs/splits/", name=file)

        """ The following lines were taken from the MIMIC example code by Pietro Barbiero"""

        val_data = TensorDataset(x_val, y_val)
        test_data = TensorDataset(x_test, y_test)
        val_loader = DataLoader(val_data, batch_size = len(x_val))
        test_loader = DataLoader(test_data, batch_size = len(x_test))

        checkpoint_callback = ModelCheckpoint(dirpath=base_dir, monitor='val_loss', mode='min', save_top_k=1)
        
        trainer = Trainer(max_epochs=100, gpus=1, auto_lr_find=True, deterministic=True,
                        check_val_every_n_epoch=1, default_root_dir=base_dir,
                        weights_save_path=base_dir, callbacks=[checkpoint_callback, early_stopping_callback],
                        logger=logger, enable_progress_bar=False, gradient_clip_val=0.5)

        model = Explainer(n_concepts=n_concepts, n_classes=n_classes, l1=1e-3, lr=layers[1],
                        explainer_hidden=layers[0], temperature=0.7)

        # Training the model
        trainer.fit(model, train_loader, val_loader)
        model.freeze()

        """ End of reference code """

        # Precision, Recall, F1
        y_pred = torch.argmax(model(x_test), axis=1)
        y_test_argmax = torch.argmax(y_test, axis=1)

        scores = [f1_score(y_test_argmax.numpy(), y_pred.numpy(), average='macro'), 
                recall_score(y_test_argmax.numpy(), y_pred.numpy(), average='macro'), 
                precision_score(y_test_argmax.numpy(), y_pred.numpy(), average='macro')]

        print(f"Before loading best: {scores}")

        # Loading in the best weights from training
        model = model.load_from_checkpoint(checkpoint_callback.best_model_path)

        # Precision, Recall, F1
        scores = [f1_score(y_test_argmax.numpy(), y_pred.numpy(), average='macro'), 
                recall_score(y_test_argmax.numpy(), y_pred.numpy(), average='macro'), 
                precision_score(y_test_argmax.numpy(), y_pred.numpy(), average='macro')]

        print(f"{file} split {split+1} scores: {scores}")

        print("\nTesting...\n")
        # test_loader is giving a new batch of testing values, hence why the output here is different than above.
        model_results = trainer.test(model, dataloaders=test_loader)


        print("\nExplaining\n")

        start = time.time()

        try:

            results, f = explain_with_timeout(model, val_data=val_loader, train_data=train_loader, test_data=test_loader,
                                        topk_expl=3,
                                        concepts=data.columns)

        except FunctionTimedOut:
            print("Explanation timed out, skipping...")
            continue


        end = time.time()

        """ The following lines were taken from the MIMIC example code by Pietro Barbiero"""

        print(f"Explaining time: {end - start}")
        results['model_accuracy'] = model_results[0]['test_acc_epoch']
        results['extraction_time'] = end - start

        for j in range(n_classes):
            n_used_concepts = sum(model.model[0].concept_mask[j] > 0.5)
            print(f"Number of features that impact on target {j}: {n_used_concepts}")
            print(f"Explanation for target {j}: {f[j]['explanation']}")
            print(f"Explanation accuracy: {f[j]['explanation_accuracy']}")

        """ End of reference code """


        splitResults = [results['model_accuracy'], results['extraction_time'], *scores, f]

        splitResults_list.append(splitResults)


    results_dict[file] = splitResults_list


# %%
# Helper function to remove explanation attempts that returned None.

def removeNoneExplanations(explanations):

    toRemove = []
    for idx, expl in enumerate(explanations):
        if expl['explanation'] == None:
            toRemove.append(idx)
    for i in sorted(toRemove, reverse=True):
        del explanations[i]

    return explanations

# %% [markdown]
# ### Averaging results across all folds

# %%
kFoldMeans = []


for x in results_dict:

    cols = ['file', 'model_accuracy', 'extraction_time', 'f1', 'recall', 'precision']

    # Fetching results
    rows = []

    class0Explanations = []
    class1Explanations = []

    for split in results_dict[x]:
        row = [x]
        
        # Model accuracy results
        row.extend(split[:5])

        rows.append(row)

        # Explanation accuracy results
        class0Explanations.append(split[5][0])
        class1Explanations.append(split[5][1])


    class0Explanations = removeNoneExplanations(class0Explanations)

    class1Explanations = removeNoneExplanations(class1Explanations)

    class0DF = pd.DataFrame(class0Explanations)
    class1DF = pd.DataFrame(class1Explanations)

    average0 = class0DF.mean().values
    average1 = class1DF.mean().values

    # If the explanation attempt returned None fill with zeros
    if len(class0Explanations) == 0:
        average0 = [0]*4

    if len(class1Explanations) == 0:
        average1 = [0]*4

    df = pd.DataFrame(columns=cols, data=rows)

    df = df.set_index('file')

    combinedCols = list(df.describe().columns)

    row = [x]
    row.extend(np.round(df.describe().loc['mean'].values, 2))

    row.extend(list(average0)[1:])
    row.extend(list(average1)[1:])

    kFoldMeans.append(row)



# Getting average, formatting into a dataframe

kFoldMeansCols = list(df.describe().columns)

combinedCols.insert(0, "file")


for idx, d in enumerate(results_dict[list(results_dict.keys())[0]][0][5]):
    combinedCols.extend([str(x) + "_" + str(idx) for x in list(d)[2:]])


totalMeans = pd.DataFrame(columns=combinedCols, data=kFoldMeans)

totalMeans = totalMeans.set_index('file')

cols = totalMeans.columns

cols = [c.replace("explanation", "expl").replace("accuracy", "acc").replace("complexity", "comp") for c in cols]

totalMeans.columns = cols

totalMeans = totalMeans.round(2)

totalMeans = totalMeans.drop("extraction_time", axis=1)

display(totalMeans)



timeNow = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
totalMeans.to_csv(f"./processingCache/totalMeans{timeNow}.csv")


