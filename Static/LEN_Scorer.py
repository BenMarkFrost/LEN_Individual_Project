import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import one_hot
import torch_explain as te
from torch_explain.nn.functional import l1_loss
from torch_explain.logic.nn import psi
from torch_explain.logic.metrics import test_explanation , complexity
from torch_explain.logic.nn import entropy
from torch_explain.models.explainer import Explainer
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.linear_model import LassoCV
# import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from torch_explain.logic.metrics import formula_consistency
import os
from matplotlib import pyplot as plt
import seaborn as sns
import time
from torchmetrics.functional import precision_recall
from pytorch_lightning.callbacks.early_stopping import EarlyStopping



""" The model code from this file is adapted from the following:
https://github.com/pietrobarbiero/pytorch_explain/blob/master/experiments/elens/mnist.py

Credit to Pietro Barbiero for the original code."""

class Scorer:

    """Runs categorised static data on a standard LEN network, and outputs the result."""

    def __init__(self, data, target, concept_names):

        xTensor = torch.FloatTensor(data)
        yTensor = one_hot(torch.tensor(target).to(torch.long)).to(torch.float)

        self.data = xTensor
        self.target = yTensor
        self.cuda = torch.device('cuda')
        self.concept_names = concept_names
        

    def train(self, layer_complexity=[20]):
        # x_train = torch.tensor(self.data, dtype=torch.float, device = self.cuda)
        # y_train = torch.tensor(self.target, dtype=torch.long, device = self.cuda)

        x_train = self.data
        y_train = self.target

        # print(x_train)
        # print(y_train)

        # layers = [
        #     te.nn.EntropyLinear(x_train.shape[1], 10, n_classes=4),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.Linear(10, 4),
        #     torch.nn.LeakyReLU(),
        #     torch.nn.Linear(4, 1),
        # ]


        # model = torch.nn.Sequential(*layers)


        # optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        # loss_form = torch.nn.CrossEntropyLoss()
        # model.to(device=self.cuda)
        # model.train()
        # for _ in range(1001):
        #     optimizer.zero_grad()
        #     y_pred = model(x_train.to(device=self.cuda)).squeeze(-1)
        #     # print(y_pred, y_train)
        #     loss = loss_form(y_pred, y_train)
        #     loss = loss_form(y_pred, y_train) + 0.00001 * te.nn.functional.entropy_logic_loss(model)
        #     print(loss)
        #     loss.backward()

        #     clip_grad_norm_(model.parameters(), 5)

        #     optimizer.step()

        dataset = TensorDataset(x_train, y_train)
        train_size = int(0.8 * len(dataset))

        val_size = (len(dataset) - train_size) // 2
        test_size = len(dataset) - train_size - val_size

        train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

        train_loader = DataLoader(train_data, batch_size=train_size)
        val_loader = DataLoader(val_data, batch_size=val_size)
        test_loader = DataLoader(test_data, batch_size=test_size)

        

        n_concepts = next(iter(train_loader))[0].shape[1]
        self.n_concepts = n_concepts


        n_classes = self.target.shape[1]
        self.n_classes = n_classes

        print("Training on {} classes".format(n_classes))

        print("Num concepts: {}".format(n_concepts))
        print("Num classes: {}".format(n_classes))

        base_dir = f'./results/mimicLEN/explainer'
        os.makedirs(base_dir, exist_ok=True)

        seed_everything(40)

        n_splits = 2

        self.n_splits = n_splits

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        self.skf = skf

        results_list = []
        feature_selection = []
        explanations = {i: [] for i in range(n_classes)}

        x = x_train
        y = y_train

        for split, (trainval_index, test_index) in enumerate(skf.split(x.cpu().detach().numpy(),
                                                               y.argmax(dim=1).cpu().detach().numpy())):
            
            # print(x)

            # x = x.cpu()

            # print(x)
            # x = x.to(torch.device("cpu"))
            # y = y.float()
            # y = y.to(torch.device("cpu"))
            # y = one_hot(y.to(torch.int64)).to(torch.float)

            # print(x.shape)
            # print(y, y.shape)


            print(f'Split [{split + 1}/{n_splits}]')
            x_trainval, x_test = torch.FloatTensor(x[trainval_index]), torch.FloatTensor(x[test_index])
            y_trainval, y_test = torch.FloatTensor(y[trainval_index]), torch.FloatTensor(y[test_index])
            x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.2, random_state=42)
            print(f'{len(y_train)}/{len(y_val)}/{len(y_test)}')

            train_data = TensorDataset(x_train, y_train)
            val_data = TensorDataset(x_val, y_val)
            test_data = TensorDataset(x_test, y_test)
            train_loader = DataLoader(train_data, batch_size=train_size)
            val_loader = DataLoader(val_data, batch_size=val_size)
            test_loader = DataLoader(test_data, batch_size=test_size)

            checkpoint_callback = ModelCheckpoint(dirpath=base_dir, monitor='val_loss', save_top_k=1)

            # Constructs the way that the model will be trained
            trainer = Trainer(max_epochs=200, gpus=1, auto_lr_find=True, deterministic=True,
                            check_val_every_n_epoch=1, default_root_dir=base_dir,
                            weights_save_path=base_dir, callbacks=[checkpoint_callback])

            # This is the model itself, which is extended from pytorch_lightning
            model = Explainer(n_concepts=n_concepts, n_classes=n_classes, l1=1e-3, lr=0.01,
                            explainer_hidden=layer_complexity, temperature=0.7)

            start = time.time()
            trainer.fit(model, train_loader, val_loader)
            print(f"Gamma: {model.model[0].concept_mask}")
            model.freeze()
            print("\nTesting...\n")
            model_results = trainer.test(model, test_dataloaders=test_loader)
            print("Testing results: ", model_results)
            print("\nExplaining\n")
            results, f = model.explain_class(val_loader, train_loader, test_loader,
                                            topk_explanations=10,
                                            concept_names=self.concept_names)
            end = time.time()
            print(f"Explaining time: {end - start}")
            results['model_accuracy'] = model_results[0]['test_acc']
            results['extraction_time'] = end

            results_list.append(results)
            extracted_concepts = []
            all_concepts = model.model[0].concept_mask[0] > 0.5
            common_concepts = model.model[0].concept_mask[0] > 0.5
            for j in range(n_classes):
                # print(f[j]['explanation'])
                n_used_concepts = sum(model.model[0].concept_mask[j] > 0.5)
                print(f"Number of features that impact on target {j}: {n_used_concepts}")
                print(f"Explanation for target {j}: {f[j]['explanation']}")
                print(f"Explanation accuracy: {f[j]['explanation_accuracy']}")
                explanations[j].append(f[j]['explanation'])
                extracted_concepts.append(n_used_concepts)
                all_concepts += model.model[0].concept_mask[j] > 0.5
                common_concepts *= model.model[0].concept_mask[j] > 0.5

            results['extracted_concepts'] = np.mean(extracted_concepts)
            results['common_concepts_ratio'] = sum(common_concepts) / sum(all_concepts)

            # Precision, Recall, F1
            print(x_test)
            print("Type:", type(x_test))
            y_pred = torch.argmax(model(x_test), axis=1)
            print("Predictions:", y_pred)
            y_test_argmax = torch.argmax(y_test, axis=1)
            print("Actual:", y_test_argmax)

            prec_rec = precision_recall(y_pred, y_test_argmax, num_classes = n_classes)

            print(prec_rec)

            # compare against standard feature selection
            i_mutual_info = mutual_info_classif(x_trainval, y_trainval[:, 1])
            i_chi2 = chi2(x_trainval, y_trainval[:, 1])[0]
            i_chi2[np.isnan(i_chi2)] = 0
            lasso = LassoCV(cv=5, random_state=0).fit(x_trainval, y_trainval[:, 1])
            i_lasso = np.abs(lasso.coef_)
            i_mu = model.model[0].concept_mask[1]
            # print(model.model[0].concept_mask)
            df = pd.DataFrame(np.hstack([
                i_mu.numpy(),
                # i_mutual_info / np.max(i_mutual_info),
                # i_chi2 / np.max(i_chi2),
                # i_lasso / np.max(i_lasso),
            ]).T, columns=['feature importance'])
            df['method'] = 'explainer'
            # df.iloc[90:, 1] = 'MI'
            # df.iloc[180:, 1] = 'CHI2'
            # df.iloc[270:, 1] = 'Lasso'
            df['feature'] = np.hstack([np.arange(0, n_concepts)])
            feature_selection.append(df)



        self.feature_selection = feature_selection
        # print(self.feature_selection)

        self.df = df
        self.explanations = explanations
        self.results_list = results_list

        return y_pred, y_test_argmax



    def explain(self):

        # print("Explaining class: ", class_target)

        # if self.x_train == None or self.y_train == None or self.model == None:
        #     raise Exception("Model not trained")

        # self.x_train.cpu()
        # self.y_train.cpu()
        # self.model.cpu()

        # y1h = one_hot(self.y_train.cpu())

        # explanation, _ = entropy.explain_class(self.model.cpu(), self.x_train.cpu(), y1h, self.x_train.cpu(), y1h, target_class=class_target)

        # # print(model(x_train[0]))

        # accuracy, preds = test_explanation(explanation, self.x_train, y1h, target_class=class_target)
        # explanation_complexity = complexity(explanation)

        # return [explanation, explanation_complexity, accuracy, preds]

        base_dir = f'./results/mimicLEN/explainer'

        consistencies = []
        print(self.explanations)
        for j in range(self.n_classes):
            if self.explanations[j][0] is None:
                continue
            consistencies.append(formula_consistency(self.explanations[j]))
        explanation_consistency = np.mean(consistencies)

        feature_selection = pd.concat(self.feature_selection, axis=0)

        print("Feature selection: ", feature_selection)

        f1 = feature_selection[feature_selection['feature'] <= self.n_concepts//3]
        f2 = feature_selection[(feature_selection['feature'] > self.n_concepts//3) & (feature_selection['feature'] <= (self.n_concepts*2)//3)]
        f3 = feature_selection[feature_selection['feature'] > (self.n_concepts*2)//3]

        plt.figure(figsize=[10, 10])
        plt.subplot(1, 3, 1)
        ax = sns.barplot(y=f1['feature'], x=f1.iloc[:, 0],
                        hue=f1['method'], orient='h', errwidth=0.5, errcolor='k')
        ax.get_legend().remove()
        plt.subplot(1, 3, 2)
        ax = sns.barplot(y=f2['feature'], x=f2.iloc[:, 0],
                        hue=f2['method'], orient='h', errwidth=0.5, errcolor='k')
        plt.xlabel('')
        ax.get_legend().remove()
        plt.subplot(1, 3, 3)
        sns.barplot(y=f3['feature'], x=f3.iloc[:, 0],
                    hue=f3['method'], orient='h', errwidth=0.5, errcolor='k')
        plt.xlabel('')
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, 'barplot_mimic.png'))
        plt.savefig(os.path.join(base_dir, 'barplot_mimic.pdf'))
        plt.show()

        # print(feature_selection.iloc[:, 1], feature_selection.iloc[:, 0])

        # plt.figure(figsize=[6, 4])
        # sns.boxplot(x=feature_selection.iloc[:, 1], y=feature_selection.iloc[:, 0])
        # plt.tight_layout()
        # plt.savefig(os.path.join(base_dir, 'boxplot_mimic.png'))
        # plt.savefig(os.path.join(base_dir, 'boxplot_mimic.pdf'))
        # plt.show()


        results_df = pd.DataFrame(self.results_list)
        results_df['explanation_consistency'] = explanation_consistency
        results_df.to_csv(os.path.join(base_dir, 'results_aware_mimic.csv'))
        results_df


        results_df.mean()

        results_df.sem()

        x = self.data
        y = self.target

        dt_scores, rf_scores = [], []
        for split, (trainval_index, test_index) in enumerate(
                self.skf.split(x.cpu().detach().numpy(), y.argmax(dim=1).cpu().detach().numpy())):
            print(f'Split [{split + 1}/{self.n_splits}]')
            x_trainval, x_test = x[trainval_index], x[test_index]
            y_trainval, y_test = y[trainval_index].argmax(dim=1), y[test_index].argmax(dim=1)

            dt_model = DecisionTreeClassifier(max_depth=5, random_state=split)
            dt_model.fit(x_trainval, y_trainval)
            dt_scores.append(dt_model.score(x_test, y_test))

            rf_model = RandomForestClassifier(random_state=split)
            rf_model.fit(x_trainval, y_trainval)
            rf_scores.append(rf_model.score(x_test, y_test))

        print(f'Random forest scores: {np.mean(rf_scores)} (+/- {np.std(rf_scores)})')
        print(f'Decision tree scores: {np.mean(dt_scores)} (+/- {np.std(dt_scores)})')
        print(f'Mu net scores (model): {results_df["model_accuracy"].mean()} (+/- {results_df["model_accuracy"].std()})')
        print(
            f'Mu net scores (exp): {results_df["explanation_accuracy"].mean()} (+/- {results_df["explanation_accuracy"].std()})')




    def predict(self, x):
        if not self.x_train or not self.y_train or not self.model:
            raise Exception("Model not trained")

        x = torch.tensor(x, dtype=torch.float)
        y_pred = self.model(x).squeeze(-1)
        return y_pred