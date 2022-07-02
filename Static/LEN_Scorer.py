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
import os
import time


""" The model code from this file is based on the following:
https://github.com/pietrobarbiero/pytorch_explain/blob/master/experiments/elens/mnist.py

Credit to Pietro Barbiero for the original code."""

class Scorer:

    """Runs categorised static data on a standard LEN network, and outputs the result."""

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.cuda = torch.device('cuda')
        

    def train(self):
        x_train = torch.tensor(self.data, dtype=torch.float, device = self.cuda)
        y_train = torch.tensor(self.target, dtype=torch.long, device = self.cuda)

        self.x_train = x_train
        self.y_train = y_train

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
        n_classes = len(np.unique(self.target))

        print("Num concepts: {}".format(n_concepts))
        print("Num classes: {}".format(n_classes))

        base_dir = f'./results/mimicLEN/explainer'
        os.makedirs(base_dir, exist_ok=True)

        seed_everything(40)

        n_splits = 5

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        results_list = []
        explanations = {i: [] for i in range(n_classes)}

        x = x_train
        y = y_train

        for split, (trainval_index, test_index) in enumerate(skf.split(x.cpu().detach(),
                                                               y.cpu().detach())):
            
            # print(x)

            # x = x.cpu()

            # print(x)
            x = x.to(torch.device("cpu"))
            y = y.float()
            y = y.to(torch.device("cpu"))
            y = one_hot(y.to(torch.int64)).to(torch.float)

            print(x.shape)
            print(y, y.shape)


            print(f'Split [{split + 1}/{n_splits}]')
            x_trainval = torch.FloatTensor(x[trainval_index])
            x_test = torch.FloatTensor(x[test_index])
            y_trainval = torch.FloatTensor(y[trainval_index])
            y_test = torch.FloatTensor(y[test_index])
            x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.2, random_state=42)
            print(f'{len(y_train)}/{len(y_val)}/{len(y_test)}')

            train_data = TensorDataset(x_train, y_train)
            val_data = TensorDataset(x_val, y_val)
            test_data = TensorDataset(x_test, y_test)
            train_loader = DataLoader(train_data, batch_size=train_size)
            val_loader = DataLoader(val_data, batch_size=val_size)
            test_loader = DataLoader(test_data, batch_size=test_size)

            checkpoint_callback = ModelCheckpoint(dirpath=base_dir, monitor='val_loss', save_top_k=1)
            trainer = Trainer(max_epochs=200, gpus=1, auto_lr_find=True, deterministic=True,
                            check_val_every_n_epoch=1, default_root_dir=base_dir,
                            weights_save_path=base_dir, callbacks=[checkpoint_callback])
            model = Explainer(n_concepts=n_concepts, n_classes=n_classes, l1=1e-3, lr=0.01,
                            explainer_hidden=[20], temperature=0.7)

            start = time.time()
            trainer.fit(model, train_loader.to(torch.float), val_loader.to(torch.float))
            print(f"Gamma: {model.model[0].concept_mask}")
            model.freeze()
            model_results = trainer.test(model, test_dataloaders=test_loader)
            for j in range(n_classes):
                n_used_concepts = sum(model.model[0].concept_mask[j] > 0.5)
                print(f"Extracted concepts: {n_used_concepts}")
            results, f = model.explain_class(val_loader, train_loader, test_loader,
                                            topk_explanations=10)
            end = time.time() - start

            print(f"Time: {end}")
            print(results, f)


    def explain(self, class_target):

        print("Explaining class: ", class_target)

        if self.x_train == None or self.y_train == None or self.model == None:
            raise Exception("Model not trained")

        self.x_train.cpu()
        self.y_train.cpu()
        self.model.cpu()

        y1h = one_hot(self.y_train.cpu())

        explanation, _ = entropy.explain_class(self.model.cpu(), self.x_train.cpu(), y1h, self.x_train.cpu(), y1h, target_class=class_target)

        # print(model(x_train[0]))

        accuracy, preds = test_explanation(explanation, self.x_train, y1h, target_class=class_target)
        explanation_complexity = complexity(explanation)

        return [explanation, explanation_complexity, accuracy, preds]



    def predict(self, x):
        if not self.x_train or not self.y_train or not self.model:
            raise Exception("Model not trained")

        x = torch.tensor(x, dtype=torch.float)
        y_pred = self.model(x).squeeze(-1)
        return y_pred