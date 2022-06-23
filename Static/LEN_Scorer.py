import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import one_hot
import torch_explain as te
from torch_explain.nn.functional import l1_loss
from torch_explain.logic.nn import psi
from torch_explain.logic.metrics import test_explanation , complexity
from torch_explain.logic.nn import entropy
import numpy as np

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

        layers = [
            te.nn.EntropyLinear(x_train.shape[1], 10, n_classes=4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(10, 4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(4, 1),
        ]


        model = torch.nn.Sequential(*layers)


        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss_form = torch.nn.CrossEntropyLoss()
        model.to(device=self.cuda)
        model.train()
        for _ in range(1001):
            optimizer.zero_grad()
            y_pred = model(x_train.to(device=self.cuda)).squeeze(-1)
            # print(y_pred, y_train)
            loss = loss_form(y_pred, y_train)
            loss = loss_form(y_pred, y_train) + 0.00001 * te.nn.functional.entropy_logic_loss(model)
            print(loss)
            loss.backward()

            clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

        self.model = model


    def explain(self, class_target):

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