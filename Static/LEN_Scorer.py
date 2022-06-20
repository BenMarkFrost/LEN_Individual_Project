import pandas as pd
import torch
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
        

    def score(self):
        x_train = torch.tensor(self.data.to_numpy(), dtype=torch.float)
        y_train = torch.tensor(self.target.to_numpy(), dtype=torch.long)

        print(y_train)

        layers = [
            te.nn.EntropyLinear(x_train.shape[1], 10, n_classes=4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(10, 4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(4, 4),
        ]
        model = torch.nn.Sequential(*layers)


        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss_form = torch.nn.CrossEntropyLoss()
        model.train()
        for _ in range(1001):
            optimizer.zero_grad()
            y_pred = model(x_train).squeeze(-1)
            # print((y_pred))
            loss = loss_form(y_pred, y_train)
            loss = loss_form(y_pred, y_train) + 0.00001 * te.nn.functional.entropy_logic_loss(model)
            print(loss)
            loss.backward()
            optimizer.step()

        print(y_pred, y_train)
