import pandas as pd
import torch
from torch.nn.functional import one_hot
import torch_explain as te
from torch_explain.nn.functional import l1_loss
from torch_explain.logic.nn import psi
from torch_explain.logic.metrics import test_explanation , complexity
from torch_explain.logic.nn import entropy

class Scorer:

    """Runs categorised static data on a standard LEN network, and outputs the result."""

    def __init__(self, file_name):
        self.score_file = self.load_scores(file_name)
        
        self.score()
        
    
    def load_scores(self, file_name):
        return pd.read_csv(file_name)

    self.score(self):
        x0 = torch.zeros((4, 100))
        x_train = torch.tensor([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ], dtype=torch.float)
        x_train = torch.cat([x_train, x0], dim=1)
        y_train = torch.tensor([1, 0, 0, 1], dtype=torch.long)

        layers = [
            te.nn.EntropyLinear(x_train.shape[1], 10, n_classes=2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(10, 4),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(4, 1),
        ]
        model = torch.nn.Sequential(*layers)


        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss_form = torch.nn.CrossEntropyLoss()
        model.train()
        for epoch in range(1001):
            optimizer.zero_grad()
            y_pred = model(x_train).squeeze(-1)
            loss = loss_form(y_pred, y_train)
            print(y_pred, y_train)
            loss = loss_form(y_pred, y_train) + 0.00001 * te.nn.functional.entropy_logic_loss(model)
            loss.backward()
            optimizer.step()
