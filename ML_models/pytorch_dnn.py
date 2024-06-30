import random
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Set seed
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

class NeuralNetwork(nn.Module):
    def __init__(self, input_size: int = 2048, n_layers: int = 2, hidden_size: int = 256, dropout: float = 0.1,
                 output: int = 1, ):
        super().__init__()

        self.l = nn.ModuleList()
        self.l.append(nn.Linear(input_size, hidden_size))
        self.l.append(nn.ReLU())

        for i in range(n_layers):
            self.l.append(nn.Linear(hidden_size, hidden_size))
            self.l.append(nn.ReLU())

        if dropout:
            self.drop = nn.Dropout(dropout)

        self.out = nn.Linear(hidden_size, output)

    def forward(self, x):

        for layer in self.l:
            x = layer(x)

        return self.out(x)


class FFNN:

    def __init__(self):
        self.epochs = 100
        self.train_loss = []
        self.vali_loss = []
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.device = None
        self.batch_size = 10

    def train(self, x_train, y_train, x_val, y_val, epochs=None, es_patience=10, *args, **kwargs):

        # set epochs
        epochs = self.epochs if epochs is None else epochs

        # Load data
        train_load = DataLoader(Dataset(x_train, y_train), batch_size=self.batch_size, shuffle=True)
        print(f"Training on {len(x_train)} samples")

        # set model to training mode
        self.model.train()

        patience = 0

        for epoch in range(epochs):

            if patience >= es_patience:
                print(f"Early stopping at epoch {epoch}")
                return self.model
            else:
                loss = self.epoch(train_load)
                self.train_loss.append(loss)

                # set model to evaluation mode
                self.model.eval()

                if x_val is not None and y_val is not None:
                    val_pred = self.predict(x_val)
                    val_loss = self.loss_fn(val_pred, torch.Tensor(y_val))

                self.vali_loss.append(val_loss.item())

                if val_loss < min(self.vali_loss):
                    patience = 0
                    return self.model
                else:
                    patience += 1

        return self.model

    def predict(self, x, batch_size=10):

        data_load = DataLoader(Dataset(x, None), batch_size=batch_size, shuffle=False)

        # set model to evaluation mode
        self.model.eval()

        pred = []
        with torch.no_grad():
            for i, batch_data in enumerate(data_load):
                x = batch_data[0]
                x = x.to(self.device)
                prediction = self.model(x).squeeze(1).tolist()
                pred.extend(prediction)

        return torch.tensor(pred)

    def epoch(self, train_load):

        tr_loss = 0
        for i, batch_data in enumerate(train_load):
            # get data
            x, y = batch_data
            # move to device
            x, y = x.to(self.device), y.to(self.device)
            # zero grad
            self.optimizer.zero_grad()
            # forward pass
            pred = self.model(x)
            # loss
            loss = self.loss_fn(pred, y[:, None])
            # backward pass
            loss.backward()
            # update weights
            self.optimizer.step()
            tr_loss += loss.item()*x.shape[0]

        return tr_loss/len(train_load.dataset)

    def testing(self, x_test, y_test):

        # set model to evaluation mode
        self.model.eval()

        data_load = DataLoader(Dataset(x_test, y_test), batch_size=self.batch_size, shuffle=False)

        y_true, y_pred = [], []
        with torch.no_grad():
            for i, batch_data in enumerate(data_load):
                x, y = batch_data
                x, y = x.to(self.device), y.to(self.device)
                prediction = self.model(x).squeeze(1).tolist()
                y_pred.extend(prediction)
                y_true.extend(y.tolist())

        return torch.tensor(y_true), torch.tensor(y_pred)


class DNN(FFNN):
    def __init__(self, input_size: int = 2048, n_layers: int = 2, hidden_size: int = 256, dropout: float = 0,
                 output: int = 1, lr=1e-3,
                 *args, **kwargs):
        super().__init__()

        self.model = NeuralNetwork(input_size, n_layers, hidden_size, dropout, output)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.device = device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 10
        self.epochs = 100
        self.model = self.model.to(device)
        print(device)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):

        if targets is None:
            targets = np.zeros(len(features))
        if not torch.is_tensor(features) or not torch.is_tensor(targets):
            self.features = torch.Tensor(features)
            self.targets = torch.Tensor(targets)
        else:
            self.features = features
            self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        X = self.features[index]
        y = self.targets[index]
        return X, y


# if __name__ == '__main__':
#     model = DNN(2048, 1, 256, 0, 1)
#
#     df_data = pd.read_csv('chembl_33_IC50.csv').query('tid == 206')  # .sample(n=1000, random_state=0)
#
#     from rdkit import Chem
#     from rdkit.Chem import AllChem
#
#     features_fp = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2, nBits=2048) for x in
#                    df_data.nonstereo_aromatic_smiles.values]
#     targets_pot = df_data.pPot.values
#     print(f"Features: {len(features_fp)}")
#     print(f"Targets: {len(targets_pot)}")
#
#     train_x, val_x, train_y, val_y = train_test_split(np.array(features_fp), np.array(targets_pot), train_size=0.9,
#                                                       test_size=0.1, random_state=0)
#
#     print(f"Train: {len(train_x)}, {len(train_y)}, Val: {len(val_x)}, {len(val_y)}")
#     model_tr = model.train(train_x, train_y, val_x, val_y, epochs=100, es_patience=100)
#
#     #val_pred = model.testing(val_x, val_y)
#
#     import matplotlib.pyplot as plt
#     train_losses_float = [float(loss.cpu().detach().numpy()) for loss in model.train_loss]
#     val_losses_float = [float(loss) for loss in model.vali_loss]
#     loss_indices = [i for i, l in enumerate(train_losses_float)]
#
#     plt.figure()
#     plt.plot(loss_indices, train_losses_float, val_losses_float)
#     plt.show()
