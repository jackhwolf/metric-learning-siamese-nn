import torch
import numpy as np
from torch.autograd import Variable
from noise import Noise

class hingeloss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        hinge_loss = 1 - torch.mul(output, target)
        hinge_loss[hinge_loss < 0] = 0
        return hinge_loss

class Model:

    def __init__(self, P, D, lr=1e-3, wd=1e-5, epochs=100):
        self.P = int(P)
        self.D = int(D)
        self.lr, self.wd, self.epochs = (float(lr), float(wd), int(epochs))
        self.criterion = hingeloss()
        self.x_hat = self.to_var((torch.rand((1, self.P))*2) - 1)
        self.l_hat = self.to_var((torch.rand((self.P, self.D))*2) - 1)
        params = [self.x_hat, self.l_hat]
        self.optimizer = torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd)
        self.const_inp_x = torch.FloatTensor([1])
        self.const_inp_l = torch.FloatTensor(np.identity(self.P))

    def learn_triplet(self, point_i, point_j, noisy_label, epochs=None):
        if epochs is None:
            epochs = self.epochs
        noisy_label = self.to_var(noisy_label, False)
        for i in range(epochs):
            distance = self.forward(point_i, point_j)
            loss = self.criterion(distance, noisy_label).sum()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss.item()

    def predict_triplet(self, point_i, point_j):
        pred = None
        with torch.no_grad():
            pred = self.forward(point_i, point_j)
        pred = pred.detach().numpy().item()
        scaled_pred, noisy_pred = Noise(pred)
        prediction_info = {
            'distance': pred,
            'mu': scaled_pred,
            'true': np.sign(pred),
            'noisy': noisy_pred
        }
        return prediction_info

    def forward(self, point_i, point_j):
        point_i, point_j = self.to_var(point_i), self.to_var(point_j)
        dist_i = self.forward_one(point_i)
        dist_j = self.forward_one(point_j)
        distance = (dist_j - dist_i).reshape(-1)
        return distance

    def forward_one(self, point):
        self.const_inp_x.matmul(self.x_hat)
        self.const_inp_l.matmul(self.l_hat)
        point_transform = point.matmul(self.l_hat)
        distance = (point_transform-self.x_hat.matmul(self.l_hat))
        distance = distance.pow(2).sum()
        return distance

    def to_var(self, foo, rg=True):
        if not isinstance(foo, (list, np.ndarray, torch.Tensor)):
            foo = np.array([foo])
        return Variable(torch.FloatTensor(foo), requires_grad=rg)

    def describe(self):
        out = {}
        out['lr'] = self.lr
        out['wd'] = self.wd
        out['epochs'] = self.epochs
        return out
