import os
import numpy as np
from time import time
from data import Data
from model import Model

class ExperimentBase:

    def __init__(self, P, D, N, eid, resultspath, modelargs={}):
        self.data = Data(P, D, N)
        self.model = Model(P, D, **modelargs)
        self.eid = eid
        self.resultspath = os.path.join('Files', 'results', resultspath) + '.json'

    def run(self):
        pass

    def describe(self):
        out = {}
        out.update(self.model.describe())
        out.update(self.data.describe())
        return out

class ExperimentPool:

    def __init__(self, poolsize, experiment, experiment_args, model_args):
        self.poolsize = int(poolsize)
        self.experiment = experiment
        self.eargs = experiment_args
        self.margs = model_args
        self.pool = [experiment(eid=i, modelargs=self.margs, **self.eargs) for i in range(self.poolsize)]

    def __getitem__(self, i):
        return self.pool[i]

    def __len__(self):
        return self.poolsize

class InterpolationExperiment(ExperimentBase):

    def __init__(self, P=10, D=3, N=5, eid=1, loss_threshold=1e-4, s=0.05, u=.2, r=3, modelargs={}):
        super().__init__(P, D, N, eid, 'interpolation', modelargs)
        self.loss_threshold = float(loss_threshold)
        self.s = float(s)
        self.u = float(u)
        self.mr = int(r)
        self.cr = 0
        self.triplets = []
        self.current_epochs = self.model.epochs
        self.has_interpolated = False

    def run(self):
        self.choose_sample()
        record = []
        for r in range(self.mr):
            self.has_interpolated, losses = self.interpolating_on_sample()
            record.append({'epochs': self.current_epochs, 'max_loss': np.max(losses)})
            if self.has_interpolated:
                break
            self.current_epochs = int(self.current_epochs * (1+self.u))
            self.cr += 1
        out = {'timestamp': str(time())}
        out['interpolated'] = self.has_interpolated
        out['rounds'] = r
        out['record'] = record
        out['final_epochs'] = self.current_epochs
        out['threshold'] = self.loss_threshold
        out['parameters'] = self.describe()
        return out

    def interpolating_on_sample(self):
        losses = []
        l = len(self.triplets)
        for i, triplet in enumerate(self.triplets):
            loss = self.model.learn_triplet(*triplet, epochs=self.current_epochs)
            if i % 10:
                print(f"eid={self.eid}, round={self.cr}/{self.mr}, triplet={i}/{l}: loss={loss}")
            losses.append(loss)
        losses = np.array(losses)
        if np.mean(losses) <= self.loss_threshold:
            return True, losses
        return False, losses

    def choose_sample(self):
        triplets = []
        for pi, pj, label in self.data.iterate_triplets():
            if np.random.uniform(low=0, high=1) < self.s:
                triplets.append([pi, pj, label['noisy']])
        if len(triplets) == 0:
            triplets.append([pi, pj, label['noisy']])
        self.triplets = triplets
    
    def describe(self):
        out = super().describe()
        out['s'] = self.s
        out['u'] = self.u
        out['r'] = self.mr
        return out

class ExcessRiskExperiment(ExperimentBase):

    def __init__(self, P=10, D=3, N=5, eid=1, cer_threshold=0.1, modelargs={}):
        super().__init__(P, D, N, eid, 'excess_risk', modelargs)
        self.cer_threshold = cer_threshold
        self.observed = 0
        self.curr_excess_risk = None

    def run(self):
        stop = False
        loss = None
        while not stop:
            for pi, pj, label in self.data.iterate_triplets():
                loss = self.model.learn_triplet(pi, pj, label['noisy'])
                self.curr_excess_risk = self.relative_excess_risk()
                self.observed += 1
                print(f"observed: {self.observed}, c.e.r: {self.curr_excess_risk}")
                if self.curr_excess_risk < self.cer_threshold:
                    stop = True
                    break
        out = {'timestamp': str(time())}
        out['observed'] = self.observed
        out['parameters'] = self.describe()
        return out

    ''' compute relative excess risk (R(*) - R(^)) / R(^) '''
    def relative_excess_risk(self):
        risk_star = 0
        risk_hat = 0
        count = 0
        for pi, pj, true_label_info in self.data.iterate_triplets():
            count += 1
            model_prediction_info = self.model.predict_triplet(pi, pj)
            risk_star += self.expectation(true_label_info)
            risk_hat += self.expectation(model_prediction_info)
        risk_star = risk_star/count
        risk_hat = risk_hat/count
        rel_excess_risk = abs(risk_hat - risk_star) / risk_star
        return rel_excess_risk

    ''' compute expectation for prediction '''
    def expectation(self, info):
        mu, t, f = info['mu'], info['true'], -1 * info['true']
        return (mu * t) + ((1-mu) * f)

