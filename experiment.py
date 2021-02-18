import json
import os
import numpy as np
from data import Data
from model import Model
from time import time

class ExperimentBase:

    def __init__(self, P, D, N, resultspath):
        self.data = Data(P, D, N)
        self.model = Model(P, D)
        self.resultspath = os.path.join('results', resultspath) + '.json'
        self.results = None

    def run(self):
        pass

    def save(self):
        if self.results is None:
            return
        self.results['description'] = self.describe()
        curr = []
        if os.path.exists(self.resultspath):
            with open(self.resultspath, 'r') as fp:
                curr = json.loads(fp.read())
        else:
            open(self.resultspath, 'a').close()
        curr.append(self.results)
        with open(self.resultspath, 'w') as fp:
            fp.write(json.dumps(curr))

    def describe(self):
        out = self.model.describe()
        out['P'] = self.data.P
        out['D'] = self.data.D
        out['N'] = self.data.N
        return out

class InterpolationExperiment(ExperimentBase):

    def __init__(self, P, D, N, resultspath='interpolation', \
                            loss_threshold=1e-4, s=0.05, u=.2, r=5):
        super().__init__(P, D, N, resultspath)
        self.loss_threshold = loss_threshold
        self.s = s
        self.u = u
        self.mr = r
        self.cr = 0
        self.triplets = []
        self.current_epochs = self.model.epochs
        self.has_interpolated = False

    def run(self):
        self.choose_sample()
        record = []
        for r in range(self.mr):
            self.has_interpolated, losses = self.interpolating_on_sample()
            record.append([self.current_epochs, np.max(losses)])
            if self.has_interpolated:
                break
            self.current_epochs = int(self.current_epochs * (1+self.u))
            self.cr += 1
        out = {'timestamp': str(time())}
        out['interpolated'] = self.has_interpolated
        out['rounds'] = r
        out['record'] = record
        out['final_epochs'] = self.current_epochs
        self.results = out

    def interpolating_on_sample(self):
        losses = []
        l = len(self.triplets)
        for i, triplet in enumerate(self.triplets):
            loss = self.model.learn_triplet(*triplet, epochs=self.current_epochs)
            if i % 10:
                print(f"round={self.cr}/{self.mr}, triplet={i}/{l}: {loss}")
            losses.append(loss)
        losses = np.array(losses)
        thresh = np.full_like(losses, self.loss_threshold)
        if np.all(losses <= thresh):
            return True, losses
        return False, losses

    def choose_sample(self):
        triplets = []
        for pi, pj, label in self.data.iterate_triplets():
            if np.random.uniform(low=0, high=1) < self.s:
                triplets.append([pi, pj, label['noisy']])
        self.triplets = triplets
    
class ExcessRiskExperiment(ExperimentBase):

    def __init__(self, P, D, N, resultspath='excess_risk', cer_threshold=0.1):
        super().__init__(P, D, N, resultspath)
        self.cer_threshold = 0.1
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
                # print(f"observed: {self.oberved}, c.e.r: {self.curr_excess_risk}")
                if curr < self.cer_threshold:
                    stop = True
                    break
        self.results = {'timestamp': str(time()), 'oberved': observed, 'parameters': {'p1': 'v1'}}

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

if __name__ == '__main__':
    P = 100
    D = 1
    N = 10

    exp = InterpolationExperiment(P, D, N)
    exp.run()
    # print(exp.results)
    exp.save()
