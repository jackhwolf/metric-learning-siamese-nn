import os
import numpy as np
from time import time
import torch
from data import Data
from model import Model

''' each experiment has a data obj, model obj, and id '''
class ExperimentBase:

	def __init__(self, P, D, N, eid, resultspath, modelargs={}):
		self.data = Data(P, D, N)
		self.model = Model(P, D, **modelargs)
		self.eid = eid
		self.resultspath = os.path.join('Files', 'results', resultspath) + '.json'

	def run(self):
		pass

	def describe(self):
		''' return params '''
		out = {}
		out.update(self.model.describe())
		out.update(self.data.describe())
		return out

	def average_results(self, pool_res):
		return pool_res

''' run a pool of N of the same experiments for averaging results '''
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

''' experiment to determine if model parameters will 
interpolate on each triplet of given data '''
class InterpolationExperiment(ExperimentBase):

	def __init__(self, P=10, D=3, N=5, eid=1, loss_threshold=1e-4, s=0.05, u=.2, r=3, modelargs={}):
		super().__init__(P, D, N, eid, 'interpolation', modelargs)
		self.loss_threshold = float(loss_threshold)
		self.s = float(s)
		self.u = float(u)
		self.mr = int(r)
		self.cr = 1
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
		out['rounds'] = self.cr
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

''' experiment to determine if model parameters will accurately predict '''
class PredictionExperiment(ExperimentBase):

	def __init__(self, P=10, D=3, N=5, eid=1, acc_threshold=1e-4, s=0.05, u=.2, r=3, modelargs={}):
		super().__init__(P, D, N, eid, 'prediction', modelargs)
		self.acc_threshold = float(acc_threshold)
		self.s = float(s)
		self.u = float(u)
		self.mr = int(r)
		self.cr = 1
		self.triplets = []
		self.current_epochs = self.model.epochs
		self.record = []
		self.is_acc = False
		self.acc = np.inf

	def run(self):
		self.choose_sample()
		for r in range(self.mr):
			self.acc = self.learn_predict_sample()
			self.is_acc = self.acc < self.acc_threshold
			self.record.append({'epochs': self.current_epochs, 'accuracy': self.acc})
			print(f"Round: {r} Acc: {self.acc}")
			if self.is_acc:
				break
			self.current_epochs = int(self.current_epochs * (1+self.u))
			self.cr += 1
		return self.describe()

	def learn_predict_sample(self):
		for i, t in enumerate(self.triplets):
			if i % 50 == 0:
				print(f"learned {i}/{len(self.triplets)}")
			self.model.learn_triplet(t[0], t[1], t[2]['noisy'], epochs=self.current_epochs)
		outputs = []
		targets = []
		acc, count = 0, 0
		for triplet in self.triplets:
			output = self.model.predict_triplet(triplet[0], triplet[1])['true']
			target = triplet[2]['true']
			if output == target:
				acc += 1
			count += 1
		return acc/count

	def choose_sample(self):
		triplets = []
		for pi, pj, label in self.data.iterate_triplets():
			if np.random.uniform(low=0, high=1) < self.s:
				triplets.append([pi, pj, label])
		if len(triplets) == 0:
			triplets.append([pi, pj, label])
		self.triplets = triplets

	def describe(self):
		out = {}
		out['timestamp'] = str(time())
		out['accurate'] = self.is_acc
		out['rounds'] = self.cr
		out['record'] = self.record
		out['final_epochs'] = self.current_epochs
		out['parameters'] = super().describe()
		out['parameters']['acc_threshold'] = self.acc_threshold
		out['parameters']['s'] = self.s
		out['parameters']['u'] = self.u
		out['parameters']['r'] = self.mr
		return out

''' experiment to determine number of triplets needed for relative excess
risk between model and user/data truth to be less than 0.1 '''
class ExcessRiskExperiment(ExperimentBase):

	def __init__(self, P=10, D=3, N=5, eid=1, cer_threshold=0.1, modelargs={}):
		super().__init__(P, D, N, eid, 'excess_risk', modelargs)
		self.cer_threshold = cer_threshold
		self.observed = 0
		self.risk_star = self.calc_risk_star()
		self.risk_hat = None

	def run(self):
		stop = False
		while not stop:
			for pi, pj, label in self.data.iterate_triplets():
				loss = self.model.learn_triplet(pi, pj, label['noisy'])
				self.curr_excess_risk = self.relative_excess_risk()
				self.observed += 1
				print(f"observed: {self.observed}, loss: {loss}, c.e.r: {self.curr_excess_risk}")
				if self.curr_excess_risk < self.cer_threshold or self.observed > 10:
					stop = True
					break
		out = {'timestamp': str(time())}
		out['observed'] = self.observed
		out['parameters'] = self.describe()
		return out

	''' compute relative excess risk (R(*) - R(^)) / R(^) '''
	def relative_excess_risk(self):
		self.risk_hat = self.calc_risk_hat()
		rs, rh = self.risk_star[0], self.risk_hat[0]
		return (rh - rs) / rs

	def calc_risk_hat(self):
		rh = 0
		count = 0
		for pi, pj, label in self.data.iterate_triplets():
			count += 1
			model_label = self.model.predict_triplet(pi, pj)
			rh += self.expectation(model_label)
		return (rh, count)

	def calc_risk_star(self):
		rs = 0
		count = 0
		for pi, pj, label in self.data.iterate_triplets():
			count += 1
			rs += self.expectation(label)
		return (rs, count)

	''' compute expectation for prediction 
	exp = (mu * loss(distance) + ((1-mu) * loss(distance)) '''
	def expectation(self, info):
		mu = info['mu']  # mu is P(label) == 1
		dij = info['distance']
		fn = self.model.criterion
		with torch.no_grad():
			pos, neg = torch.Tensor([1]), torch.Tensor([-1])
			dij = torch.Tensor([dij])
		exp = (mu * fn(pos, dij)) + ((1-mu) * fn(neg, dij)) # is Tensor([exp])
		return exp[0].detach().numpy().item()
	
	def average_results(self, pool_res):
		obs = np.array([res['observed'] for res in pool_res])
		obs = np.mean(obs)
		res = pool_res[0].copy()
		res['observed'] = obs
		return res