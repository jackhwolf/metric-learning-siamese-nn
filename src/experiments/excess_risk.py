import numpy as np
from time import time
import torch
from experiments.base import ExperimentBase

''' experiment to determine number of triplets needed for relative excess
risk between model and user/data truth to be less than 0.1 '''
class ExcessRiskExperiment(ExperimentBase):

	def __init__(self, P=10, D=3, N=5, eid=1, cer_threshold=0.1, initial_sample=0.0025, modelargs={}):
		super().__init__(P, D, N, eid, 'excess_risk', modelargs)
		self.cer_threshold = cer_threshold
		self.initial_sample = float(initial_sample)
		self.observed = 0
		self.risk_star = self.calc_risk_star()
		self.risk_hat = None

	def run(self):
		count = self.learn_initial()
		self.observed = 0
		stop = False
		while not stop:
			for pi, pj, label in self.data.iterate_triplets():
				loss = self.model.learn_triplet(pi, pj, label['noisy'])
				self.curr_excess_risk = self.relative_excess_risk()
				self.observed += 1
				print(f"eid: {self.eid}, observed: {self.observed}, loss: {loss}, c.e.r: {self.curr_excess_risk}")
				if self.curr_excess_risk < self.cer_threshold:
					stop = True
					break
		out = {'timestamp': str(time())}
		out['observed'] = self.observed
		out['initial_sample_frac'] = self.initial_sample
		out['initial_sample_count'] = count
		out['parameters'] = self.describe()
		return out

	def learn_initial(self):
		count = 0
		for pi, pj, label in self.data.iterate_triplets():
			if np.random.rand() > self.initial_sample:
				continue
			closs = self.model.learn_triplet(pi, pj, label['noisy'])
			count += 1
			print(f"eid: {self.eid}, learned: {count}, loss: {closs}")
		return count

	''' compute relative excess risk (R(*) - R(^)) / R(^) '''
	def relative_excess_risk(self):
		self.risk_hat = self.calc_risk_hat()
		rs, rh = self.risk_star[0], self.risk_hat[0]
		return (-1*(rh - rs)) / rs

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
	exp = (mu * loss(distance) + ((1-mu) * loss(-distance)) '''
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
		res = pool_res[0].copy()
		for key in ['observed', 'initial_sample_count']:
			obs = np.array([r[key] for r in pool_res])
			obs = np.mean(obs)
			res[key] = obs
		return res
