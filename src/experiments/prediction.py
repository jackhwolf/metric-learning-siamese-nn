import numpy as np
from time import time
from experiments.base import ExperimentBase

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
