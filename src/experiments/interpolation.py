import numpy as np
from time import time
from distributed import worker_client
from experiments.base import ExperimentBase

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
		self.starting_epochs = self.model.epochs
		self.has_interpolated = False
		self.interpolated_epochs = -1

	def run(self):
		self.choose_sample()
		futures = []
		epochs = self.starting_epochs
		with worker_client() as wc:
			for r in range(self.mr):
				futures.append(wc.submit(self.interpolating_on_sample, epochs))
				epochs = int(epochs * (1 + self.u))
				self.cr += 1 # doesnt mean anything w dask
			futures = wc.gather(futures)
		for future in futures:
			intd, epochs = future
			if intd:
				self.has_interpolated = True
				self.interpolated_epochs = epochs
				break
		return self.describe()

	def interpolating_on_sample(self, epochs):
		losses = []
		l = len(self.triplets)
		for i, triplet in enumerate(self.triplets):
			loss = self.model.learn_triplet(*triplet, epochs=epochs)
			if i % 10:
				print(f"eid={self.eid}, epochs={epochs}, round={self.cr}/{self.mr}, triplet={i}/{l}: loss={loss}")
			losses.append(loss)
		losses = np.array(losses)
		if np.mean(losses) <= self.loss_threshold:
			return (True, epochs)
		return (False, epochs)

	def choose_sample(self):
		triplets = []
		for pi, pj, label in self.data.iterate_triplets():
			if np.random.uniform(low=0, high=1) < self.s:
				triplets.append([pi, pj, label['noisy']])
		if len(triplets) == 0:
			triplets.append([pi, pj, label['noisy']])
		self.triplets = triplets
	
	def describe(self):
		out = {} 
		out['timestamp'] = str(time())
		out['interpolated'] = self.has_interpolated
		out['interpolated_epochs'] = self.interpolated_epochs
		out['starting_epochs'] = self.starting_epochs
		out['parameters'] = super().describe()
		out['parameters']['loss_threshold'] = self.loss_threshold
		out['parameters']['s'] = self.s
		out['parameters']['u'] = self.u
		out['parameters']['r'] = self.mr
		return out
