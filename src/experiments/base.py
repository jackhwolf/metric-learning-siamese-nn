import os
import numpy as np
from time import time
import torch
from distributed import worker_client
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
		self.pool = [experiment(eid=i, modelargs=self.margs, **self.eargs)
		                        for i in range(self.poolsize)]

	def __getitem__(self, i):
		return self.pool[i]

	def __len__(self):
		return self.poolsize
