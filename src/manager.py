import json
import os
import asyncio
from dask.distributed import Scheduler, Worker, Client
from contextlib import AsyncExitStack
import yaml
from experiments.base import ExperimentPool
from experiments.interpolation import InterpolationExperiment
from experiments.prediction import PredictionExperiment
from experiments.excess_risk import ExcessRiskExperiment

emap = {'InterpolationExperiment': InterpolationExperiment,
        'PredictionExperiment': PredictionExperiment,
        'ExcessRiskExperiment': ExcessRiskExperiment}

class ManagerInput:

    def __init__(self, fname):
        with open(fname) as f:
            self.content = yaml.load(f, Loader=yaml.FullLoader)
        self.content['experiment'] = emap[self.content['experiment']]
        
class Manager:

    def __init__(self, fname, workers):
        self.workers = workers
        self.pool = ExperimentPool(**ManagerInput(fname).content)
        self.results = []

    def distributed_run(self):
        async def f():
            async with Scheduler() as sched:
                async with AsyncExitStack() as stack:
                    ws = []
                    for i in range(self.workers):
                        ws.append(await stack.enter_async_context(Worker(sched.address)))
                    async with Client(sched.address, asynchronous=True) as client:
                        futures = []
                        for i in range(len(self.pool)):
                            futures.append(client.submit(self.pool[i].run))
                        result = await client.gather(futures)
                        result = self.pool[0].average_results(result)
                        self.save(result)
                        return result
        return asyncio.get_event_loop().run_until_complete(f())

    def save(self, results):
        rp = self.pool[0].resultspath
        curr = []
        if os.path.exists(rp):
            with open(rp, 'r') as fp:
                curr = json.loads(fp.read())
        else:
            open(rp, 'a').close()
        if isinstance(results, list):
            curr.extend(results)
        else:
            curr.append(results)
        with open(rp, 'w') as fp:
            fp.write(json.dumps(curr))

if __name__ == '__main__':
    import sys

    m = Manager(sys.argv[1], int(sys.argv[2]))
    print(m.distributed_run())
