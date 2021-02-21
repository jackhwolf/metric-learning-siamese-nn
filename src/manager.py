import json
import os
import asyncio
from dask.distributed import Scheduler, Worker, Client
from contextlib import AsyncExitStack
import yaml
from experiment import ExperimentPool, InterpolationExperiment, ExcessRiskExperiment

class ManagerInput:

    def __init__(self, fname):
        with open(fname) as f:
            self.content = yaml.load(f, Loader=yaml.FullLoader)
        if self.content['experiment'] == 'InterpolationExperiment':
            self.content['experiment'] = InterpolationExperiment
        else:
            self.content['experiment'] = ExcessRiskExperiment
        
class Manager:

    def __init__(self, fname, workers):
        self.workers = workers
        self.pool = ExperimentPool(**ManagerInput(fname).content)
        self.results = []

    async def distributed_run(self):
        async with Scheduler() as sched:
            async with AsyncExitStack() as stack:
                ws = [await stack.enter_async_context(Worker(sched.address)) for i in range(self.workers)]
                async with Client(sched.address, asynchronous=True) as client:
                    futures = []
                    for i in range(len(self.pool)):
                        futures.append(client.submit(self.pool[i].run))
                    result = await client.gather(futures)
                    self.save(result)
                    return result

    def save(self, results):
        rp = self.pool[0].resultspath
        curr = []
        if os.path.exists(rp):
            with open(rp, 'r') as fp:
                curr = json.loads(fp.read())
        else:
            open(rp, 'a').close()
        curr.extend(results)
        with open(rp, 'w') as fp:
            fp.write(json.dumps(curr))

if __name__ == '__main__':
    import sys

    m = Manager(sys.argv[1], int(sys.argv[2]))
    f = m.distributed_run
    out = asyncio.get_event_loop().run_until_complete(f())
    print(out)
