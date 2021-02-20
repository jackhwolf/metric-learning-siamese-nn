import json
import os
import asyncio
from dask.distributed import Scheduler, Worker, Client
from contextlib import AsyncExitStack
import libtmux
import pyyaml
from experiment import ExperimentPool, InterpolationExperiment, ExcessRiskExperiment

class Manager:

    def __init__(self, workers=2, session_name='metric-learning'):
        self.workers = workers
        # self.session = libtmux.Server().new_session(session_name, kill_session=True)
        # self.session.attach_session()
        self.pool = ExperimentPool('5', InterpolationExperiment, '20', '3', '100', modelargs={'epochs': '1'})
        self.results = []

    async def distributed_run(self):
        async with Scheduler() as sched:
            async with AsyncExitStack() as stack:
                ws = [await stack.enter_async_context(Worker(sched.address)) for i in range(self.workers)]
                async with Client(sched.address, asynchronous=True) as client:
                    futures = []
                    for i in range(self.pool.n):
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
    m = Manager()
    f = m.distributed_run
    out = asyncio.get_event_loop().run_until_complete(f())
    print(out)
