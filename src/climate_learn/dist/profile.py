import gptl4py as gp

# import scorep.user as sp
from contextlib import contextmanager
import torch.distributed as dist
import os
import torch


class ProfileTimer:
    def __init__(self, device):
        self.device = device
        self.hist = dict()
        self.last = dict()
        gp.initialize()

    def begin(self, name):
        torch.cuda.synchronize(device=self.device)
        gp.start(name)

    def end(self, name):
        torch.cuda.synchronize(device=self.device)
        gp.stop(name)

        count, wallclock = gp.query_raw(name)
        if name not in self.hist:
            self.hist[name] = list()
            self.last[name] = 0.0
        self.hist[name].append((count, wallclock - self.last[name]))
        self.last[name] = wallclock

    def reset(self):
        gp.reset()
        self.hist = dict()
        self.last = dict()

    def dump(self):
        world_rank = int(os.getenv('SLURM_PROCID','0'))
        ## GPTL timer output
        output_dir = os.getenv("OUTPUT_DIR", "")
        gp.pr_file(os.path.join(output_dir, f"gp_timing.p{world_rank}"))
        gp.pr_summary_file(os.path.join(output_dir, f"gp_timing.summary"))

        with open(os.path.join(output_dir, f"gp_full.p{world_rank}"), "w") as f:
            f.write("rank,label,count,wallclock\n")
            for key, value_list in self.hist.items():
                for count, wallclock in value_list:
                    f.write(f"{world_rank},{key},{count},{wallclock}\n")

    def finalize(self):
        gp.finalize()
