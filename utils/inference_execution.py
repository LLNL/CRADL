import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.nn.init as init
import time

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank       = comm.Get_rank()
world_size = comm.Get_size()

#Inference Execution Classes
class TorchTester(object):

    def __init__(self,
                 model_data,
                 data_loader,
                 device,
                 data_parallel,
                 distributed_data,
                 pinned_memory):

        self.model            = model_data["network"] 
        self.data_loader      = data_loader
        self.device           = device
        self.data_parallel    = data_parallel
        self.distributed_data = distributed_data
        self.non_blocking     = pinned_memory
        self.model.to(self.device, non_blocking=self.non_blocking)

        if self.data_parallel:
           self.model = nn.DataParallel(self.model)
        elif self.distributed_data:
           torch.distributed.init_process_group(backend='nccl', 
                                                rank=rank, 
                                                world_size=world_size)
           self.model = DistributedDataParallel(self.model)




    def fullEval(self, stream_sync):

        self.model.eval()
        num_batches = len(self.data_loader)

        with torch.no_grad():
            
            start_overhead_timer = time.time()
            cumulative_batch_timer = 0

            for batch_idx, data in enumerate(self.data_loader):
                inputs, labels = data

                labels         = labels.reshape(labels.shape[0])
                
                
                start_batch_timer = time.time()

                inputs, labels = inputs.to(self.device, non_blocking = self.non_blocking), labels.to(self.device, non_blocking = self.non_blocking)
                if stream_sync:
                   torch.cuda.synchronize()
                outputs = self.model(inputs)
                outputs = outputs.to('cpu')
                
                time_batch = time.time() - start_batch_timer
                cumulative_batch_timer += time_batch

                last_batch = (batch_idx == (num_batches - 1))
            time_overhead = time.time() - start_overhead_timer
        return cumulative_batch_timer, time_overhead         
