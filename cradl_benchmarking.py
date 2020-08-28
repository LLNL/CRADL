import pandas as pd
import argparse
import numpy as np
import time
import GPUtil
import json

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

# ALE-friendly Torch Interfaces
from utils.DataTools import generateALEDataLoader
from utils.inference_execution import TorchTester

from mpi4py import MPI

comm       = MPI.COMM_WORLD
rank       = comm.Get_rank()
world_size = comm.Get_size()

start_total_execution = time.time()

parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="CRADL version 1.0")
parser.add_argument('--device', '-d', type=str, default='cuda',
                    help="Specify 'cpu', 'cuda', or 'cuda:0' with device id being the number")
parser.add_argument('--apex', action='store_true',
                    help="Enable Nvidia APEX library for tensor core GPU architecture")
parser.add_argument('--cuda_bench', action='store_true',
                    help="Enable cuda benchmark performance optimizer")
parser.add_argument('--inference_dir', type=str,
                    help="Path to data directory")
parser.add_argument('--batch_size', type=int, default=2000,
                    help="Size of data to offload to device") #change to user-friendly description
parser.add_argument('--cycles', type=int, default=10,
                    help="Number of cycles to distribute data.") #change to user-friendly description
parser.add_argument('--data_percent', type=float, default=10,
                    help="Percentage of data in directory to use")
parser.add_argument('--verbose_gpu', action='store_true',
                    help="Print very detailed GPU usage report at end of run")
parser.add_argument('--stream_sync', action='store_true',
                    help="Synchronize GPU before inference for different timing results")
parser.add_argument('--data_parallel', action='store_true',
                    help="Enable PyTorch DataParallel. Should only be used with a single MPI rank.")
parser.add_argument('--distributed_data', action='store_true',
                    help="Enable PyTorch DistributedDataParallel. Safe with multiple MPI ranks.")
parser.add_argument('--pinned_memory', action='store_true',
                    help="Enable the use of pinned memory to accelerate memory management operations. Also enables non-blocking data transfer.")


args = parser.parse_args()

inference_device  = args.device
inference_dir     = args.inference_dir
verbose           = args.verbose_gpu
stream_sync       = args.stream_sync
perc              = args.data_percent
apex              = args.apex
data_parallel     = args.data_parallel
distributed_data  = args.distributed_data
pinned_memory     = args.pinned_memory
total_cycles      = args.cycles
torch.device(inference_device)

#APEX is the Nvidia library for optimizing machine learning in half precision,
#it isn't required to run, and will through an error if not installed with
# "--apex True" as an input option.
 
if args.apex:
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Nvidia APEX installation required")

# The cudnn.benchmark optimizes data loading in batches of constant size
torch.backends.cudnn.benchmark = args.cuda_bench
torch.nn.Module.dump_patches = True

if rank == 0:
   print("Loading Model")

with open("ale_data.json", "r") as read_file:
   ale_data = json.load(read_file)

# Experimental data from profiled hydrocode runs.

shocktube_slack = ale_data['shocktube_cycle']
shocktube_nodes = ale_data['shocktube_nodes']
hohlraum_slack  = ale_data['hohlraum_cycle']
hohlraum_nodes  = ale_data['hohlraum_nodes']

#Load stored model using JIT to capture full model architecture
start_model_load_transfer = time.time()
net = torch.jit.load("./Conv1Dv2.pth")
time_model_load_transfer = time.time() - start_model_load_transfer

if __name__ == '__main__':
    inference_f = []
    if rank == 0:
        print("Loading Data")

    start_data_load_batch = time.time()
    label_type     = np.long
    is_time_series = True
    inf_data, data_loader, samples, data_size = generateALEDataLoader(inference_dir,
                                                                      inference_f,
                                                                      perc/100,
                                                                      args.batch_size,
                                                                      label_type,
                                                                      is_time_series,
                                                                      pinned_memory,
                                                                      rank)
    time_data_load_batch = time.time() - start_data_load_batch

    model_data = {}
    model_data["network"] = net
    
    #TorchTester initializes all the values required for inference. This operation moves the model onto the accelerator of choice.
    inference_set = TorchTester(model_data,
                                data_loader,
                                inference_device,
                                data_parallel,
                                distributed_data,
                                pinned_memory)
    
    #Within this loop inference is performed each cycle on the total data specified by the user's data_percent input. This is not how true inline inference in a multiphysics code
    cumulative_batch_timer    = 0
    cumulative_overhead_timer = 0 
    for i in range(total_cycles):
       batch_timer, overhead_timer = inference_set.fullEval(stream_sync)
       cumulative_batch_timer     += batch_timer
       cumulative_overhead_timer  += overhead_timer 
    time_inference = cumulative_overhead_timer 
    time_inference = cumulative_batch_timer 
    
    time_total_execution = time.time() - start_total_execution
    
    inference_time_per_cycle           = time_inference / total_cycles
    inference_time_per_cycle_per_node  = inference_time_per_cycle / samples
    total_data                         = total_cycles*samples*5*10*data_size / 1e6
    # Create final report for performance details and slack time 
    # comparisons.
    if rank == 0:
        print("=======================")
        print("CRADL Execution Details")
        print("=======================")
        if "cuda" in args.device.lower():
            print("Device Used: " + str(torch.cuda.get_device_name()))
            print("APEX Library Used: " + str(args.apex))
            print("PyTorch Benchmarking Used: " +
                str(torch.backends.cudnn.benchmark))
            print("Pinned Memory Used: " +str(pinned_memory))
            print("\nTotal Cycles: {:d}".format(total_cycles))
            print("Total Memory: {:.4e} MB".format(total_cycles))
            print("\n~~~~~~~~~~~~~~~~~")
            print("Timing and Memory")
            print("~~~~~~~~~~~~~~~~~")
            if verbose is True:
                print("\nVerbose GPU Utilization Report")
                GPUtil.showUtilization(all=True)
            else:
                print("\nStandard GPU Utilization Report")
                GPUtil.showUtilization()
            print("\n-----------------------")
        print("Time for Loading Trained Model Weights and Architecture: {:.4e} s".format(
            time_model_load_transfer)) 
        print("Time for Constructing and Preprocessing Data Pool:       {:.4e} s".format(
            time_data_load_batch))
        print("Time for Inference:                                      {:.4e} s".format(
            time_inference)) 
        print("Time for Batch Processing:                               {:.4e} s".format(
            cumulative_batch_timer))
        print("Time for Inference per Cycle:                            {:.4e} s/cycles".format(
            inference_time_per_cycle))
        print("Inference Time per Node per Cycle:                       {:.4e} s/nodes".format(
            inference_time_per_cycle_per_node)) 
        print("Data Throughput:                                         {:.4e} MB/s".format(
            total_data / time_inference )) #5 and 10 are shapes from the data repo packaged with CRADL.
        print("-----------------------")

    slack_fraction_shocktube           = inference_time_per_cycle / shocktube_slack  
    slack_fraction_per_cycle_shocktube = inference_time_per_cycle_per_node / (shocktube_slack/(shocktube_nodes/16))
    slack_fraction_hohlraum            = inference_time_per_cycle / hohlraum_slack
    slack_fraction_per_cycle_hohlraum  = inference_time_per_cycle_per_node  / (hohlraum_slack/(hohlraum_nodes/36))

    if rank == 0:
        print("\n~~~~~~~~~~~~~~~~~~~~~~")
        print("Slack Time Comparisons")
        print("~~~~~~~~~~~~~~~~~~~~~~")
        print("NOTE: SCC is Slacktime Coverage per Cycle, SCCN is Slacktime Coverage per Cycle per Node.")
        print("Problem Type |   % SCC |   % SCCN")
        print("--------------------")
        print("ShockTube    | {:.2f}    |   {:.2f}".format(
               100 *
               slack_fraction_shocktube, (100 * slack_fraction_per_cycle_shocktube)))
        print("Hohlraum     |  {:.2f}   |   {:.2f}".format(
               100 *
               slack_fraction_hohlraum, (100 * slack_fraction_per_cycle_hohlraum)))

