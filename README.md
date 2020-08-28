![CRADL logo](img/cradl_logo_small.png)
# Concurrent Relaxation through Accelerated Deep Learning (CRADL)

The CRADL proxy application captured performance metrics during inference on data from multiphysics codes, specifically ALE hydrodynamics codes.

## Building the Environment

Install the third party libraries with:

```
bash INSTALL
```

## Simple Run

First allocate an interactive GPU compute node on your HPC platform or operate locally if GPUs are available on your local system.
 
CRADL allows for a number of run configurations, most of which have default settings. The minimum required by the user is to point to their directory where inference data is stored.

```
python cradl_benchmarking.py --inference_dir ./data
```
## Input Options

The full list of command line arguments, and their type, in CRADL is given below. All boolean options only require the option name to be given on the command line to enable them, the user does not need to explicitly write True or False.

**NOTE**:data_parallel and distributed_data are mutually exclusive. If a user enables both, CRADL will run in data_parallel mode only.

- device *string*
  - **default:** 'cuda'
  - Input options for device take the form of the device strings used in PyTorch. The options available are 'cpu', 'cuda' and 'cuda:[*i*]', where *i* is the index of a specific GPU device visible to the host CPU.

- apex *bool*
  - **default:** False
  - Enable Nvidia's APEX library for accelerated half-precsion optimizations. This library only works on Nvidia tensorcores, beginning with the Volta architecture generation of GPUs.
 
- cuda_bench *bool*
  - **default:** False
  - Enable the cudnn autotuner to optimize model performance based on inputs. If input data size does not change during execution, there may be a performance increase.

- inference_dir *string*
  - **default:** *None*
  - The user must specify a directory of data for CRADL to use during execution. Currently this data takes the form of a set of .npy files from a previously executed hydrodynamics simulation.
  - **NOTE:** Data in the .npy files requires a specific format that is (samples, mesh quality metrics, history size).
  - *samples:* This is the number of nodes in a set of simulation data.
  - *mesh quality metrics:* These are values corresponding to geometric information about the mesh that is condensed from zone centered values to node centered ones. Currently there are only five metrics used by CRADL.
  - *history size:* This is the number of timesteps in a captured timeseries for a single node. Currently CRADL performs inference on 10 timesteps at a time.

- batch_size *int*
  - **default:** 2000
  - The number of samples to stream to the GPU at a time using the PyTorch data loader. Best practice is to start with a small batch size, assess GPU memory utilization in the final output and increase as needed.

- data_percent *float*
  - **default:** 10
  - Select percentage of data to use from the pool in --inference_dir. This functionality is provided to allow users a means of scaling their input data size easily.

- verbose_gpu *bool*
  - **default:** False
  - Select whether or not to enable verbose GPU reporting at the end of execution. This functionality is provided by the GPUtil library. The standard reporting provides percentage of memory and compute utilization for each visible device. The verbose report provides explicit details on the amount of memory used and device type, in addition to information in the standard report.

- stream_sync *bool*
  - **default:** False
  - Select whether or not to perform a synchronization between loading data onto the GPU and performing inference. This will result in a difference for time spent performing inference.

- data_parallel *bool*
  - **default:** False
  - Select whether or not to enable PyTorch DataParallel. This mode should not be used with MPI, it is intended for a single CPU/ multi-GPU configuration.

- distributed_data *bool*
  - **default:** False
  - Select whether or not to enable PyTorch DistributedDataParallel. This mode can be safely used with MPI. Best practice is to match the number of MPI ranks with the number of available GPUs.

- cycles *int*
  - **default:** 10
  - Select the number of cycles for CRADL to iterate over the data selected with data_percent.

- pinned_memory *bool*
  - **default:** False
  - Select whether or not to use pinned memory to accelerate data transfer to the GPU. When pinned memory is enabled the movement of the model and data onto the GPU is changed to a non-blocking operation.

## Output Explanations
CRADL outputs a substantial amount of data for users to assess hardware performance. An explanation of each of the output values is below.

- Output Data Shape
  - This array takes the form of a 1D array with a length equal to the number of nodes in a batch for inference. The array is filled with bool values corresponding to whether or not a node should be recommended for relaxation.

- Time for Loading Trained Model Weights and Architecture
  - This is the time to load the stored machine learning model. The model is stored using TorchScript.

- Time for Constructing and Preprocessing Data Pool
  - This is the time needed to read in the user selected percentage of the data pool and initialize the PyTorch DataLoader construct.

- Time for Inference
  - This is the time required to move the data onto the accelerator, perform inference, and transfer the results back to the host device.

- Time for Batch Processing
  - This time is strictly for performing inference, with none of the overhead of creating the batches.

- Time for Inference per Cycle
  - This is the *Time for Inference* divided by the user selected number of total cycles. It is used for the slack time comparison.

- Inference time per Node per Cycle
  - This is the *Time for Inference per Cycle* divided by the total number of nodes inference is performed on.

- Data Throughput
  - This is the total amount of data that inference is performed on divided by the *Time for Inference*.

- Slacktime Coverage per Cycle **SCC**
  - This is the value from *Time for Inference per Cycle* divided by slacktime from ALE simulations multiplied by 100. It represents what percentage of the inference computation is exposed as a factor of the ALE simulation cycle slacktime. Values are located in the ale_data.json file.

- Slacktime Coverage per Cycle per Node **SCCN**
  - This is the value from *Time for Inference per Node per Cycle* divided by slacktime per cycle per node from ALE simulations multiplied by 100. It represents what percentage of the inference computation is exposed as a factor of the ALE simulation cycle slacktime divided by the number of nodes used in the ALE simulations. Values are located in the ale_data.json file.

## Requirements

CRADL requires Python 3.7. At this time, there are no plans to extend support for older versions.

### Current Dependencies

* [PyTorch](https://pytorch.org/) - The ML library of choice for this project.
* [GPUtil](https://github.com/anderskm/gputil) - Python package used to query GPU memory and workload.


#### Optional Dependencies

* [Apex/AMP](https://github.com/NVIDIA/apex) - Nvidia's Apex library allows for targeted use of 'tensor cores' on Turing architecture GPUs.

## Authors

* **Kris Zieb**

## Acknowledgements

* Alister Maguire
* Rob Blake
* Charles Doutriaux
* Katie Lewis
* Josh Kallman
* Allen Toreja
