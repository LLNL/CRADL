import numpy as np
import os
import random
import shutil
import sys
import torch
from torch.utils import data

def loadDataset(run_idx, data_dir, verbose=False):
    input_name = "inputs_%i.npy" % run_idx
    label_name = "labels_%i.npy" % run_idx

    input_pth  = os.path.join(data_dir, input_name)
    label_pth  = os.path.join(data_dir, label_name)

    if verbose:
        print("\nLoading")
        print("    inputs: %s" % input_pth)
        print("    labels: %s" % label_pth)

    inputs = np.load(input_pth)
    labels = np.load(label_pth)

    return inputs, labels


def validateRunFiles(df_numbers, data_dir):
    validated_runs = []

    for run_idx in df_numbers:
        input_name = "inputs_%i.npy" % run_idx
        input_pth  = os.path.join(data_dir, input_name)

        if os.path.exists(input_pth):
            validated_runs.append(run_idx)
        else:
            print("Missing run: %s" % str(run_idx))
            print("Skipping...")

    return validated_runs


def loadCombinedDatasets(df_numbers, data_dir, verbose=False):
    df_numbers = validateRunFiles(df_numbers, data_dir)

    if len(df_numbers) == 0:
        print("ERROR: unable to find any valid run files...")
        input_data = None
        label_data = None
    else:
        input_data, label_data = loadDataset(df_numbers[0], data_dir, verbose)
    
    for i in range(1, len(df_numbers)):
        run_idx        = df_numbers[i]
        inputs, labels = loadDataset(run_idx, data_dir, verbose)
        input_data     = np.concatenate((input_data, inputs))
        label_data     = np.concatenate((label_data, labels))

    return input_data, label_data


def getDataFileNumbers(f_path):
    file_numbers = []
    for f_name in os.listdir(f_path):
        if f_name.endswith(".npy") and f_name.startswith("labels"):
            file_numbers.append(int(f_name.split(".")[0].split("_")[-1]))
    return np.array(file_numbers)


class ALEDataset(data.Dataset):
    """
        A torch dataset for handling ALE simulation data.
    """

    def __init__(self, 
                 data_dir="", 
                 label_type=np.long, 
                 is_time_series=False,
                 df_numbers=[],
                 randomize=True,
                 perc=1.0,
                 rank=0):

        if data_dir == "":
            self.inputs = np.array([])
            self.labels = np.array([])
            return

        #
        # Load all of the files from the directory.
        #
        if df_numbers == []:
            df_numbers = getDataFileNumbers(data_dir)

        self.inputs, self.labels = loadCombinedDatasets(df_numbers, data_dir)

        if is_time_series:
            self.inputs = self.inputs.swapaxes(1, 2)

        self.labels = self.labels.astype(label_type)

        if is_time_series:
            self.num_samples   = self.inputs.shape[0]
            self.num_timesteps = self.inputs.shape[2]
            self.num_features  = int(self.inputs.shape[1])
        else:
            self.num_samples   = self.inputs.shape[0]
            self.num_timesteps = 1
            self.num_features  = int(self.inputs.shape[1])

        if label_type == np.float32 or label_type == np.float64:
            # Regression. Not really a class, but this needs to be set.
            # Maybe we should just change this to "out_size" or something?
            self.num_classes = 1
        else:
            self.num_classes = self.labels.max()

        if perc < 1.0:
            self.num_samples = int(float(self.num_samples) * perc)
            self.inputs = self.inputs[:self.num_samples]
            self.labels = self.labels[:self.num_samples]

        if label_type == np.float32 or label_type == np.float64:
            self.num_positive_idxs = np.where(self.labels < 1.0)[0].size
            self.num_negative_idxs = np.where(self.labels == 1.0)[0].size
        else:
            self.num_positive_idxs = np.where(self.labels == 1)[0].size
            self.num_negative_idxs = np.where(self.labels == 0)[0].size
        if rank==0:
           print("\nNOTE: Data is formatted as (nodes, mesh quality metrics, history size)\n")
           print("Input ALE Data Shape:        %s" % str(self.inputs.shape))
           print("Output Data Shape:           %s" % str(self.labels.shape))
           
    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.labels[idx]

        return x, y

    def __len__(self): 
        return self.num_samples


def generateALEDataLoader(data_dir,
                          data_file_numbers,
                          keep_perc,
                          batch_size,
                          label_type,
                          is_time_series,
                          pin_memory,
                          rank):

    ad = ALEDataset(data_dir       = data_dir,
                    label_type     = label_type,
                    is_time_series = is_time_series,
                    df_numbers     = data_file_numbers,
                    perc           = keep_perc,
                    rank           = rank)

    ad_loader = torch.utils.data.DataLoader(ad, 
                                            batch_size = batch_size,
                                            shuffle    = True,
                                            pin_memory = pin_memory)
    num_samples = ad.num_samples
    data_size = ad.inputs.itemsize
    return ad, ad_loader, num_samples, data_size
