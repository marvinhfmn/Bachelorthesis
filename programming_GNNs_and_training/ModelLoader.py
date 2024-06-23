#!/usr/bin/env python3
import os
from glob import glob
from time import gmtime, strftime
import pandas as pd
import torch
from argparse import ArgumentParser, Namespace
from typing import Sequence, Optional, List, Union
import yaml
from pytorch_lightning.loggers import CSVLogger
from graphnet.models.graphs import KNNGraph
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.graphs.nodes.nodes import PercentileClusters
from graphnet.models import StandardModel
from graphnet.models.gnn import DynEdge
from graphnet.models.task.reconstruction import EnergyReconstruction
from graphnet.models import StandardModel, Model
from graphnet.utilities.config import ModelConfig
from graphnet.utilities.logging import Logger
from graphnet.data.dataloader import DataLoader
from graphnet.training.loss_functions import LogCoshLoss
import CustomAdditionsforGNNTraining as Custom
import Analysis

# Environment Configuration
#Setting environment variable to not run out of memory: avoid fragmentation between reserved and unallocated
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'   
#Clear the cache to free up memory reserved by PyTorch but not currently used
torch.cuda.empty_cache()
#Trades in precision for performance This does not change the output dtype of float32 matrix multiplications, 
#it controls how the internal computation of the matrix multiplication is performed.
torch.set_float32_matmul_precision('medium')

PATH = "/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_and_saved_models(old_datamergedate20240526)"
TIMEOFRUN = "run_from_2024.06.20_15:20:54_UTC" 
FILENAME = "GNN_DynEdge_mergedNuEandNuTau"

NUM_WORKERS = 32
BATCH_SIZE = 12

#Paths
#most recent databases
datasetpath = glob("/home/wecapstor3/capn/capn106h/l2_labeled/merged/*.db")
datasetpath.sort()

NuE_datasetIDs = {'22614', '22613', '22612'}
NuEfiles = [path for path in datasetpath if (path.split("/")[-1]).split("_")[0] in NuE_datasetIDs]

NuMu_datasetIDs = {'22646', '22645', '22644'}
NuMufiles = [path for path in datasetpath if (path.split("/")[-1]).split("_")[0] in NuMu_datasetIDs]

NuTau_datasetIDs = {'22633', '22634', '22635'}
NuTaufiles = NuEfiles = [path for path in datasetpath if (path.split("/")[-1]).split("_")[0] in NuTau_datasetIDs]

CorsikaSimfiles = [datasetpath[3]]

def evaluateModel(model, path=PATH, timeofrun=TIMEOFRUN, filename=FILENAME):
    """
    If there is already a test result do nothing else create dataset and evaluate the model on this test dataset, save results
    """
    logger = Logger()
    if os.path.isfile(f"{path}/{timeofrun}/test_results.h5"):
        #maybe Logger statement
        return
    else:
        #TODO: Read config file in path/subfolder

        # Access the graph definition
        graph_definition = model._graph_definition
        # print("Graph Definition:", graph_definition)

        feature_names = model._graph_definition._input_feature_names

        # Access the training parameter
        training_parameter = model.target_labels[0]
        # print("Training Parameter:", training_parameter)

        # Construct dataloaders
        training_dataset, validation_dataset, testing_dataset = Custom.CreateCustomDatasets(path=NuEfiles, 
                                                                                            graph_definition=graph_definition, 
                                                                                            features=feature_names, 
                                                                                            truth=training_parameter
                                                                                            )


        dataloader_testing_instance = DataLoader(testing_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        
        test_results = model.predict_as_dataframe(dataloader = dataloader_testing_instance,
                                            additional_attributes = [training_parameter],
                                            gpus = [0])
        
        #save test results in hdf5 file
        try: 
            test_results.to_hdf(os.path.join(f"{path}/{timeofrun}", 'test_results.h5'), key='test_results', mode='w')
            logger.info("Succesfully evaluated the model. Results can be found under {}".format(os.path.join(f"{path}/{timeofrun}", 'test_results.h5')))
        except:
            logger.info("Saving test results unsuccesful.")

def LoadModel(path=PATH, timeofrun=TIMEOFRUN, filename=FILENAME, load_weights=True):
    """
    Load model from config and initialise weights if there is a '.pth' file to be found.
    """
    # Construct the full paths
    config_file_path = os.path.join(path, timeofrun, f"{filename}.yml")
    weight_file_path = os.path.join(path, timeofrun, f"{filename}.pth")

    # Load model configuration
    model_config = ModelConfig.load(config_file_path)
    
    # Initialize model with randomly initialized weights
    model = Model.from_config(model_config, trust=True)

    # Check if the weight file exists before loading
    if load_weights and os.path.isfile(weight_file_path):
        # Load the trained weights into the model
        model.load_state_dict(torch.load(weight_file_path))

    return model

#TODO: look at how checkpoints work and how to integrate them into the workflow 
def resumeGNNTrainingfromcheckpoint():
    """
    Using the ckpt_path argument (is internally passed to the pytorch 'Trainer') to resume training from a checkpoint. 
    using this approach is useful for continuing training from a specific point, including optimizer states, 
    learning rate schedules, and the current epoch.
    """
    subfolder = ''
    model_config = ModelConfig.load("model.yml")
    model = Model.from_config(model_config)  # With randomly initialised weights.
    model.load_from_checkpoint("checkpoint.ckpt")  # Now with trained weight.
    return

def GNNTraining():
    """
    Using the load_from_checkpoint method to instantiate a new model instance from a checkpoint outside 
    the context pytorch 'Trainer'.
    Useful for fine tuning and transfer learning. 
    """
    return 

def InspectModelParameters():
    model = LoadModel()
    print("Model parameters: ", model.parameters)

def main():
    model = LoadModel()
    # print(model.parameters)
    # detector = model._graph_definition._detector
    # print(detector)
    # print(dir(model._graph_definition._node_definition._trainer))
    # print(model._graph_definition._node_definition.parameters)
    # print(model.target_labels)

    print(dir(model))
    # evaluateModel(model)

if __name__ == "__main__":
    main()