#!/usr/bin/env python3
# import numpy as np
# import matplotlib.pyplot as plt
# import sqlite3 as sql
# import pandas as pd
from glob import glob
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' #Setting environment variable to not run out of memory         
from pathlib import Path
import torch
torch.cuda.empty_cache() #Clear the cache to free up memory reserved by PyTorch but not currently used
from torch.optim import Adam
from torch.optim.optimizer import ParamsT
torch.set_float32_matmul_precision('medium') 
#Trades in precision for performance This does not change the output dtype of float32 matrix multiplications, it controls how the internal computation of the matrix multiplication is performed.

from graphnet.models.graphs import KNNGraph
from graphnet.models.detector.icecube import IceCube86
import graphnet.training.utils as utils
from graphnet.data.dataloader import DataLoader
from graphnet.models.graphs.nodes.nodes import PercentileClusters
from graphnet.models import StandardModel
from graphnet.models.gnn import DynEdge
from graphnet.models.task.reconstruction import EnergyReconstruction
from graphnet.training.loss_functions import LogCoshLoss
from pytorch_lightning.loggers import CSVLogger
from time import gmtime, strftime
# from termcolor import colored 

import plotter
import CustomDataLoaderandDataset as Custom

#most recent databases
datasetpath = glob("/home/wecapstor3/capn/capn106h/l2_labeled/merged/*.db")
datasetpath.sort()
NuEfiles = datasetpath[0:3]
CorsikaSimfiles = [datasetpath[3]]
NuTaufiles = datasetpath[4:7]
NuMufiles = datasetpath[7:]


# scp marvinhfmn@data.icecube.wisc.edu:/data/ana/graphnet/l2_labeled/22633/0000000-0000999/Level2_NuTau_NuGenCCNC.022633.000029.db /home/saturn/capn/capn108h/sampleset.db

def main(training_parameter):

        # PercentileClusters
        #define feature names, properties that should be clustered on and the percentiles to cluster with
        #Should return a cluster for each feature that is not included in cluster_on
        feature_names = ['dom_x', 'dom_y', 'dom_z', 'dom_time', 'charge', 'rde', 'pmt_area', 'hlc']
        cluster_on = ['dom_x', 'dom_y', 'dom_z']
        percentiles = [10, 20, 50, 70, 90]

        percentile_clustering_instance = PercentileClusters(cluster_on=cluster_on, percentiles=percentiles, input_feature_names=feature_names)

        #Define graph and use percentile clustering
        graph_definition = KNNGraph(detector = IceCube86(), 
                                #     node_definition=percentile_clustering_instance
                                )

        #Create Datasets for training, validation and testing
        training_dataset, validation_dataset, testing_dataset = Custom.CreateCustomDatasets(path=NuEfiles, graph_definition=graph_definition, features=feature_names, truth=training_parameter)

        num_workers = 32
        batch_size = 16
        #make Dataloaders
        dataloader_training_instance = DataLoader(training_dataset, batch_size=batch_size, num_workers=num_workers)
        dataloader_validation_instance = DataLoader(validation_dataset, batch_size=batch_size, num_workers=num_workers)
        dataloader_testing_instance = DataLoader(testing_dataset, batch_size=batch_size, num_workers=num_workers)
        
        

        # define folder to save to
        folder = "/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_and_saved_models/"
        now = strftime("%d.%m.%Y_%H:%M:%S", gmtime())
        run_name = f"run_from_{now}_UTC"
        sub_folder = os.path.join(folder, run_name)
        os.makedirs(sub_folder, exist_ok=True)

        #Loss Logger
        loss_logger = CSVLogger(save_dir=sub_folder, name="losses", version=0)

        # Select backbone
        backbone = DynEdge(nb_inputs = graph_definition.nb_outputs,
                        global_pooling_schemes=["min", "max", "mean"])

        #Handle Entries in Database where training parameter is zero. would otherwise lead to infinite losses (transform target)
        def HandleZeros(x):
                if not torch.any(zero_mask): #if there are no 0 entries apply log10 directly
                        return torch.log10(x)
                else:
                        zero_mask = (x == 0) #Create mask for entries that are 0 
                        result = torch.empty_like(x) #Initialize the result tensor with the same shape as elements
                        result[zero_mask] = -7 #apply -7 to zero elements, which is equivalent to torch.log10(torch.tensor(1e-7)), where 1e-7 is an arbitrarily chosen value
                        result[~zero_mask] = torch.log10(x) #apply log10 to non-zero elements
                        return result

        # build task    
        task = EnergyReconstruction(target_labels = [training_parameter],
                                hidden_size=backbone.nb_outputs,
                                loss_function = LogCoshLoss(),
                                transform_prediction_and_target = HandleZeros,
                                #transform_prediction_and_target = lambda x: torch.log10(x+1e-7) if torch.any(x) == 0 else torch.log10(x),
                                transform_inference = lambda x: torch.pow(10,x),
                                )

        # instantiate model
        model = StandardModel(graph_definition = graph_definition,
                        backbone = backbone,
                        tasks = task,
                        optimizer_kwargs = {'lr' : 1e-5}
                        )

        #train model
        model.fit(train_dataloader = dataloader_training_instance,
                val_dataloader = dataloader_validation_instance,
                distribution_strategy = "ddp_notebook",
                logger = loss_logger, 
                max_epochs=5, 
                # early_stopping_patience=2, 
                gpus = [0])

        test_results = model.predict_as_dataframe(dataloader = dataloader_testing_instance,
                                                additional_attributes = [training_parameter],
                                                gpus = [0])    

        
         
        name = 'GNN_DynEdge_mergedNuE'
        modelpath = os.path.join(sub_folder, name)
        model.save_config(f'{modelpath}.yml')
        model.save_state_dict(f'{modelpath}.pth')

        return test_results, sub_folder


if __name__ == "__main__":
    
        training_parameter = 'first_vertex_energy'  
        results, sub_folder = main(training_parameter)
    
        plotter.plot_result(results, sub_folder, reco=training_parameter)
        plotter.plot_losses(sub_folder)

