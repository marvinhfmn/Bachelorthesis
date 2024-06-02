#!/usr/bin/env python3
# import numpy as np
# import matplotlib.pyplot as plt
# import sqlite3 as sql
# import pandas as pd
from glob import glob
import os
from pathlib import Path
import torch
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

#old
all_files = glob("/home/wecapstor3/capn/capn106h/sim_databases/*/*/*.db")
corsika_files = glob("/home/wecapstor3/capn/capn106h/sim_databases/22615/*/*.db")
nugen_files = [file for file in all_files if file not in corsika_files]

# all_files = glob("/home/wecapstor3/capn/capn106h/l2_labeled/merged/*.db")
# corsika_files = glob("/home/wecapstor3/capn/capn106h/l2_labeled/merged/22615_merged.db")
# nugen_files = [file for file in all_files if file not in corsika_files]

#most recent databases
datasetpath = glob("/home/wecapstor3/capn/capn106h/l2_labeled/merged/*.db")
datasetpath.sort()
NuEfiles = datasetpath[0:3]
CorsikaSimfiles = [datasetpath[3]]
NuTaufiles = datasetpath[4:7]
NuMufiles = datasetpath[7:]


# scp marvinhfmn@data.icecube.wisc.edu:/data/ana/graphnet/l2_labeled/22633/0000000-0000999/Level2_NuTau_NuGenCCNC.022633.000029.db /home/saturn/capn/capn108h/sampleset.db

file0 = "/home/saturn/capn/capn108h/sampleset.db" #/data/ana/graphnet/l2_labeled/22633/0000000-0000999/Level2_NuTau_NuGenCCNC.022633.000029.db
file1 = "/home/saturn/capn/capn108h/sampleset2.db" #data/ana/graphnet/l2_labeled/22633/0000000-0000999/Level2_NuTau_NuGenCCNC.022633.000039.db
file2 = "/home/saturn/capn/capn108h/sampleset3.db" #data/ana/graphnet/l2_labeled/22633/0000000-0000999/Level2_NuTau_NuGenCCNC.022633.000111.db

file3 = "/home/saturn/capn/capn108h/l2_NuMu_sampleset.db" #/data/ana/graphnet/l2_labeled/22646/0000000-0000999/Level2_NuMu_NuGenCCNC.022646.000019.db


def main(training_parameter, name = 'GNN_DynEdge_sampledata'):
          

        # PercentileClusters
        #define feature names, properties that should be clustered on and the percentiles to cluster with
        #Should return a cluster for each feature that is not included in cluster_on
        feature_names = ['dom_x', 'dom_y', 'dom_z', 'dom_time', 'charge', 'rde', 'pmt_area', 'hlc']
        cluster_on = ['dom_x', 'dom_y', 'dom_z']
        percentiles = [10, 20, 50, 70, 90]

        percentile_clustering_instance = PercentileClusters(cluster_on=cluster_on, percentiles=percentiles, input_feature_names=feature_names)

        #TODO add clustering to node definiton of graph
        graph_definition = KNNGraph(detector = IceCube86(), 
                                #     node_definition=percentile_clustering_instance
                                )

        #Create Datasets for training, validation and testing
        training_dataset, validation_dataset, testing_dataset = Custom.CreateCustomDatasets(path=NuEfiles, graph_definition=graph_definition, features=feature_names, truth=training_parameter)

        #make Dataloaders
        dataloader_training_instance = DataLoader(training_dataset, batch_size=64)
        dataloader_validation_instance = DataLoader(validation_dataset, batch_size=64)
        dataloader_testing_instance = DataLoader(testing_dataset, batch_size=64)
        


        # dataloader_training_instance = utils.make_dataloader(db = trainingdatabase, graph_definition=graph_definition, pulsemaps=['InIceDSTPulses'], 
        #                                         features= feature_names, truth=[training_parameter], truth_table='truth',
        #                                         batch_size=64, shuffle=False, 
        #                                         selection=selection_indices_training
        #                                         )

        # dataloader_validation_instance = utils.make_dataloader(db = valdatabase, graph_definition=graph_definition, pulsemaps=['InIceDSTPulses'], 
        #                                         features= feature_names, truth=[training_parameter], truth_table='truth',
        #                                         batch_size=64, shuffle=False, 
        #                                         selection=selection_indices_validation
        #                                         )


        # dataloader_testing_instance = utils.make_dataloader(db = testdatabase, graph_definition=graph_definition, pulsemaps=['InIceDSTPulses'], 
        #                                         features= feature_names, truth=[training_parameter], truth_table='truth',
        #                                         batch_size=64, shuffle=False, 
        #                                         selection=selection_indices_testing
        #                                         )
        
        

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

        # build task
        task = EnergyReconstruction(target_labels = [training_parameter],
                                hidden_size=backbone.nb_outputs,
                                loss_function = LogCoshLoss(),
                                transform_prediction_and_target = lambda x: torch.log10(x),
                                transform_inference = lambda x: torch.pow(10,x),
                                )

        # instantiate model
        model = StandardModel(graph_definition = graph_definition,
                        backbone = backbone,
                        tasks = task)

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

        
        
        path = os.path.join(sub_folder, name)
        model.save_config(f'{path}.yml')
        model.save_state_dict(f'{path}.pth')

        return test_results, sub_folder


if __name__ == "__main__":
    
        training_parameter = 'first_vertex_energy'  
        results, sub_folder = main(training_parameter, 
                                   name='GNN_DynEdge_mergedNuE'
                                   )
    
        plotter.plot_result(results, sub_folder, reco=training_parameter)
        plotter.plot_losses(sub_folder)

