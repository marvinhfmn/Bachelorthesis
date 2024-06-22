#!/usr/bin/env python3
import os
from glob import glob
from time import gmtime, strftime
# import csv
import pandas as pd
import torch
from argparse import ArgumentParser, Namespace
from typing import Sequence, Optional, List, Union
import yaml
from graphnet.models.graphs import KNNGraph
from graphnet.models.detector.icecube import IceCube86
from graphnet.data.dataloader import DataLoader
from graphnet.models.graphs.nodes.nodes import PercentileClusters
from graphnet.models import StandardModel
from graphnet.models.gnn import DynEdge
from graphnet.models.task.reconstruction import EnergyReconstruction
from graphnet.training.loss_functions import LogCoshLoss
from pytorch_lightning.loggers import CSVLogger
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


def parse_args(argv: Optional[Sequence[str]] = None) -> Namespace:
        """
        Method to parse arguments as a command line interface.
        """
        parser = ArgumentParser()
        parser.add_argument('--config_path', type=str, help="Path to the config file")
        return parser.parse_args(argv)

def read_yaml(config_path):
    """
    Read a yaml file. Used to read the config file. 
    """
    if not config_path or not os.path.isfile(config_path):
        raise FileNotFoundError("Configuration file not found or not specified.")
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def save_yaml(data, filename: str ='config.yml'):
    with open(filename, 'w') as file:
        yaml.dump(data, file)

def get_datasetpaths(config):
    """
    Returns all relevant datasetpaths as a list from config depending on the specified flavor in config, and adds the corsika databases if required.
    """
    # Handling dataset paths based on flavors
    datasetpaths = []
    flavors = config['flavor']
    for flavor in flavors:
        key = f"{flavor}_datasetpaths"
        if key in config:
            datasetpaths.extend(config[key])
    if config.get('use_corsika_bool', False):
        datasetpaths.extend(config.get('corsikasim_datasetpaths', []))
    return datasetpaths

def create_sub_folder(timestamp: str, base_folder: str) -> str:
        """
        Creates a sub-folder for the current run.
        """
        run_name = f"run_from_{timestamp}_UTC"
        sub_folder = os.path.join(base_folder, run_name)
        os.makedirs(sub_folder, exist_ok=True)
        return sub_folder

def save_params(params: dict, basefolder: str, filename: str = 'gnn_params'):
        """
        Method to save some additional information about the run in a 'global' csv file. Convenient for searches for specific parameters.
        What is saved depends on the params attribute.
        """
        # Allows more dynamic adjustments in case the params schema differs from previous runs
        # Ensure the directory exists
        os.makedirs(basefolder, exist_ok=True)

        temp = os.path.join(basefolder, f"{filename}.csv")
        mode = 'a' if os.path.isfile(temp) else 'w'
        
        # Convert params to DataFrame and append separator
        df = pd.DataFrame.from_dict(data=params, orient='index')
        separator = pd.DataFrame([['--' * 20]])
        df = pd.concat([df, separator])
        
        # Save DataFrame to CSV
        df.to_csv(temp, header=False, mode=mode, lineterminator=os.linesep)

def train_and_evaluate(config, sub_folder: str) -> pd.DataFrame:
        """
        Reading in nessecary variables from the config file, creating Datasets and Dataloaders.
        Build and train a model, validate it and then apply a testing dataset, saving results.
        """
        #Get parameters from the config yaml file    
        DETECTOR = None  
        if config['detector'].lower() in ['icecube86', 'icecube']:
                DETECTOR = IceCube86()
        else: 
                raise NotImplementedError("No supported detector found.")
        #todo Include other detectors
        
        FEATURE_NAMES = [*DETECTOR.feature_map()] 
        #accesses the feature names from the detector class but faster than: feature_names = list(DETECTOR.feature_map().keys())
        # feature_names = ['dom_x', 'dom_y', 'dom_z', 'dom_time', 'charge', 'rde', 'pmt_area', 'hlc'] #features for the IceCube86 detector 

        BACKBONE_CLASS = None
        if config['backbone'].lower() == 'dynedge':
                BACKBONE_CLASS = DynEdge
        else: 
                raise NotImplementedError("No supported detector found.")
        #todo Include other backbones

        percentile_clustering_instance = None
        if 'node_definition' in config.keys():
                NODEDEF = config['node_definition']
                if 'cluster_on' in NODEDEF.keys():
                        CLUSTER_ON = NODEDEF['cluster_on']
                        PERCENTILES =  NODEDEF['percentiles']
                        # PercentileClusters
                        #Should return a cluster for each feature that is not included in cluster_on
                        percentile_clustering_instance = PercentileClusters(cluster_on=CLUSTER_ON, percentiles=PERCENTILES, input_feature_names=FEATURE_NAMES)

        MAX_EPOCHS = config['max_epochs']
        BATCH_SIZE = config['batch_size']
        NUM_WORKERS = config['num_workers']
        TRAINING_PARAMETER = config['training_parameter'] #Target label in task
        ADDED_ATTRIBUTES = config['addedattributes'] #List of event-level columns in the input files that should be used added as attributes on the  graph objects.
        #ensure that training parameter and added attributes are lists
        if isinstance(TRAINING_PARAMETER, str):
                TRAINING_PARAMETER = [TRAINING_PARAMETER]
        if isinstance(ADDED_ATTRIBUTES, str):
                ADDED_ATTRIBUTES = [ADDED_ATTRIBUTES]

        TRUTH = TRAINING_PARAMETER + ADDED_ATTRIBUTES

        LEARNING_RATE = config.get("lr", 0.001)
        if "lr" not in config:
                print(f"No learning rate specified in config. Using standard value: {LEARNING_RATE}.")
        
        EARLY_STOPPING_PATIENCE = config['early_stopping_patience']
        ACCUMULATE_GRAD_BATCHES = config['accumulate_grad_batches']
        
        datasetpaths = get_datasetpaths(config=config)

        #Define graph and use percentile clustering
        graph_definition = KNNGraph(detector = DETECTOR, 
                                node_definition=percentile_clustering_instance
                                )

        #Create Datasets for training, validation and testing 
        #I am not quite sure about the energy distribution between the created datasets (depends on the random state variable)
        training_dataset, validation_dataset, testing_dataset = Custom.CreateCustomDatasets(path=datasetpaths, 
                                                                                        graph_definition=graph_definition, 
                                                                                        features=FEATURE_NAMES, 
                                                                                        training_target_label=TRAINING_PARAMETER,
                                                                                        truth=TRUTH,
                                                                                        random_state=config.get('random_state', 42)
                                                                                        )

        #make Dataloaders
        dataloader_training_instance = DataLoader(training_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        dataloader_validation_instance = DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        dataloader_testing_instance = DataLoader(testing_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        
        #Loss Logger
        loss_logger = CSVLogger(save_dir=sub_folder, name="losses", version=0)

        # Select backbone
        # backbone_args = ...
        
        BACKBONE = BACKBONE_CLASS(nb_inputs = graph_definition.nb_outputs,
                        global_pooling_schemes=["min", "max", "mean"])

        # build task    
        task = EnergyReconstruction(target_labels = TRAINING_PARAMETER,
                                hidden_size=BACKBONE.nb_outputs,
                                loss_function = LogCoshLoss(),
                                transform_prediction_and_target = lambda x: torch.log(x),
                                transform_inference = lambda x: torch.pow(10,x),
                                )

        # instantiate model
        model = StandardModel(graph_definition = graph_definition,
                        backbone = BACKBONE,
                        tasks = task,
                        optimizer_kwargs = {'lr': config.get('lr', 0.001)}
                        )
        
        #save model configuration before training so it is accessible even if training doesn't fully complete
        name = f"GNN_{config['backbone']}_merged{'_'.join(config['flavor'])}"
        modelpath = os.path.join(sub_folder, name)
        model.save_config(f'{modelpath}.yml')

        #save additional parameters that aren't included in the model yml file and save them in a more 'globalised' csv file 
        #for parameter search convenience  
        params = {
        'Runfolder': sub_folder.split("/")[-1],
        'Batch_Size': BATCH_SIZE,
        'Num of max epochs': MAX_EPOCHS,
        'Num of workers': NUM_WORKERS,
        'Training paramter': TRAINING_PARAMETER, 
        'List of other considered truth event labels': TRUTH, 
        'Early stopping': EARLY_STOPPING_PATIENCE,
        }
        
        BASEFOLDER = config['basefolder']
        save_params(params=params, basefolder=BASEFOLDER)

        #define some keyword arguments for training the model
        fit_kwargs = {
                'train_dataloader': dataloader_training_instance,
                'val_dataloader': dataloader_validation_instance,
                'distribution_strategy': "ddp_notebook",
                'logger': loss_logger,
                'max_epochs': MAX_EPOCHS,
                'precision': '32',
                'gpus': [0]
        }
        
        if (EARLY_STOPPING_PATIENCE != -1):
                fit_kwargs['early_stopping_patience'] = EARLY_STOPPING_PATIENCE
        if (ACCUMULATE_GRAD_BATCHES != -1):
                fit_kwargs['accumulate_grad_batches'] = ACCUMULATE_GRAD_BATCHES

        #train the model
        model.fit(**fit_kwargs)

        #save the model weights and biases after training
        #load weights from checkpoint if that Training was interrupted before finishing the model.fit step
        model.save_state_dict(f'{modelpath}.pth')

        #get predictions as a pd.DataFrame
        test_results = model.predict_as_dataframe(dataloader = dataloader_testing_instance,
                                                additional_attributes = TRUTH,
                                                gpus = [0])   

        #save test results in hdf5 file
        test_results.to_hdf(os.path.join(sub_folder, 'test_results.h5'), key='test_results', mode='w')

        return test_results

def PlottingRoutine(results: pd.DataFrame = None, subfolder: str = None, target_label: str = None):
        """
        Function to handle all the plotting. Mainly relies on the methods defined in Analysis.py
        """
        if subfolder is None:
                raise FileNotFoundError("No subfolder found. Please make sure there is a folder available for loading test results and/or storing Plots.")
               
        # Check if metrics file is created
        metrics_path = os.path.join(subfolder, "losses/version_0/metrics.csv")
        if os.path.exists(metrics_path):
                print(f"Metrics file created at: {metrics_path}")
        else:
                print("ERROR: Metrics file not found.")

        #if no result dataframe is handed over, load the dataframe from the subfolder
        if results is None:
               results = Analysis.loadresults(subfolder=subfolder)

        #All the Plotting calls
        Analysis.plot_lossesandlearningrate(subfolder)
        Analysis.plot_resultsashisto(results, subfolder=subfolder, target_label=target_label)
        Analysis.plotEtruevsEreco(results, subfolder=subfolder , normalise=['E_true', 'E_reco', 'nonormalisation'])
        # Analysis.plotEtruevsEreco(results, subfolder=subfolder, normalise='E_reco')
        Analysis.plotIQRvsEtrue(results, subfolder=subfolder)

def main(argv: Optional[Sequence[str]] = None):
        args = parse_args(argv)
        config = read_yaml(args.config_path)

        BASEFOLDER = config["basefolder"]
        timestamp = strftime("%Y.%m.%d_%H:%M:%S", gmtime())
        sub_folder = create_sub_folder(timestamp=timestamp, base_folder=BASEFOLDER)

        # Save the YAML config file into subfolder
        config_filename = os.path.join(sub_folder, 'config.yaml')
        save_yaml(config, config_filename)

        results = train_and_evaluate(config, sub_folder)
        PlottingRoutine(results=results, subfolder=sub_folder, target_label=config['training_parameter'])

if __name__ == "__main__":
        main()    
