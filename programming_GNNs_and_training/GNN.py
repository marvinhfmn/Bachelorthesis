#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace
from typing import Sequence, Optional, List, Union, Tuple
import os
from glob import glob
from time import gmtime, strftime
from datetime import timedelta
from termcolor import colored
import yaml
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
# import warnings
# warnings.filterwarnings("ignore", message="Starting from v1.9.0, `tensorboardX` has been removed")
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from graphnet.models.graphs import KNNGraph
from graphnet.models.detector.icecube import IceCube86, CustomIceCube86
from graphnet.data.dataloader import DataLoader
from graphnet.models.graphs.nodes.nodes import PercentileClusters
from graphnet.models import StandardModel, Model
from graphnet.models.gnn import DynEdge
from graphnet.models.task.reconstruction import EnergyReconstruction
from graphnet.training.loss_functions import LogCoshLoss
from graphnet.utilities.logging import Logger
from graphnet.utilities.config import ModelConfig
from graphnet.training.callbacks import ProgressBar

import CustomAdditionsforGNNTraining as Custom
import Analysis

# Environment Configuration
#Setting environment variable to not run out of memory: avoid fragmentation between reserved and unallocated
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'   
#Clear the cache to free up memory reserved by PyTorch but not currently used
# torch.cuda.empty_cache()
#Trades in precision for performance This does not change the output dtype of float32 matrix multiplications, 
#it controls how the internal computation of the matrix multiplication is performed.
torch.set_float32_matmul_precision('medium')

def parse_args(argv: Optional[Sequence[str]] = None) -> Namespace:
        """
        Method to parse arguments as a command line interface.
        """
        parser = ArgumentParser()
        parser.add_argument('--config_path', type=str, help="Path to the config file")
        parser.add_argument('--resumefromckpt', type=bool, 
                            help="""Decides whether or not to resume training from a checkpoint. 
                            The path to this checkpoint can be specified in the config file, if not takes the ckpt of the last available run.""", 
                            default=False)
        parser.add_argument('--dataset_path', type=str, help="Path to the dataset files")
        return parser.parse_args(argv)

def read_yaml(config_path: str) -> dict:
    """
    Read a yaml file. Used to read the config file. 
    """
    if not config_path or not os.path.isfile(config_path):
        raise FileNotFoundError("Configuration file not found or not specified.")
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def save_yaml(data: dict, filename: str ='config.yml') -> None:
    """
    Save a yaml file.
    """
    with open(filename, 'w') as file:
        yaml.dump(data, file)

def save_params(params: dict, basefolder: str, filename: str = 'gnn_params') -> None:
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
            # Convert DataFrame to CSV string
        csv_string = df.to_csv(header=False, mode=mode, lineterminator=os.linesep)
    
        # Append separator manually
        csv_string += '--' * 20 + '\n'
        # Write to file
        with open(temp, mode) as f:
            f.write(csv_string)       

def create_sub_folder(timestamp: str, base_folder: str, logger: Logger) -> str:
        """
        Creates a sub-folder for the current run.
        """
        run_name = f"run_from_{timestamp}_UTC"
        sub_folder = os.path.join(base_folder, run_name)
        os.makedirs(sub_folder, exist_ok=True)
        logger.info(f"Subfolder created: {sub_folder}")
        return sub_folder

def get_lastrun_path(runfoldername: str = 'runs_and_saved_models') -> str:
        """
        Returns the most recent runfolder under the runfoldername directory, given that they are sorted and the most recent is at the bottom 
        """
        dir_path = os.path.realpath(os.path.join(os.path.dirname(__file__), runfoldername))
        temp = os.listdir(dir_path)
        temp.sort()
        directories = [os.path.join(dir_path, item) for item in temp if os.path.isdir(os.path.join(dir_path, item))]
        if len(directories) == 0:
               raise ("No previous runs found.")
        return directories[-1]

def get_model_config_path(sub_folder: str, logger: Logger) -> Optional[str]:
        patterns = [os.path.join(sub_folder, 'GNN_*.yml'), os.path.join(sub_folder, 'GNN_*.yaml')]
        files = []
        for pattern in patterns:
            files.extend(glob(pattern))
        if files:
                if len(files) > 1:
                        logger.info(f"More than one model config found that fits the criteria in {sub_folder}. Choosing..")
                return files[0]
        else:
               logger.warning(f"No model config found in {sub_folder}")
               return None

def get_datasetpaths(config: dict) -> List[str]:
    """
    Returns all relevant datasetpaths as a list from config depending on the specified flavor in config, and adds the corsika databases if required.
    """
    # Handling dataset paths based on flavors
    datasetpaths = []
    flavors = config.get('flavor', ['NuE'])
    for flavor in flavors:
        key = f"{flavor}_datasetpaths"
        if key in config:
            datasetpaths.extend(config.get(key))
    if config.get('use_corsika_bool', False):
        datasetpaths.extend(config.get('corsikasim_datasetpaths', []))
    return datasetpaths

def train_and_evaluate(
              config: dict,
              sub_folder: str,
              logger: Logger,
              resumefromckpt: bool = False,
              path_to_datasets: str = '',
              ) -> pd.DataFrame:
        """
        Reading in nessecary variables from the config file, creating Datasets and Dataloaders.
        Build and train a model, validate it and then apply a testing dataset, saving results.
           
        Args:
                config (Dict): Configuration dictionary.
                sub_folder (str): Subfolder where model configurations are stored and yaml files are saved to.
                logger (Logger): Logger instance for logging information.
                resumefromckpt (bool): Flag to resume from checkpoint. Defaults to False.
    
        Returns:
                pd.DataFrame: DataFrame containing the evaluation results.
        """
        #Get parameters from the config yaml file    
        TRAINING_PARAMETER = config.get('training_parameter', ['deposited_energy']) #Target label in task
        ADDED_ATTRIBUTES = config.get('addedattributes_trainval', []) #List of event-level columns in the input files that should be used added as attributes on the  graph objects.
        #ensure that training parameter and added attributes are lists
        
        if isinstance(ADDED_ATTRIBUTES, str):
                ADDED_ATTRIBUTES = [ADDED_ATTRIBUTES]

        #define name to save model under 
        name = f"GNN_{config.get('backbone')}_merged{'_'.join(config.get('flavor'))}"
        modelpath = os.path.join(sub_folder, name)

        model_config_path = get_model_config_path(sub_folder=sub_folder, logger=logger) if resumefromckpt else None
        if resumefromckpt and model_config_path:
        #        logger.info()
               # Load model configuration
               model_config = ModelConfig.load(model_config_path)

               # Initialize model with randomly initialized weights
               model = Model.from_config(model_config, trust=True)

               #Access graph definition
               graph_definition = model._graph_definition
               FEATURE_NAMES = model._graph_definition._input_feature_names

        else: 
                #Maybe add support custom IceCube detector class without unnecessary pmt area
                DETECTOR = None 

                if config.get('detector').lower() in ['icecube86', 'icecube']:
                        DETECTOR = CustomIceCube86()
                else: 
                        raise NotImplementedError("No supported detector found.")
                #todo Include other detectors
                
                FEATURE_NAMES = [*DETECTOR.feature_map()] 
                #accesses the feature names from the detector class but faster than: feature_names = list(DETECTOR.feature_map().keys())
                # feature_names = ['dom_x', 'dom_y', 'dom_z', 'dom_time', 'charge', 'rde', 'pmt_area', 'hlc'] #features for the IceCube86 detector and without pmt area for Custom

                BACKBONE_CLASS = None
                if config.get('backbone').lower() == 'dynedge':
                        BACKBONE_CLASS = DynEdge
                else: 
                        raise NotImplementedError("No supported detector found.")
                #todo Include other backbones

                percentile_clustering_instance = None
                if 'node_definition' in config.keys():
                        NODEDEF = config.get('node_definition')

                        if 'cluster_on' in NODEDEF.keys():
                                CLUSTER_ON = NODEDEF['cluster_on']
                                PERCENTILES =  NODEDEF['percentiles']
                                # PercentileClusters
                                #Should return a cluster for each feature that is not included in cluster_on
                                percentile_clustering_instance = PercentileClusters(cluster_on=CLUSTER_ON, percentiles=PERCENTILES, input_feature_names=FEATURE_NAMES)

                graph_definition = KNNGraph(detector = DETECTOR, 
                                        node_definition=percentile_clustering_instance
                                        )

                # Select backbone
                # backbone_args = ...
                
                BACKBONE = BACKBONE_CLASS(
                                nb_inputs = graph_definition.nb_outputs,
                                global_pooling_schemes=["min", "max", "mean"]
                )

                # build task    
                task = EnergyReconstruction(
                                        target_labels = TRAINING_PARAMETER,
                                        hidden_size=BACKBONE.nb_outputs,
                                        loss_function = LogCoshLoss(),
                                        transform_prediction_and_target = lambda x: torch.log(x),
                                        transform_inference = lambda x: torch.pow(10,x),
                )

                # instantiate model
                model = StandardModel(
                                graph_definition = graph_definition,
                                backbone = BACKBONE,
                                tasks = task,
                                optimizer_kwargs = config.get('optimizer_kwargs'),
                                scheduler_class = ReduceLROnPlateau,
                                scheduler_kwargs = config.get('scheduler_kwargs'),
                                scheduler_config = config.get('scheduler_config'),
                )
                
                #save model configuration before training so it is accessible even if training doesn't fully complete
                model.save_config(f'{modelpath}.yml')

        MAX_EPOCHS = config.get('max_epochs', 10)
        # BATCH_SIZE = config.get('batch_size', 8)
        # NUM_WORKERS = config.get('num_workers', 32)
        
        EARLY_STOPPING_PATIENCE = config.get('early_stopping_patience', -1)
        ACCUMULATE_GRAD_BATCHES = config.get('accumulate_grad_batches', -1)
        if path_to_datasets:
                training_dataset, validation_dataset, testing_dataset = torch.load(path_to_datasets)
        else:
                datasetpaths = get_datasetpaths(config=config)
                #Create Datasets for training, validation and testing 
                #random energy distribution between the created datasets (depends on the random state variable)
                if config.get('training_parameter_inDatabase', False):
                        TRUTH = TRAINING_PARAMETER + ADDED_ATTRIBUTES
                        training_dataset, validation_dataset, testing_dataset = Custom.CreateCustomDatasets_traininglabelinDataset(
                                                                                                path=datasetpaths, 
                                                                                                graph_definition=graph_definition, 
                                                                                                features=FEATURE_NAMES, 
                                                                                                training_target_label=TRAINING_PARAMETER,
                                                                                                truth=TRUTH,
                                                                                                random_state=config.get('random_state', 42),
                        )
                else: 
                        lowercase_trainparam = [x.lower() for x in TRAINING_PARAMETER]
                        if 'deposited_energy' in lowercase_trainparam or 'deposited energy' in lowercase_trainparam:
                                training_target_label = Custom.CustomLabel_depoEnergy()
                        
                        TRUTH = config.get('addedattributes_trainval')
                        TEST_TRUTH = config.get('addedattributes_test', None)
                        training_dataset, validation_dataset, testing_dataset = Custom.CreateCustomDatasets_CustomTrainingLabel(
                                                                                                        path=datasetpaths,
                                                                                                        graph_definition=graph_definition, 
                                                                                                        features=FEATURE_NAMES,
                                                                                                        training_target_label=training_target_label,
                                                                                                        truth=TRUTH,
                                                                                                        classifications = config.get('classifications_to_train_on', [8, 9, 19, 20, 22, 23, 26, 27]),
                                                                                                        test_truth=TEST_TRUTH,
                                                                                                        config_for_dataset_datails= config.get('dataset_details', {}),
                                                                                                        root_database_lengths = config.get("database_lengths", {}), 
                                                                                                        test_size=config.get('dataset_details', {}).get('test_size'),
                                                                                                        random_state=config.get('random_state', 42),
                                                                                                        logger=logger,
                                )
                        logger.info(f"Detected training_target_label: {training_target_label}")
                
        logger.info(f"Length of training dataset: {len(training_dataset)}")
        
        #make Dataloaders
        dataloader_training_instance = DataLoader(training_dataset, **config.get('dataloader_config', {'batch_size': 8, 'num_workers': 32}), shuffle=True)
        dataloader_validation_instance = DataLoader(validation_dataset, **config.get('dataloader_config', {'batch_size': 8, 'num_workers': 32}))
        dataloader_testing_instance = DataLoader(testing_dataset, **config.get('dataloader_config', {'batch_size': 8, 'num_workers': 32}))

        logger.info(f"Size of training dataloader: {len(dataloader_training_instance)}")
        #Loss Logger
        n=0
        while os.path.isdir(sub_folder + f"/losses/version_{n}"):
                n += 1
        metrics_logger = CSVLogger(save_dir=sub_folder, name="losses", version=f"version_{n}")

        # define callbacks
        callbacks = [ProgressBar(refresh_rate=10000)]
        callbacks.append(
                EarlyStopping(
                monitor = "val_loss",
                patience = EARLY_STOPPING_PATIENCE,
                )
        )
        callbacks.append(
                ModelCheckpoint(
                save_top_k = 1,
                monitor = "val_loss",
                mode = "min",
                every_n_epochs = 1,
                dirpath = sub_folder + f"/losses/version_{n}/checkpoints",
                filename=f"{model.backbone.__class__.__name__}"
                    + "-{epoch}-{val_loss:.2f}-{train_loss:.2f}",
                )
        )
        callbacks.append(
                ModelCheckpoint(
                save_top_k = 1,
                monitor = "step",
                mode = "max",
                train_time_interval = timedelta(hours=0, minutes=30),
                dirpath = sub_folder + "/in_between_checkpoint",
                filename = "{epoch}_{step}",
                )
        )

        #save additional parameters that aren't included in the model yml file and save them in a more 'globalised' csv file 
        #for parameter search convenience  
        params = {
        'Runfolder': sub_folder.split("/")[-1],
        'Num of max epochs': MAX_EPOCHS,
        'Training paramter': TRAINING_PARAMETER, 
        'List of other considered truth event labels': config.get('addedattributes_trainval') + config.get('addedattributes_test', None), 
        'Early stopping': EARLY_STOPPING_PATIENCE,
        'resumed from ckpt': resumefromckpt,
        }
        params.update(config.get('dataloader_config', {'batch_size': 8, 'num_workers': 32}))
        BASEFOLDER = config.get('basefolder')
        save_params(params=params, basefolder=BASEFOLDER)

        #define some keyword arguments for training the model
        fit_kwargs = {
                'train_dataloader': dataloader_training_instance,
                'val_dataloader': dataloader_validation_instance,
                'distribution_strategy': "ddp",
                'logger': metrics_logger,
                'max_epochs': MAX_EPOCHS,
                'callbacks': callbacks,
                'precision': '32',
                'gpus': config.get('gpus', [0]) #try -1 instead of [0] to use all available 
        }
        
        if (EARLY_STOPPING_PATIENCE != -1):
                fit_kwargs['early_stopping_patience'] = EARLY_STOPPING_PATIENCE
        if (ACCUMULATE_GRAD_BATCHES != -1):
                fit_kwargs['accumulate_grad_batches'] = ACCUMULATE_GRAD_BATCHES
        if resumefromckpt:
                ckpt_path = config.get('ckpt_path')
                default_ckpt_paths = None
                if not ckpt_path:
                        # Only compute the default checkpoint path if not provided in the config
                        default_ckpt_paths = glob(os.path.join(sub_folder, 'losses/version_*/checkpoints', '*'))
                        # default_ckpt_paths = glob(os.path.join(sub_folder, 'epoch_checkpoints', '*'))
                        logger.info(f"found default_ckpt_paths: {default_ckpt_paths}")

                if default_ckpt_paths:
                        default_ckpt_paths.sort()
                        ckpt_path = default_ckpt_paths[-1]

                if ckpt_path:
                        logger.info(f"Resuming from checkpoint path: {ckpt_path}")
                        fit_kwargs['ckpt_path'] = ckpt_path
                else:
                        logger.warning("No checkpoint path found to resume from.")

        #train the model
        model.fit(**fit_kwargs)

        #save the model weights and biases after training
        #load weights from checkpoint if that training was interrupted before finishing the model.fit step
        model.save_state_dict(f'{modelpath}.pth')

        #get predictions as a pd.DataFrame
        test_results = model.predict_as_dataframe(dataloader = dataloader_testing_instance,
                                                additional_attributes = model.target_labels + config.get('addedattributes_test', ['classification', 'cosmic_primary_type']),
                                                gpus = [0]) 
          
        logger.info(f"Prediction finished. Writing to hdf5 file...")

        #save test results in hdf5 file
        test_results.to_hdf(os.path.join(sub_folder, 'test_results.h5'), key='test_results', mode='w')

        return test_results

def PlottingRoutine(
              config: dict, 
              logger: Logger,
              results: pd.DataFrame = None, 
              subfolder: str = None, 
              target_label: Union[str, List[str]] = None
              ):
        """
        Function to handle all the plotting. Mainly relies on the methods defined in Analysis.py
        """
        if isinstance(target_label, str):
                target_label = [target_label]
        
        if subfolder is None:
                raise FileNotFoundError("No subfolder found. Please make sure there is a folder available for loading test results and/or storing Plots.")
        
        os.makedirs(os.path.join(subfolder, 'plots'), exist_ok =True)

        # Check if metrics file is created
        metrics_path = os.path.join(subfolder, "losses/version_0/metrics.csv")
        if os.path.exists(metrics_path):
                print(f"Metrics file created at: {metrics_path}")
                Analysis.plot_lossesandlearningrate(subfolder)
        else:
                message = colored("ERROR", "red") + f": Metrics file not found under {subfolder}."
                logger.info(message)

        #if no result dataframe is handed over, load the dataframe from the subfolder
        if results is None:
               results = Analysis.loadresults(subfolder=subfolder)

        #All the other plotting calls
        Analysis.plot_resultsashisto(results, subfolder=subfolder, target_label=target_label, backbone=config.get('backbone', 'DynEdge'))
        Analysis.plotEtruevsEreco(results, subfolder=subfolder , normalise=['E_true', 'E_reco', 'nonormalisation'])
        Analysis.plotIQRvsEtrue(results, subfolder=subfolder)
        for i in config.get('classifications_to_train_on'):
                Analysis.plotIQR(dataframe=results, savefolder=subfolder, plot_type='energy', classification=i)
                Analysis.plotIQR(dataframe=results, savefolder=subfolder, plot_type='r', classification=i)
                Analysis.plotIQR(dataframe=results, savefolder=subfolder, plot_type='z', classification=i)

def get_config_and_sub_folder(
              args: Namespace, 
              resumefromckpt: bool,
              logger: Logger,
              is_main_process: bool = True, 
              foldername_allruns: str = 'runs_and_saved_models',
              ) -> Tuple[dict, str]:
        """
        Method to handle some calls that are misplaced in the main method. Determines the config path and sets up the sub folder if necessary.
        Returns the config_path and the sub_folder of the run
        """
        if args.config_path:
                config_path = args.config_path
                logger.info(f"Using config_path: {config_path} as specified.")
        elif resumefromckpt:
                config_path = os.path.join(get_lastrun_path(foldername_allruns), 'config.yaml')
                logger.info(f"Since the resumefromckpt flag has been passed, the config file of the last available run in the subfolder will be used: {config_path}")
        else:
                config_path = os.path.realpath(os.path.join(os.path.dirname(__file__), "config.yaml"))
                logger.info(f"Using the standard config at {config_path}, since neither config_path nor resumefromckpt was specified.")

        config = read_yaml(config_path)
        
        if resumefromckpt:
                if config.get('ckpt_path') and config.get('ckpt_path') not in {'last', 'best'}:
                        sub_folder = get_lastrun_path(foldername_allruns)
                else:
                        sub_folder = get_lastrun_path(foldername_allruns)  
                logger.info(f"Request to resume from checkpoint accepted, using subfolder: {sub_folder} to save files to.")
        else:
                if is_main_process:
                        base_folder = config.get("basefolder")
                        timestamp = strftime("%Y.%m.%d_%H:%M:%S", gmtime())
                        sub_folder = create_sub_folder(timestamp=timestamp, base_folder=base_folder, logger=logger)                        
                        # Save the YAML config file into subfolder if not resuming from checkpoint
                        save_yaml(config, os.path.join(sub_folder, 'config.yaml'))
                else:
                       sub_folder = get_lastrun_path(foldername_allruns)
                       logger.info("This is not the main process. sub_folder has been initialize to the last available subfolder in hopes that it corresponds to the one created during the main process.")

        return config, sub_folder

def main(argv: Optional[Sequence[str]] = None):
        logger = Logger()
        args = parse_args(argv)
        resumefromckpt = args.resumefromckpt
        path_to_datasets = args.dataset_path

        # Check if the current process is the main process for multi gpu training
        is_main_process = (os.environ.get("LOCAL_RANK", "0") == "0")
        localrank = os.getenv("LOCAL_RANK")
        print(f"LOCAL_RANK: {localrank}")
        
        config, sub_folder = get_config_and_sub_folder(args, resumefromckpt, logger=logger, is_main_process=is_main_process)        
        
        results = train_and_evaluate(config, sub_folder, path_to_datasets=path_to_datasets, resumefromckpt=resumefromckpt, logger=logger)
        if is_main_process:
                PlottingRoutine(config=config, results=results, subfolder=sub_folder, target_label=config.get('training_parameter', ['first_vertex_energy']), logger=logger)

if __name__ == "__main__":
        main()    

#maybe add pyyaml constructors (yaml.add_constructor) for Detector and backbones  