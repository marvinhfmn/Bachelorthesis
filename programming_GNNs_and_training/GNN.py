#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace
from typing import Sequence, Optional, List, Union, Tuple
import os
from glob import glob
import re
from datetime import timedelta
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
# from graphnet.data.dataset import EnsembleDataset
from graphnet.models.graphs.nodes.nodes import PercentileClusters
from graphnet.models import StandardModel, Model
from graphnet.models.gnn import DynEdge
from graphnet.models.task.reconstruction import EnergyReconstruction
from graphnet.training.loss_functions import LogCoshLoss
from graphnet.utilities.logging import Logger
from graphnet.utilities.config import ModelConfig
from graphnet.training.callbacks import ProgressBar

import CustomAdditionsforGNNTraining as Custom
import utilities as utils
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
        parser.add_argument('--config_path', 
                        type=str, 
                        default="/home/saturn/capn/capn108h/programming_GNNs_and_training/config.yaml",
                        help="Path to the config file",
        )
        parser.add_argument('--resumefromckpt', 
                            default=False,
                            action="store_true",
                            help="""Decides whether or not to resume training from a checkpoint. 
                            The path to this checkpoint can be specified in the config file, 
                            if not takes the ckpt of the last available run.""", 
        )
        # parser.add_argument('--dataset_path', type=str, help="Path to the dataset files")
        parser.add_argument('--use_tmpdir',
                            default=False,
                            action="store_true",
                            help="""Whether or not to use the high bandwidth, 
                            low latency $TMPDIR of the gpu node for accessing the databases""", 
        )

        return parser.parse_args(argv)

def evaluate_GNN(
              model: Union[Model, StandardModel],
              dataloader_testing_instance: DataLoader,
              config: dict, 
              sub_folder: str,
              logger: Logger,
)-> pd.DataFrame:
        """
        Evaluate the (pretrained) Graph Neural Network model on the testing dataset and save the results.
        Args:
                model (Union[Model, StandardModel]): The trained model instance.
                dataloader_testing_instance (DataLoader): DataLoader instance for testing data.
                config (Dict[str, Union[str, List[str]]]): Configuration dictionary.
                sub_folder (str): Subfolder where results will be saved.
                logger (Logger): Logger instance for logging information.

        Returns:
                pd.DataFrame: DataFrame containing the evaluation results.
        """
        #get predictions as a pd.DataFrame
        test_results = model.predict_as_dataframe(dataloader = dataloader_testing_instance,
                                                additional_attributes = model.target_labels + config.get('addedattributes_test', ['classification', 'cosmic_primary_type']),
                                                gpus = [0]) 
          
        logger.info(f"Prediction finished. Writing to hdf5 file...")

        #save test results in hdf5 file
        test_results.to_hdf(os.path.join(sub_folder, 'test_results.h5'), key='test_results', mode='w')

        return test_results

def train_and_evaluate_GNN(
              config: dict,
              sub_folder: str,
              logger: Logger,
              resumefromckpt: bool = False,
              use_tmpdir: bool = True,
        #       path_to_datasets: str = '',
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

        model_config_path = utils.get_model_config_path(sub_folder=sub_folder, logger=logger) if resumefromckpt else None
        if resumefromckpt and model_config_path:
        #        logger.info()
               # Load model configuration
               model_config = ModelConfig.load(model_config_path)

               # Initialize model with randomly initialized weights (weights are loaded from the ckpt at training)
               model = Model.from_config(model_config, trust=True)

               #Access graph definition
               graph_definition = model._graph_definition
               FEATURE_NAMES = model._graph_definition._input_feature_names

        else: 
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

        lowercase_trainparam = [x.lower() for x in TRAINING_PARAMETER]
        if 'deposited_energy' in lowercase_trainparam or 'deposited energy' in lowercase_trainparam:
                training_target_label = Custom.CustomLabel_depoEnergy()
        else:
                training_target_label = TRAINING_PARAMETER

        training_dataset, validation_dataset, testing_dataset = utils.create_custom_datasets(
                config=config,
                graph_definition=graph_definition,
                feature_names=FEATURE_NAMES,
                training_target_label=training_target_label,
                logger=logger,
                use_tmpdir=use_tmpdir
        )

        logger.info(f"Length of training dataset: {len(training_dataset)}")
        
        #make Dataloaders
        dataloader_training_instance = DataLoader(training_dataset, **config.get('dataloader_config', {'batch_size': 8, 'num_workers': 16}), shuffle=True)
        dataloader_validation_instance = DataLoader(validation_dataset, **config.get('dataloader_config', {'batch_size': 8, 'num_workers': 16}))
        dataloader_testing_instance = DataLoader(testing_dataset, **config.get('dataloader_config', {'batch_size': 8, 'num_workers': 16}))

        logger.info(f"Size of training dataloader: {len(dataloader_training_instance)}")
        #Loss Logger
        n=0
        while os.path.isdir(sub_folder + f"/losses/version_{n}"):
                n += 1
        metrics_logger = CSVLogger(save_dir=sub_folder, name="losses", version=f"version_{n}")

        # define callbacks
        callbacks = [ProgressBar(refresh_rate=config.get('refresh_rate', 10000))]
        
        callbacks.append(
                ModelCheckpoint(
                save_top_k = 1,
                monitor = "val_loss",
                mode = "min",
                every_n_epochs = 1,
                dirpath = os.path.join(sub_folder, "checkpoints"),
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
        params.update({"Slurm job id": int(os.environ.get("SLURM_JOB_ID"))})
        BASEFOLDER = config.get('basefolder')
        utils.save_params(params=params, basefolder=BASEFOLDER)

        #define some keyword arguments for training the model
        fit_kwargs = {
                'train_dataloader': dataloader_training_instance,
                'val_dataloader': dataloader_validation_instance,
                'distribution_strategy': config.get('distribution_strategy', "auto"),
                'logger': metrics_logger,
                'max_epochs': MAX_EPOCHS,
                'callbacks': callbacks,
                'precision': '32',
                'gpus': config.get('gpus', [0]) #try -1 instead of [0] to use all available 
        }
        
        if (EARLY_STOPPING_PATIENCE != -1):
                fit_kwargs['early_stopping_patience'] = EARLY_STOPPING_PATIENCE
        
        #Add an early stopping callback to make sure best fit model is loaded after training: 
        # see line 179 in easySyntax/easy_model -> fit()
        callbacks.append(
                EarlyStopping(
                monitor = "val_loss",
                patience = EARLY_STOPPING_PATIENCE if EARLY_STOPPING_PATIENCE != -1 else MAX_EPOCHS,
                )
        )

        if (ACCUMULATE_GRAD_BATCHES != -1):
                fit_kwargs['accumulate_grad_batches'] = ACCUMULATE_GRAD_BATCHES
                
        if resumefromckpt:
                ckpt_path = config.get('ckpt_path')
                default_ckpt_paths = None
                if not ckpt_path:
                        # Only compute the default checkpoint path if not provided in the config
                        default_ckpt_paths = glob(os.path.join(sub_folder, 'checkpoints', '*'))
                        # default_ckpt_paths = glob(os.path.join(sub_folder, 'epoch_checkpoints', '*'))
                        logger.info(f"found default_ckpt_paths: {default_ckpt_paths}")

                if default_ckpt_paths:
                        # default_ckpt_paths.sort()
                        default_ckpt_paths = sorted(default_ckpt_paths, key=lambda x: int(re.search(r'epoch=(\d+)', x).group(1)))
                        ckpt_path = default_ckpt_paths[-1]

                if ckpt_path:
                        logger.info(f"Resuming from checkpoint path: {ckpt_path}")
                        fit_kwargs['ckpt_path'] = ckpt_path
                else:
                        logger.warning("No checkpoint path found to resume from.")

        #train the model
        model.fit(**fit_kwargs)

        #save the model weights and biases after training, saves statedict of the last epoch 
        model.save_state_dict(f'{modelpath}.pth')

        test_results = evaluate_GNN(model=model,
                                    dataloader_testing_instance=dataloader_testing_instance,
                                    config=config,
                                    sub_folder=sub_folder,
                                    logger=logger,
        )

        return test_results

def main(argv: Optional[Sequence[str]] = None):
        logger = Logger()
        args = parse_args(argv)
        resumefromckpt = args.resumefromckpt
        use_tmpdir = args.use_tmpdir
        config_path = args.config_path

        # Check if the current process is the main process for multi gpu training
        rank = int(os.environ["SLURM_PROCID"])
        is_main_process = (rank == 0)
        print(f"RANK: {rank}")
        localrank = os.environ.get("LOCAL_RANK")
        print(f"LOCAL_RANK: {localrank}")

        print(f"ismain?: {is_main_process} (based on rank==0)")

        config, sub_folder = utils.get_config_and_sub_folder(config_path, 
                                                        resumefromckpt, 
                                                        logger=logger, 
                                                        # is_main_process=is_main_process
        )    
        
        results = train_and_evaluate_GNN(config, 
                                        sub_folder, 
                                        logger=logger, 
                                        resumefromckpt=resumefromckpt, 
                                        use_tmpdir=use_tmpdir, 
        )

        if is_main_process:
                Analysis.PlottingRoutine( 
                                results=results, 
                                subfolder=sub_folder, 
                                config=config,
                                target_label=config.get('training_parameter', ['deposited_energy']), 
                                logger=logger
                )
        
if __name__ == "__main__":
        main()    

#maybe add pyyaml constructors (yaml.add_constructor) for Detector and backbones  