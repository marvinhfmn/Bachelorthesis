#!/usr/bin/env python3

import os
from glob import glob
from time import gmtime, strftime
# import csv
import pandas as pd
import torch
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

# Constants (constant for a specified run)
NUM_WORKERS = 32
BATCH_SIZE = 8       
MAX_EPOCHS = 15
BASEFOLDER = "/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_and_saved_models/"
TRAINING_PARAMETER = 'first_vertex_energy'
LEARNING_RATE = 1e-4
EARLY_STOP_BOOL = False
EARLY_STOPPING_PATIENCE = 5
ACCUMULATE_GRAD_BATCHES_BOOL = True
ACCUMULATE_GRAD_BATCHES = 10

#Paths
#most recent databases
datasetpath = glob("/home/wecapstor3/capn/capn106h/l2_labeled/merged/*.db")
datasetpath.sort()
NuEfiles = datasetpath[0:3]
CorsikaSimfiles = [datasetpath[3]]
NuTaufiles = datasetpath[4:7]
NuMufiles = datasetpath[7:]


def create_sub_folder(timestamp: str, base_folder: str = BASEFOLDER) -> str:
    """
    Creates a sub-folder for the current run.
    """
    run_name = f"run_from_{timestamp}_UTC"
    sub_folder = os.path.join(base_folder, run_name)
    os.makedirs(sub_folder, exist_ok=True)
    return sub_folder

def save_params(params: dict, filename: str = 'gnn_params'):
        """
        Method to save some additional information about the run in a csv file.
        What is saved depends on the params attribute 
        """
        # Allows more dynamic adjustments in case the params schema differs from previous runs
        # Ensure the directory exists
        os.makedirs(BASEFOLDER, exist_ok=True)

        temp = os.path.join(BASEFOLDER, f"{filename}.csv")
        mode = 'a' if os.path.isfile(temp) else 'w'
        
        # Convert params to DataFrame and append separator
        df = pd.DataFrame.from_dict(data=params, orient='index')
        separator = pd.DataFrame([['--' * 20]])
        df = pd.concat([df, separator])
        
        # Save DataFrame to CSV
        df.to_csv(temp, header=False, mode=mode, lineterminator=os.linesep)


        #Code for gnn_params_v1
        # file_exists = os.path.isfile(os.path.join(BASEFOLDER, filename))
        # with open(os.path.join(BASEFOLDER, filename), mode='a', newline='') as file:
        #         writer = csv.DictWriter(file, params.keys())
        #         if not file_exists:
        #              writer.writeheader()
        #         writer.writerow(params)]

def train_and_evaluate(training_parameter: str, sub_folder: str) -> pd.DataFrame:
        """
        Train the model, validate it and then apply a testing dataset, saving results.
        """

        #defines features as required by the Detector class
        feature_names = ['dom_x', 'dom_y', 'dom_z', 'dom_time', 'charge', 'rde', 'pmt_area', 'hlc']

        # PercentileClusters
        #define feature names, properties that should be clustered on and the percentiles to cluster with
        #Should return a cluster for each feature that is not included in cluster_on
        cluster_on = ['dom_x', 'dom_y', 'dom_z']
        percentiles = [10, 20, 50, 70, 90]

        percentile_clustering_instance = PercentileClusters(cluster_on=cluster_on, percentiles=percentiles, input_feature_names=feature_names)

        #Define graph and use percentile clustering
        graph_definition = KNNGraph(detector = IceCube86(), 
                                node_definition=percentile_clustering_instance
                                )

        #Create Datasets for training, validation and testing
        training_dataset, validation_dataset, testing_dataset = Custom.CreateCustomDatasets(path=NuEfiles, graph_definition=graph_definition, features=feature_names, truth=training_parameter)

        #make Dataloaders
        dataloader_training_instance = DataLoader(training_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        dataloader_validation_instance = DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        dataloader_testing_instance = DataLoader(testing_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        
        #Loss Logger
        loss_logger = CSVLogger(save_dir=sub_folder, name="losses", version=0)

        # Select backbone
        backbone = DynEdge(nb_inputs = graph_definition.nb_outputs,
                        global_pooling_schemes=["min", "max", "mean"])

        # build task    
        task = EnergyReconstruction(target_labels = [training_parameter],
                                hidden_size=backbone.nb_outputs,
                                loss_function = LogCoshLoss(),
                                transform_prediction_and_target = lambda x: torch.log(x),
                                transform_inference = lambda x: torch.pow(10,x),
                                )

        # instantiate model
        model = StandardModel(graph_definition = graph_definition,
                        backbone = backbone,
                        tasks = task,
                        optimizer_kwargs = {'lr' : LEARNING_RATE}
                        )
        

        #train model
        fit_kwargs = {
                'train_dataloader': dataloader_training_instance,
                'val_dataloader': dataloader_validation_instance,
                'distribution_strategy': "ddp_notebook",
                'logger': loss_logger,
                'max_epochs': MAX_EPOCHS,
                'precision': '32',
                'gpus': [0]
        }
        
        if EARLY_STOP_BOOL:
                fit_kwargs['early_stopping_patience'] = EARLY_STOPPING_PATIENCE
        if ACCUMULATE_GRAD_BATCHES_BOOL:
                fit_kwargs['accumulate_grad_batches'] = ACCUMULATE_GRAD_BATCHES
        
        model.fit(**fit_kwargs)

        
        #save model
        name = 'GNN_DynEdge_mergedNuE'
        modelpath = os.path.join(sub_folder, name)
        model.save_config(f'{modelpath}.yml')
        model.save_state_dict(f'{modelpath}.pth')

        # save additional parameters that aren't included in the yml file 
        params = {
        'Time of run': sub_folder.split("/")[-1],
        'Batch_Size': BATCH_SIZE,
        'Num of max epochs': MAX_EPOCHS,
        'Num of workers': NUM_WORKERS,
        'Early stopping activated': EARLY_STOP_BOOL,
        }

        if EARLY_STOP_BOOL:
                params['Early stopping patience'] = EARLY_STOPPING_PATIENCE
        
        save_params(params=params)

        #get predictions
        test_results = model.predict_as_dataframe(dataloader = dataloader_testing_instance,
                                                additional_attributes = [training_parameter],
                                                gpus = [0])   

        #save test results in hdf5 file
        test_results.to_hdf(os.path.join(sub_folder, 'test_results.h5'), key='test_results', mode='w')

        return test_results

def PlottingRoutine(results: pd.DataFrame = None, subfolder: str = None):
        """
        Function to handle all the plotting. Mainly relies on the methods defined in Analysis.py
        
        """
        #if no subfolder is specified take the last one in the runs_and_saved_models folder
        if subfolder is None:
               subfolder = sorted(os.listdir(BASEFOLDER))[-1]
               if len(subfolder) == 0:
                      raise FileNotFoundError("No subfolder found. Please make sure there is a folder available for loading test results and/or storing Plots.")
               
        # Check if metrics file is created
        metrics_path = os.path.join(subfolder, "losses/version_0/metrics.csv")
        if os.path.exists(metrics_path):
                print(f"Metrics file created at: {metrics_path}")
        else:
                print("ERROR: Metrics file not found.")

        #if no result dataframe is handed over, load the dataframe from the specified subfolder
        if results is None:
               results = Analysis.loadresults(subfolder=subfolder)

        #All the Plotting calls
        Analysis.plot_lossesandlearningrate(subfolder)
        Analysis.plot_resultsashisto(results, subfolder=subfolder, truth=TRAINING_PARAMETER)
        Analysis.plotEtruevsEreco(results, subfolder=subfolder , normalise='E_true')
        Analysis.plotEtruevsEreco(results, subfolder=subfolder, normalise='E_reco')
        Analysis.plotIQRvsEtrue(results, subfolder=subfolder)

def main():

        timestamp = strftime("%Y.%m.%d_%H:%M:%S", gmtime())
        sub_folder = create_sub_folder(timestamp=timestamp, base_folder=BASEFOLDER)
        results = train_and_evaluate(TRAINING_PARAMETER, sub_folder)
        PlottingRoutine(results=results, subfolder=sub_folder)

if __name__ == "__main__":
        # args = parse_args()   
        main()    
