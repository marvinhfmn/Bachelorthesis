#!/usr/bin/env python3
import os
import pandas as pd
import re
from argparse import ArgumentParser, Namespace
from typing import Sequence, Optional, List, Union
from graphnet.models import StandardModel, Model
from graphnet.utilities.config import ModelConfig
from graphnet.utilities.logging import Logger
from graphnet.data.dataloader import DataLoader
import CustomAdditionsforGNNTraining as Custom
import Analysis
import utilities as utils

def parse_args(argv: Optional[Sequence[str]] = None) -> Namespace:
        """
        Method to parse arguments as a command line interface.
        """
        parser = ArgumentParser()
        parser.add_argument('--config_path', 
                            type=str, 
                            default="/home/saturn/capn/capn108h/programming_GNNs_and_training/config.yaml",
                            help="""Path to the config file""",
        )
        parser.add_argument('--subfolder', 
                            type=str, 
                            default=False,
                            help="""Path to subfolder, where the model config and potentially weights are saved.""", 
        )
        parser.add_argument('--modelname', 
                            type=str, 
                            default="GNN_DynEdge_mergedNuE_NuMu_NuTau",
                            help="""Name of the model that is in use""",
        )
        parser.add_argument('--use_tmpdir',
                            default=False,
                            action="store_true",
                            help="Whether or not to use the high bandwidth, low latency $TMPDIR of the gpu node for accessing the databases",
        )
        parser.add_argument('--perturb',
                            default=False,
                            action="store_true",
                            help="Whether to use a pertubation dict on the graph definition before evaluating the model.",
        )
        parser.add_argument('--ckpt_or_statedict_path',
                            type=str,
                            default='',
                            help="Checkpoint or state dict path to maybe load weights from.",
        )
        parser.add_argument('--plotting', 
                            dest='plotting', 
                            action='store_true', 
                            default=True, 
                            help="Whether to plot after evaluating the model.")
        parser.add_argument('--no-plotting', 
                            dest='plotting', 
                            action='store_false', 
                            help="Disable plotting after evaluating the model.")

        return parser.parse_args(argv)

# def get_dataloader_testing_instance():
#         # Construct dataloaders
#         training_dataset, validation_dataset, testing_dataset = Custom.CreateCustomDatasets(path=NuEfiles, 
#                                                                                             graph_definition=graph_definition, 
#                                                                                             features=feature_names, 
#                                                                                             truth=training_parameter
#                                                                                             )


#         dataloader_testing_instance = DataLoader(testing_dataset, config.get('dataloader_config', {'batch_size': 8, 'num_workers': 32}))
# 
def get_versioned_filename(subfolder, test_result_name, logger):
    if logger is None:
        logger = Logger()
    version = 1
    base_name = test_result_name
    while os.path.isfile(f"{subfolder}/{test_result_name}.h5"):
        logger.info(f"Test result '{test_result_name}.h5' already exists. Incrementing version...")
        test_result_name = f"{base_name}_v{version}"
        version += 1
    return test_result_name 

def evaluateModel(
        model: Union[Model, StandardModel], 
        subfolder: str, 
        config: dict, 
        # dataloader_testing_instance: DataLoader,
        test_result_name: str ='test_result', 
        use_tmpdir: bool = False,
        logger: Logger = None,
        )-> pd.DataFrame:
    """
    If there is already a test result with  do nothing else create dataset and evaluate the model on this test dataset, save results
    """
    if logger is None:
        logger = Logger()

    test_result_name = get_versioned_filename(subfolder, test_result_name, logger = logger)

    # Access the graph definition
    graph_definition = model._graph_definition
    # print("Graph Definition:", graph_definition)

    feature_names = model._graph_definition._input_feature_names

    # Access the training parameter
    training_target_label = model.target_labels
    print("Training Parameter:", training_target_label)

    lowercase_trainparam = [x.lower() for x in training_target_label]
    if 'deposited_energy' in lowercase_trainparam or 'deposited energy' in lowercase_trainparam:
            training_target_label = Custom.CustomLabel_depoEnergy()
    
    # Construct dataloaders
    training_dataset, validation_dataset, testing_dataset = utils.create_custom_datasets(
                                                                            config=config,
                                                                            graph_definition=graph_definition,
                                                                            feature_names=feature_names,
                                                                            training_target_label=training_target_label,
                                                                            logger=logger,
                                                                            use_tmpdir=use_tmpdir,
    )
    print(f"training dataset length {len(training_dataset)}")
    print(f"validation dataset length {len(validation_dataset)}")
    print(f"testing dataset length {len(testing_dataset)}")

    dataloader_testing_instance = DataLoader(testing_dataset, **config.get('dataloader_config', {'batch_size': 8, 'num_workers': 16}))
    
    logger.info("Predicting..")
    test_results = model.predict_as_dataframe(dataloader = dataloader_testing_instance,
                                        additional_attributes =  model.target_labels + config.get('addedattributes_test', ['classification', 'cosmic_primary_type']),
                                        gpus = [0]
                                        )
    logger.info("predicting done.")
    
    #save test results in hdf5 file
    try: 
        test_result_path = os.path.join(f"{subfolder}", test_result_name+'.h5')
        test_results.to_hdf(test_result_path, key='test_results', mode='w')
        logger.info(f"Succesfully evaluated the model. Results can be found under {test_result_path}")
    except:
        logger.info("Saving test results unsuccesful.")
    
    return test_results


def evaluateModel_v2(
        model: Union[Model, StandardModel], 
        subfolder: str, 
        dataloader_testing_instance: DataLoader = None, 
        path_to_datasets: str = "/home/woody/capn/capn108h",
        filename: str = 'test_results',
        key: str ='test_results',
        logger: Logger = None,
        use_tmpdir: bool = False,
        gpus: Optional[Union[List[int], int]] = None,
        )-> pd.DataFrame:
    """
    If there is already a test result do nothing else create datasets and evaluate the model on the test dataset, save results
    """
    if logger is None:
        logger = Logger()
        print("logger initiated")
        
    save_path = os.path.join(f"{subfolder}", filename+'.h5')

    if os.path.isfile(save_path):
        logger.info(f"There is already a {filename} file at {save_path}. Skipping evaluate method.")
        return
    else:
        # Access the graph definition
        graph_definition = model._graph_definition
        # print("Graph Definition:", graph_definition)

        feature_names = model._graph_definition._input_feature_names

        # Access the training parameter
        training_parameter_key = model.target_labels
        print("Training Parameter key:", training_parameter_key)
        truth = ["first_vertex_energy", "second_vertex_energy", "third_vertex_energy", "visible_track_energy", "visible_spur_energy"]
        test_truth = ['classification', 'cosmic_primary_type', 'first_vertex_x', 'first_vertex_y', 'first_vertex_z', 'sim_weight']

        # Construct dataloaders
        training_dataset, validation_dataset, testing_dataset = Custom.CreateCustomDatasets_CustomTrainingLabel(
                                                                                        path=path_to_datasets,
                                                                                        graph_definition=graph_definition, 
                                                                                        features=feature_names,
                                                                                        training_target_label=Custom.CustomLabel_depoEnergy(),
                                                                                        truth=truth,
                                                                                        classifications = [8, 9, 19, 20, 22, 23, 26, 27],
                                                                                        test_truth=test_truth,
                                                                                        random_state= 42,
                                                                                            )
        # torch.save(training_dataset, '/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_and_saved_models/training.pt')


        dataloader_testing_instance = DataLoader(testing_dataset, batch_size=8, num_workers=32)
        
        test_results = model.predict_as_dataframe(dataloader = dataloader_testing_instance,
                                            additional_attributes = training_parameter_key + test_truth,
                                            gpus = [0],
                                            )
        
        #save test results in hdf5 file
        try: 
            test_results.to_hdf(save_path, key=key, mode='w')
            logger.info(f"Succesfully evaluated the model. Results can be found under {save_path}")
        except:
            logger.info("Saving test results unsuccesful.")
        
        return test_results


# def InspectModelParameters():
#     model = utils.LoadModel()
#     print("Model parameters: ", model.parameters)

def main(argv: Optional[Sequence[str]] = None):
    logger = Logger()
    print("hi from main")
    args = parse_args(argv)

    subfolder = args.subfolder
    use_tmpdir = args.use_tmpdir
    perturb = args.perturb
    config_path = args.config_path
    modelname = args.modelname
    ckpt_or_statedict_path = args.ckpt_or_statedict_path
    plotting_bool = args.plotting

    path_to_modelconfig = os.path.join(subfolder, modelname+'.yml')

    model = utils.LoadModel(
                    path_to_modelconfig=path_to_modelconfig, 
                    path_to_statedict_or_ckpt = ckpt_or_statedict_path,
                    load_weights=True,
    )
    
    # Regular expression to match the epoch number
    epoch_match = re.search(r'epoch=(\d+)', ckpt_or_statedict_path)

    # Initialize test_result_name
    test_result_name = 'test_result'

    # Check if the regex found an epoch and append the number
    if epoch_match:
        epoch_number = epoch_match.group(1)
        test_result_name += f'_epoch{epoch_number}'
    
    config = utils.read_yaml(config_path=config_path)

    results = evaluateModel(
                model=model,
                subfolder=subfolder,
                config=config,
                test_result_name=test_result_name,
                use_tmpdir=use_tmpdir,
                logger=logger,
    )

    if  plotting_bool:
        logger.info("Plotting..")
        Analysis.PlottingRoutine(results=results,
                                subfolder=subfolder,
                                config=config,
                                logger=logger,
        )
    
if __name__ == "__main__":
    main()




# print(model.parameters)
# detector = model._graph_definition._detector
# print(detector)
# print(dir(model._graph_definition._node_definition._trainer))
# print(model._graph_definition._node_definition.parameters)
# print(model.target_labels)

# print(type(model))
# evaluateModel(model)
