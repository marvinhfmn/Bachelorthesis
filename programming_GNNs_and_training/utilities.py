from typing import Dict, List, Optional, Tuple, Union, Callable
import os
import numpy as np
import pandas as pd
import sqlite3 as sql
import h5py
import shutil
from time import time, gmtime, strftime
from termcolor import colored
from datetime import timedelta
import yaml
import torch
from torch_geometric.data import Data
from glob import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from graphnet.data.dataset import SQLiteDataset, EnsembleDataset
from graphnet.utilities.logging import Logger
from graphnet.models.graphs import GraphDefinition
from graphnet.models import StandardModel, Model
from graphnet.utilities.config import ModelConfig
from graphnet.training.utils import make_dataloader
from graphnet.training.labels import Label
import CustomAdditionsforGNNTraining as Custom
import Analysis
from CustomSQL_dataset import CustomSQLiteDataset 

def DataloaderSelection(
          db: Union[str, List[str]], 
          value: float = 1.0, 
          column: str = 'classification'
          ):
    """
    Selects indices where the column fulfills the specified value. In case of column = 'classification and value = 1.0 throughgoing tracks would be selected.
    Returns the database filepaths and the selection indices in that order
    """
    indices = []
    #ensure that db is a list
    if isinstance(db, str):
        db = [db]
    
    for dbfile in db:
            con = sql.connect(dbfile)
            cur = con.cursor()
            ids = cur.execute(f"SELECT event_no FROM truth WHERE {column}={value}")
            ids = ids.fetchall()
            ids = [[dbfile, int(event_id[0])] for event_id in ids]
            indices += ids
    
    dbfilepaths = [path for [path, index] in indices]
    selection_indices = [ids for [path, ids] in indices]
    return dbfilepaths, selection_indices

def GetSelectionIndicesExcludeNULL(
          db:  Union[str, List[str]], 
          column: str
          ):
    """
    Selects events from db that are not NULL and returns database filepaths and the selection indices in that order
    """
    indices = []
    #ensure that db is a list
    if isinstance(db, str):
        db = [db]
    
    for dbfile in db:
            con = sql.connect(dbfile)
            cur = con.cursor()
            ids = cur.execute(f"SELECT event_no FROM truth WHERE {column} IS NOT NULL")
            ids = ids.fetchall()
            ids = [[dbfile, int(event_id[0])] for event_id in ids]
            indices += ids
    dbfilepaths = [path for [path, index] in indices]
    selection_indices = [ids for [path, ids] in indices]
    return dbfilepaths, selection_indices 

def GetSelectionIndicesExcludeNULLandZERO(
          db:  Union[str, List[str]], 
          column: str
          ):
    """
    Selects events from db that are not NULL and are not 0 (zero) and returns database filepaths and the selection indices in that order
    """
    indices = []
    #ensure that db is a list
    if isinstance(db, str):
        db = [db]
    
    for dbfile in db:
            con = sql.connect(dbfile)
            cur = con.cursor()
            ids = cur.execute(f"SELECT event_no FROM truth WHERE {column} IS NOT NULL AND {column} != 0")
            ids = ids.fetchall()
            ids = [[dbfile, int(event_id[0])] for event_id in ids]
            indices += ids
    dbfilepaths = [path for [path, index] in indices]
    selection_indices = [ids for [path, ids] in indices]
    return dbfilepaths, selection_indices

def GetSelectionIndicesExcludebelowThreshold(
          db:  Union[str, List[str]], 
          column: str, 
          threshold: int = 10, #threshold energy in units of db entries
          ):
    """
    Selects events from db that are above a certain threshold and returns database filepaths and the selection indices, in that order.
    """
    indices = []
    logger = Logger()
    if isinstance(column, List):
        if len(column) > 1:
            logger.info("more than one truth parameter not supported yet... taking the first entry in the truth list")
        column = column[0]
        
    #ensure that db is a list
    if isinstance(db, str):
        db = [db]
    
    for dbfile in db:
            con = sql.connect(dbfile)
            cur = con.cursor()
            ids = cur.execute(f"SELECT event_no FROM truth WHERE {column} > {threshold}")
            ids = ids.fetchall()
            ids = [[dbfile, int(event_id[0])] for event_id in ids]
            indices += ids
    dbfilepaths = [path for [path, index] in indices]
    selection_indices = [ids for [path, ids] in indices]
    return dbfilepaths, selection_indices

def GetSelectionIndices_basedonClassification(
        db:  Union[str, List[str]], 
        classifications: Union[int, List[int]] = [8, 9, 19, 20, 22, 23, 26, 27], 
        deposited_energy_cols: Union[str, List[str]] = ["first_vertex_energy", "second_vertex_energy", "third_vertex_energy", "visible_track_energy", "visible_spur_energy"],
        threshold: float = 10.0, #threshold energy in units of db entries
        )-> Tuple[List[str], List[List[int]]]:
    """
    Get the indices of events with a specific classification and where the sum of specified deposition energies is above a threshold.
    
    Parameters:
    db (Union[str, List[str]]): Database file name or list of database file names.
    classifications (Union[int, List[int]]): Either a single classification or a list of classifications.
                                             If -1, no filter based on classification is applied.
    deposited_energy_cols (Union[str, List[str]]): List of column names for vertex energies.
    threshold (float): Threshold for the sum of vertex energies.
    
    Returns:
    Tuple[List[str], List[List[int]]]: Tuple containing:
                                       - List of database file names.
                                       - List of lists of event numbers that match the criteria.
    """

    if isinstance(db, str):
        db = [db]

    if classifications == -1:
        classification_condition = "1=1"       
    else:
        if isinstance(classifications, int):
            classifications = [classifications]
        classification_condition = f"classification IN ({', '.join(map(str, classifications))})"
    
    if isinstance(deposited_energy_cols, str):
        deposited_energy_cols = [deposited_energy_cols]
    deposited_energy_str = " + ".join([f"IFNULL({col}, 0)" for col in deposited_energy_cols])

    result  = []

    for database in db:
        with sql.connect(database) as con:
            cur = con.cursor()
            query = f"""
            SELECT event_no 
            FROM truth 
            WHERE {classification_condition} 
            AND ({deposited_energy_str}) > ?
            """
            ids = cur.execute(query, (threshold,)).fetchall()
            selected_event_nos = [id[0] for id in ids]

            result.append((database, selected_event_nos))
    
    databases, indices = zip(*result) if result else ([], [])
    return list(databases), list(indices)

def GetTrainingLabelEntries(
    db: Union[str, List[str]], 
    column: str = 'deposited_energy', 
    threshold: int = 10,  # threshold energy in units of db entries
    rows_to_calculate_column: List[str] = ["first_vertex_energy", "second_vertex_energy", "third_vertex_energy", "visible_track_energy", "visible_spur_energy"],  
) -> List[float]:
    """
    Selects values of column from the db where they are above a certain threshold and returns a list of these values.
    If the column is 'deposited_energy' and not directly present in the db, calculates it using provided columns.
    """
    energy_values = []
    
    # Ensure that db is a list
    if isinstance(db, str):
        db = [db]
    
    for dbfile in db:
        with sql.connect(dbfile) as con:
            cur = con.cursor()
            # Check if the column exists in the 'truth' table
            cur.execute("PRAGMA table_info(truth)")
            columns = [info[1] for info in cur.fetchall()]

            if column not in columns:
                deposited_energy_str = " + ".join([f"IFNULL({col}, 0)" for col in rows_to_calculate_column])
                query = f"""
                SELECT ({deposited_energy_str}) AS deposited_energy
                FROM truth 
                WHERE ({deposited_energy_str}) > ?
                """
            else:
                query = f"SELECT {column} FROM truth WHERE {column} > ?"
            
            cur.execute(query, (threshold,))
            values = cur.fetchall()
            energy_values += [float(value[0]) for value in values]
    
    return energy_values
     
#Idea: Create a custom dataset as e.g. a subdataset where the classification is specified and 
# the data is split into train, val and test sets
def CreateCustomDatasets_traininglabelinDataset(
                        path: Union[str, List[str]], 
                        graph_definition: GraphDefinition, 
                        features: List[str], 
                        truth: Union[str, List[str]], 
                        training_target_label: Union[str, List[str]] = 'first_vertex_energy',
                        pulsemap: Union[str, List[str]] = 'InIceDSTPulses', 
                        truth_table: str = 'truth',
                        test_size: float = 0.1,
                        random_state = 42,
                        logger: Logger = None,
                        setThreshold: bool = True, 
                        threshold: int = 10,
                        INCLUDEZEROS: bool = False, #'0' entries in databases are abandoned by default
                        INCLUDENULL: bool = False, #NULL entries in databases are abandoned by default
                        ) -> Tuple[EnsembleDataset, EnsembleDataset, EnsembleDataset]:
    """
    Creates a custom Dataset, which can be reproduced due to the random_state variable. Although I am not sure about the event distribution inside these datasets
    Only works if training label is in database. Method not applicable if you want to use a custom label, that is calculated on the fly using GraphNet.
    """
    if logger == None:
        logger = Logger()
    
    if(INCLUDENULL):
        # logger.info(f"Jokes on you, there is no functionality in setting INCLUDENULL to {INCLUDENULL}, except you don't get a dataset. Nice.")
        raise Exception(f"Jokes on you, there is no functionality in setting INCLUDENULL to {INCLUDENULL}, except you don't get a dataset. Nice.")
      
    #Assures truth (or training parameter) is a list
    if isinstance(truth, str):
        truth = [truth]
        #truth has to be a list or tupel for line 283 in dataset.py: assert isinstance method to not fail
    if isinstance(training_target_label, list):
        training_target_label = training_target_label[0]
        logger.info(f"Currently only one training label supported. Therefore choosing the first entry in provided list: {training_target_label}")

    common_kwargs = dict(
                pulsemaps=pulsemap,
                features=features,
                truth=truth, 
                truth_table=truth_table,
                graph_definition=graph_definition,
            )
    
    if isinstance(path, str) or (isinstance(path, List)):
        if isinstance(path, str):
            logger.info("One dataset path detected.")
            path = [path] # enumerate(path) would otherwise lead to the string splitting into single characters
        elif path and all(isinstance(s, str) for s in path):
            logger.info("Multiple dataset paths detected")
    
        
        trainingdatasets = []
        validationdatasets = []
        testingdatasets = []

        #splitting into train, val, test
        for index, singledataset in enumerate(path):
            #Get indices where the training parameter fulfills requirements 
            if(setThreshold):
                assert (INCLUDEZEROS==False), "Combination of INCLUDEZERO: True and setThreshold: True not supported"
                temp, VALIDINDICES = GetSelectionIndicesExcludebelowThreshold(db = singledataset, column=training_target_label, threshold=threshold)
            elif(INCLUDEZEROS):
                temp, VALIDINDICES = GetSelectionIndicesExcludeNULL(db = singledataset, column=training_target_label) 
            else:
                temp, VALIDINDICES = GetSelectionIndicesExcludeNULLandZERO(db = singledataset, column=training_target_label) 

            trainval_selection, test_selection = train_test_split(
            VALIDINDICES, test_size=test_size, random_state=random_state
            )
            training_selection, validation_selection = train_test_split(
            trainval_selection, test_size=test_size, random_state=random_state
            )
            trainingdatasets.append(SQLiteDataset(path=singledataset, **common_kwargs, selection=training_selection))
            validationdatasets.append(SQLiteDataset(path=singledataset, **common_kwargs, selection=validation_selection))
            testingdatasets.append(SQLiteDataset(path=singledataset, **common_kwargs, selection=test_selection))
        
        training_ensemble = EnsembleDataset(trainingdatasets)
        validation_ensemble = EnsembleDataset(validationdatasets)
        testing_ensemble = EnsembleDataset(testingdatasets)

        return training_ensemble, validation_ensemble, testing_ensemble
    
    else: 
        logger.info("Path is an empty list ... Execution terminated")
        return
    
    #save dataset configuration doesn't work for ensembleDatasets
    # dataset.config.dump(f"savedDatasets/{datasetname}.yml")

def CreateCustomDatasets_CustomTrainingLabel(
                        path: Union[str, List[str]], 
                        graph_definition: GraphDefinition, 
                        features: List[str], 
                        truth: Union[str, List[str]], 
                        classifications: Union[int, List[int]] = [8, 9, 19, 20, 22, 23, 26, 27],
                        training_target_label: Label = Custom.CustomLabel_depoEnergy(),
                        deposited_energy_cols: List[str] = ["first_vertex_energy", "second_vertex_energy", "third_vertex_energy", "visible_track_energy", "visible_spur_energy"],
                        pulsemap: Union[str, List[str]] = 'InIceDSTPulses', 
                        truth_table: str = 'truth',
                        config_for_dataset_datails: Dict[int, int] = {},
                        root_database_lengths: dict = {},
                        test_size: float = 0.1,
                        random_state = 42,
                        test_truth: Union[str, List[str]] =  ['cosmic_primary_type', 'first_vertex_x', 'first_vertex_y', 'first_vertex_z', 'sim_weight'],
                        logger: Logger = None, 
                        threshold: int = 10,
                        save_database_yaml: bool = False,
                        ) -> Tuple[EnsembleDataset, EnsembleDataset, EnsembleDataset]:
    """
    Creates custom training, validation, and testing datasets from SQLite databases.

    Parameters:
    - path: File path or list of file paths to SQLite database(s) containing data.
    - graph_definition: Definition of graph structure.
    - features: List of features to include in the datasets.
    - truth: List of event level truths to add to the graph of all datasets (train, val and test).
    - classifications: List of classification IDs to filter events by. If -1, no filter is applied based on classification.
    - training_target_label: Target label or feature to train on (e.g., energy deposition).
    - deposited_energy_cols: List of columns representing deposited energy to consider for thresholding.
    - pulsemap: Name or list of names of pulsemap tables in the databases.
    - truth_table: Table name in the databases containing truth information.
    - config_for_dataset_datails: dictionary of details about the database to be used with the dataset. e.g. 'energy_levels': what database id corresponds to high, mid, low energy events
        'ratio': what is the ratio between those and 'total_events_per_flavor': what is the desired total amount of events to be used per flavor. 
    - test_size: Proportion of data to reserve for testing. The training and validation datasets are split with the same proportion. 
        e.g test_size=0.1: trainval 90%, test 10%; then validation 10% of trainval, so 9% of the original data size and 81% training. 
    - random_state: Seed for random number generator to ensure reproducibility.
    - test_truth: Optional List of event level truths to add to the graph of only the testing dataset. list elements should be exclusive (no overlap with truth)
    - logger: Optional logger object for logging messages.
    - threshold: Threshold value for deposited energy columns to filter events.
    - save_database_yaml: Whether to save the yaml file for the databases or not

    Returns:
    - Tuple containing three EnsembleDataset objects:
        - Training dataset.
        - Validation dataset.
        - Testing dataset.
    """
    if logger == None:
        logger = Logger()

    #Assures truth (or training parameter) is a list
    if isinstance(truth, str):
        truth = [truth]
    #truth has to be a list or tupel for line 283 in dataset.py: assert isinstance method to not fail
    if isinstance(test_truth, str):
        test_truth = [test_truth]

    trainval_kwargs = dict(
                pulsemaps=pulsemap,
                features=features,
                truth=truth, 
                truth_table=truth_table,
                graph_definition=graph_definition,
            )
    test_kwargs = dict(
                pulsemaps=pulsemap,
                features=features,
                truth=truth + test_truth, 
                truth_table=truth_table,
                graph_definition=graph_definition,
            )
    
    if isinstance(path, str) or (isinstance(path, List)):
        if isinstance(path, str):
            logger.info("One dataset path detected.")
            path = [path] # enumerate(path) would otherwise lead to the string splitting into single characters
        elif path and all(isinstance(s, str) for s in path):
            logger.info("Multiple dataset paths detected")
        else: 
            logger.info("Path is an empty list ... Execution terminated")
            return
    
    # Define the energy levels
    energy_levels = config_for_dataset_datails.get('energy_levels', {
        'high': [22612, 22644, 22635],
        'mid': [22613, 22645, 22634],
        'low': [22614, 22646, 22633]
    }
    )

    # Define the ratio
    ratios = config_for_dataset_datails.get('ratios', {'high': 1, 'mid': 4, 'low': 10})

    # Calculate the total ratio sum
    total_ratio = sum(ratios.values())

    # Determine the number of valid indices for each energy level
    total_events_per_flavor = config_for_dataset_datails.get('total_events_per_flavor', total_ratio*1e5)  # total number of events to be distributed

    num_high_events = int(total_events_per_flavor * (ratios['high'] / total_ratio))
    num_mid_events = int(total_events_per_flavor * (ratios['mid'] / total_ratio))
    num_low_events = int(total_events_per_flavor * (ratios['low'] / total_ratio))
    
    dataset_savedir = '/home/saturn/capn/capn108h/programming_GNNs_and_training/datasets/'
    hdf5_filedir = os.path.join(dataset_savedir, 'dataset_indices')

    target_label_str = training_target_label
    if not isinstance(target_label_str, str):
        target_label_str = target_label_str.key

    os.makedirs(hdf5_filedir, exist_ok=True)
    hdf5_filepath = os.path.join(hdf5_filedir, f'valid_indices_for_{target_label_str}.h5')  

    trainingdatasets = []
    validationdatasets = []
    testingdatasets = []

    #splitting into train, val, test
    for singledataset in path:

        database_id = int(singledataset.split('/')[-1].split('_')[0])

        root_database_length = root_database_lengths.get(database_id)
        # Determine the number of events to select based on energy level
        if database_id in energy_levels['high']:
            num_events = num_high_events
        elif database_id in energy_levels['mid']:
            num_events = num_mid_events
        elif database_id in energy_levels['low']:
            num_events = num_low_events
        else:
            num_events = 3* total_events_per_flavor  # for background, just take the same amount as for each flavor
        
        # print(hdf5_filepath)
        # print('\n')
        valid_ids = None

        # Load saved indices if available
        grp_name = f"{database_id}_{'_'.join(map(str, classifications))}"
        if os.path.exists(hdf5_filepath):
            with h5py.File(hdf5_filepath, 'r') as f:
                if grp_name in f:
                    valid_ids = f[grp_name][:]
                    # logger.info(f"Loaded indices from {hdf5_filepath}")
        
        if valid_ids is None:
            logger.info(f'No presaved incides found. Creating Datasets...')
            # Get indices where classification fulfills requirements           
            database, valid_ids = GetSelectionIndices_basedonClassification(
                                                db=singledataset, 
                                                classifications=classifications, 
                                                deposited_energy_cols=deposited_energy_cols,
                                                threshold=threshold
                                                )
            valid_ids = valid_ids[0]
            # print(f'length of valid indices for {database_id}: {len(valid_ids)}')

            # Save the indices for future use
            with h5py.File(hdf5_filepath, 'a') as f:
                if grp_name in f:
                    del f[grp_name]
                f.create_dataset(grp_name, data=valid_ids)

        print(f'ID: {database_id}, Num_events: {num_events}, valid ids: {len(valid_ids)}')

        # Randomly select a subset of valid_ids based on the num_events calculated
        if len(valid_ids) > num_events:
            np.random.seed(seed=random_state)
            valid_ids = np.random.choice(valid_ids, num_events, replace=False)
            print(f'valid ids larger than num events')
        else: 
            print(f'valid ids smaller than num events')        
        
        # print("\n")

        trainval_selection, test_selection = train_test_split(
        valid_ids, test_size=test_size, random_state=random_state
        )
        training_selection, validation_selection = train_test_split(
        trainval_selection, test_size=test_size, random_state=random_state
        )
        training_dataset = CustomSQLiteDataset(path=singledataset, **trainval_kwargs, selection=training_selection)
        training_dataset.add_label(training_target_label)
        trainingdatasets.append(training_dataset)
        print(f'id {database_id} Length train selection {len(training_selection)}')

        validation_dataset = CustomSQLiteDataset(path=singledataset, **trainval_kwargs, selection=validation_selection)
        validation_dataset.add_label(training_target_label)
        validationdatasets.append(validation_dataset)
        print(f'id {database_id}, Length val selection {len(validation_selection)}')

        testing_dataset = CustomSQLiteDataset(path=singledataset, **test_kwargs, selection=test_selection)
        testing_dataset.add_label(training_target_label)
        testingdatasets.append(testing_dataset)
        print(f'id {database_id} Length test selection {len(test_selection)}')

        if(save_database_yaml):
            yaml_save_path = '/home/saturn/capn/capn108h/programming_GNNs_and_training/datasets'
            dataset_types = ["train", "val", "test"]
            datasets = {
                "train": training_dataset,
                "val": validation_dataset,
                "test": testing_dataset
            }

            for dataset_type in dataset_types:
                dir_path = os.path.join(yaml_save_path, f"{dataset_type}_dataset_configs")
                os.makedirs(dir_path, exist_ok=True)
                datasets[dataset_type].save_config(os.path.join(dir_path, f"{database_id}_{dataset_type}_config.yml"))


    training_ensemble = EnsembleDataset(trainingdatasets)
    print(f"Length training ensemble: {len(training_ensemble)}")

    validation_ensemble = EnsembleDataset(validationdatasets)
    print(f"Length validation ensemble: {len(validation_ensemble)}")

    testing_ensemble = EnsembleDataset(testingdatasets)
    print(f"Length testing ensemble: {len(testing_ensemble)}")

    print(f"Total events considered: {len(training_ensemble) + len(validation_ensemble)+ len(testing_ensemble)}")

    # total_ensemble = EnsembleDataset([training_ensemble, validation_ensemble, testing_ensemble])

    dataset_savedir_for_training_label = os.path.join(dataset_savedir, f"classif{'_'.join(map(str, classifications))}")
    os.makedirs(dataset_savedir_for_training_label, exist_ok=True)
    ensemble_path = os.path.join(dataset_savedir_for_training_label, f"TotalEnsemble__{target_label_str}_rng{random_state}.pt")
    if not os.path.exists(ensemble_path):
        torch.save([training_ensemble, validation_ensemble, testing_ensemble], ensemble_path)
        logger.info(f"New Total Ensemble saved to: {ensemble_path}")

    return training_ensemble, validation_ensemble, testing_ensemble

def CreateCustomDatasets_CustomTrainingLabel_splitDatabases(
                        path: Union[str, List[str]], 
                        graph_definition, 
                        features: List[str], 
                        truth: Union[str, List[str]],
                        training_target_label: Label = Custom.CustomLabel_depoEnergy(),
                        # deposited_energy_cols: List[str] = ["first_vertex_energy", "second_vertex_energy", "third_vertex_energy", "visible_track_energy", "visible_spur_energy"],
                        pulsemap: Union[str, List[str]] = 'InIceDSTPulses', 
                        truth_table: str = 'truth',
                        test_truth: Union[str, List[str]] =  ['cosmic_primary_type', 'first_vertex_x', 'first_vertex_y', 'first_vertex_z', 'sim_weight'],
                        logger: Logger = None, 
                        ) -> Tuple[EnsembleDataset, EnsembleDataset, EnsembleDataset]:
    """
    Creates custom training, validation, and testing datasets from SQLite databases.

    Parameters:
    - path: File path or list of file paths to SQLite database(s) containing data. Must be {databaseid}_{train,val,test}_selection.db
    - graph_definition: Definition of graph structure.
    - features: List of features to include in the datasets.
    - truth: List of event level truths to add to the graph of all datasets (train, val and test).
    - training_target_label: Target label or feature to train on (e.g., energy deposition).
    - deposited_energy_cols: List of columns representing deposited energy to consider for thresholding.
    - pulsemap: Name or list of names of pulsemap tables in the databases.
    - truth_table: Table name in the databases containing truth information.
    - test_truth: Optional List of event level truths to add to the graph of only the testing dataset. list elements should be exclusive (no overlap with truth)
    - logger: Optional logger object for logging messages.

    Returns:
    - Tuple containing three EnsembleDataset objects:
        - Training dataset.
        - Validation dataset.
        - Testing dataset.
    """
    if isinstance(path, str):
        path = [path]

    trainval_kwargs = dict(
                pulsemaps=pulsemap,
                features=features,
                truth=truth, 
                truth_table=truth_table,
                graph_definition=graph_definition,
            )
    test_kwargs = dict(
                pulsemaps=pulsemap,
                features=features,
                truth=truth + test_truth, 
                truth_table=truth_table,
                graph_definition=graph_definition,
            )
    
    trainingdatasets = []
    validationdatasets = []
    testingdatasets = []

    for singledataset in path:
        if "train_selection" in singledataset:
            training_dataset = CustomSQLiteDataset(path=singledataset, **trainval_kwargs)
            training_dataset.add_label(training_target_label)
            trainingdatasets.append(training_dataset)
        elif "val_selection" in singledataset:
            validation_dataset = CustomSQLiteDataset(path=singledataset, **trainval_kwargs)
            validation_dataset.add_label(training_target_label)
            validationdatasets.append(validation_dataset)
        elif "test_selection" in singledataset:
            testing_dataset = CustomSQLiteDataset(path=singledataset, **test_kwargs)
            testing_dataset.add_label(training_target_label)
            testingdatasets.append(testing_dataset)

    if len(trainingdatasets) > 0:
        training_ensemble = EnsembleDataset(trainingdatasets)
        print(f"Length training ensemble: {len(training_ensemble)}")
    else:
        training_ensemble = []
        print(f"Length of training datasets is {len(trainingdatasets)}")
    
    if len(validationdatasets) > 0:
        validation_ensemble = EnsembleDataset(validationdatasets)
        print(f"Length validation ensemble: {len(validation_ensemble)}")
    else:
        validation_ensemble = []
        print(f"Length of validation datasets is {len(validationdatasets)}")

    if len(testingdatasets) > 0:
        testing_ensemble = EnsembleDataset(testingdatasets)
        print(f"Length testing ensemble: {len(testing_ensemble)}")
    else:
        testing_ensemble = []
        print(f"Length of testing datasets is {len(testingdatasets)}")

    return training_ensemble, validation_ensemble, testing_ensemble

#Aggregate function to decide what Custom Database creation method to choose from 
def create_custom_datasets(
    config: dict,
    graph_definition: GraphDefinition,
    feature_names: List[str],
    training_target_label: str,
    logger: Logger,
    use_tmpdir: bool,
) -> Tuple[EnsembleDataset, EnsembleDataset, EnsembleDataset]:
    """
    Create custom datasets based on the configuration and use_tmpdir flag.

    Parameters:
    - config: Configuration dictionary
    - graph_definition: Graph definition
    - feature_names: List of feature names
    - training_target_label: Target label for training
    - logger: Logger instance
    - use_tmpdir: Flag to decide which method to use for dataset creation

    Returns:
    - training_dataset: Training dataset
    - validation_dataset: Validation dataset
    - testing_dataset: Testing dataset
    """

    if use_tmpdir:
        tmpdir = os.environ["TMPDIR"]
        db_files = []
        for root, _, files in os.walk(tmpdir):
            for file in files:
                if file.endswith(".db"):
                    db_files.append(os.path.join(root, file))
        # logger.info(f"Selection database folder contents: {db_files}")

        TRUTH = config.get('addedattributes_trainval', ["first_vertex_energy", "second_vertex_energy", "third_vertex_energy", "visible_track_energy", "visible_spur_energy"])
        TEST_TRUTH = config.get('addedattributes_test', ['classification', 'cosmic_primary_type', 'first_vertex_x', 'first_vertex_y', 'first_vertex_z', 'sim_weight'])

        training_dataset, validation_dataset, testing_dataset = CreateCustomDatasets_CustomTrainingLabel_splitDatabases(
            path=db_files,
            graph_definition=graph_definition, 
            features=feature_names,
            truth=TRUTH,
            training_target_label=training_target_label,
            test_truth=TEST_TRUTH,
            logger=logger,
        )

    else:
        datasetpaths = get_datasetpaths(config=config)

        if config.get('training_parameter_inDatabase', False):
            TRUTH = config.get('training_parameter', []) + config.get('addedattributes_trainval', [])
            training_dataset, validation_dataset, testing_dataset = CreateCustomDatasets_traininglabelinDataset(
                path=datasetpaths, 
                graph_definition=graph_definition, 
                features=feature_names, 
                training_target_label=training_target_label,
                truth=TRUTH,
                random_state=config.get('random_state', 42),
            )
        else: 
            TRUTH = config.get('addedattributes_trainval', [])
            TEST_TRUTH = config.get('addedattributes_test', None)
            training_dataset, validation_dataset, testing_dataset = CreateCustomDatasets_CustomTrainingLabel(
                path=datasetpaths,
                graph_definition=graph_definition, 
                features=feature_names,
                training_target_label=training_target_label,
                truth=TRUTH,
                classifications=config.get('classifications_to_train_on', [8, 9, 19, 20, 22, 23, 26, 27]),
                test_truth=TEST_TRUTH,
                config_for_dataset_datails=config.get('dataset_details', {}),
                root_database_lengths=config.get("database_lengths", {}), 
                test_size=config.get('dataset_details', {}).get('test_size'),
                random_state=config.get('random_state', 42),
                logger=logger,
                save_database_yaml=config.get('save_database_yaml', False),
            )
    logger.info(f"Detected training_target_label: {training_target_label}")

    return training_dataset, validation_dataset, testing_dataset


def HandleZeros(x):
    """
    Handle Entries in Database where training parameter is zero. would otherwise lead to infinite losses (in transform target)
    Not relevant if you use GetSelectionIndicesExcludeNULLandZERO or GetSelectionIndicesExcludebelowThreshold with a threshold above zero
    """
    zero_mask = (x == 0) #Create mask for entries that are 0 
    if torch.is_tensor(x):
        if not torch.any(zero_mask): #if there are no 0 entries apply log10 directly
                return torch.log10(x)
        else:
                result = torch.empty_like(x) #Initialize the result tensor with the same shape as elements
                result[zero_mask] = -7 #apply -7 to zero elements, which is equivalent to torch.log10(torch.tensor(1e-7)), where 1e-7 is an arbitrarily chosen value
                result[~zero_mask] = torch.log10(x[~zero_mask]) #apply log10 to non-zero elements
                return result


# def LoadModel(
#         subfolder: str, 
#         model_config_file_path: str = None, 
#         weight_file_path: str = None, 
#         modelname: str = "GNN_DynEdge_mergedNuEandNuTau", 
#         load_weights: bool = True,
#         ):
#     """
#     Load model from config and initialise weights if there is a '.pth' file to be found.
#     """
#     # Construct the full paths
#     if not model_config_file_path:
#         model_config_file_path = os.path.join(subfolder, f"{modelname}.yml")
#     if not weight_file_path:
#         weight_file_path = os.path.join(subfolder, f"{modelname}.pth")

#     # Load model configuration
#     model_config = ModelConfig.load(model_config_file_path)
    
#     # Initialize model with randomly initialized weights
#     model = Model.from_config(model_config, trust=True)

#     # Check if the weight file exists before loading
#     if load_weights and os.path.isfile(weight_file_path):
#         # Load the trained weights into the model
#         if weight_file_path.endswith(".pth"):
#             model.load_state_dict(torch.load(weight_file_path))
#         elif weight_file_path.endswith(".ckpt"):
#             model.load_state_dict(torch.load(weight_file_path)['state_dict'])

#     return model

def LoadModel(
        path_to_modelconfig: str,
        path_to_statedict_or_ckpt: str,  
        load_weights: bool = True,
        )-> StandardModel:
    """
    Load model from config and initialise weights, either from ckpt or .pth file.
    """

    # Load model configuration
    model_config = ModelConfig.load(path_to_modelconfig)
    
    # Initialize model with randomly initialized weights
    model = Model.from_config(model_config, trust=True)

    # Check if the weight file exists before loading
    if load_weights:
        if path_to_statedict_or_ckpt.endswith(".pth"):
            # Load the trained weights into the model
            model.load_state_dict(torch.load(path_to_statedict_or_ckpt))
        else:
            model.load_state_dict(torch.load(path_to_statedict_or_ckpt)['state_dict'])

    return model


## used to be in GNN.py:
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

def create_sub_folder(run_name: str, base_folder: str, logger: Logger) -> str:
        """
        Creates a sub-folder for the current run.
        """
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


def copy_selection_databases(config) -> str:
        # Get the root folder for the selection databases from the config, or use the default path.
        selection_database_root_folder = config.get('selection_database_root_folder', '/home/wecapstor3/capn/capn108h/selection_databases/allflavor_classif8_9_19_20_22_23_26_27')

        # Copy selection databases to tmpdir for quicker accessibility.
        tmpdir = os.environ.get("TMPDIR")
        if not tmpdir:
                raise EnvironmentError("TMPDIR environment variable is not set.")

        selection_database_folder = os.path.join(tmpdir, "selection_databases")
        print(f"Copying train, val, and test selection databases from {selection_database_root_folder} to {selection_database_folder} ...")

        # Measure the time taken to copy the files.
        t1 = time()

        # Create the destination directory if it doesn't exist.
        os.makedirs(selection_database_folder, exist_ok=True)

        # Copy the entire directory tree.
        shutil.copytree(src=selection_database_root_folder, dst=selection_database_folder, dirs_exist_ok=True)

        t2 = time()
        copy_time = timedelta(seconds=(t2 - t1))
        print(f"Copying train, val and test selection databases done after {copy_time}.")

        return selection_database_folder


def get_config_and_sub_folder(
              config_path: str, 
              resumefromckpt: bool,
              logger: Logger,
              foldername_allruns: str = 'runs_and_saved_models',
              ) -> Tuple[dict, str]:
        """
        Method to handle some calls that are misplaced in the main method. Determines the config path and sets up the sub folder if necessary.
        Returns the config_path and the sub_folder of the run
        """
        if config_path:
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
                        import re
                        # Regular expression to match the path up to "run_from_*"
                        pattern = r'(.*/(?:run_from_[^/]+|run_jobid[^/]+|run_[^/]+))'

                        # Search for the pattern in the path
                        sub_folder = re.search(pattern, config.get('ckpt_path')).group(1)
                else:
                        sub_folder = get_lastrun_path(foldername_allruns)  
                logger.info(f"Request to resume from checkpoint accepted, using subfolder: {sub_folder} to save files to.")
        else:
                run_name = f"run_jobid_{int(os.environ.get('SLURM_JOB_ID'))}"
                basefolder = config.get('basefolder', '/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_and_saved_models/')
                path_to_check = os.path.join(basefolder, run_name)

                if os.path.exists(path=path_to_check):
                        sub_folder=path_to_check
                        logger.info(f"Subfolder {sub_folder} already exists.")

                else:
                        sub_folder = create_sub_folder(run_name=run_name, base_folder=basefolder, logger=logger) 
                        # Save the YAML config file into subfolder if not resuming from checkpoint
                        save_yaml(config, os.path.join(sub_folder, 'config.yaml'))
         
        return config, sub_folder


def PlottingRoutine(
              results: pd.DataFrame = None, 
              subfolder: str = None, 
              target_label: Union[str, List[str]] = ['deposited_energy'],
              config: dict = None, 
              logger: Logger = None,
              ):
        """
        Function to handle all the plotting. Mainly relies on the methods defined in Analysis.py
        """
        if logger is None:
            logger = Logger()
        
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
        for i in config.get('classifications_to_train_on', [8, 9, 19, 20, 22, 23, 26, 27]):
                Analysis.plot_IQR_and_relerror(dataframe=results, runfolder=subfolder, plot_type='energy', classification=i)
                Analysis.plot_IQR_and_relerror(dataframe=results, runfolder=subfolder, plot_type='r', classification=i)
                Analysis.plot_IQR_and_relerror(dataframe=results, runfolder=subfolder, plot_type='z', classification=i)

def GNNTrainingwithmodelfromckpt():
    """
    Using the load_from_checkpoint method to instantiate a new model instance from a checkpoint outside 
    the context pytorch 'Trainer'.
    Useful for fine tuning and transfer learning. 
    No functionality yet
    """
    return 
