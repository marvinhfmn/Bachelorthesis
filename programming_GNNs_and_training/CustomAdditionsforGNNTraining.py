from typing import Dict, List, Optional, Tuple, Union, Callable
import os
import numpy as np
import pandas as pd
import sqlite3 as sql
import h5py
import torch
from torch_geometric.data import Data
from glob import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from graphnet.data.dataset import Dataset, SQLiteDataset, ParquetDataset, EnsembleDataset
from graphnet.utilities.logging import Logger
from graphnet.models.graphs import GraphDefinition
from graphnet.training.utils import make_dataloader
from graphnet.training.labels import Label
from graphnet.models.detector.detector import Detector
from graphnet.constants import ICECUBE_GEOMETRY_TABLE_DIR
from CustomSQL_dataset import CustomSQLiteDataset 


class CustomLabel_depoEnergy(Label):
    """Class for producing my label."""
    def __init__(
            self,
            key: str = 'deposited_energy',
            first_vertex_energy: str = 'first_vertex_energy', 
            second_vertex_energy: str = 'second_vertex_energy', 
            third_vertex_energy: str = 'third_vertex_energy', 
            visible_track_energy: str = 'visible_track_energy',
            visible_spur_energy: str = 'visible_spur_energy',
            ):
        """Construct `deposited_energy` label."""
        self._firstve = first_vertex_energy
        self._secondve = second_vertex_energy
        self._thirdve = third_vertex_energy
        self._vte = visible_track_energy
        self._vse = visible_spur_energy

        # Base class constructor
        super().__init__(key=key)

    def __call__(self, graph: Data) -> torch.tensor:
        """Compute label for `graph`."""
        vertex_energies = torch.nan_to_num(graph[self._firstve]) + torch.nan_to_num(graph[self._secondve]) + torch.nan_to_num(graph[self._thirdve])
        tracks_and_spurs = torch.nan_to_num(graph[self._vte]) + torch.nan_to_num(graph[self._vse])
        deposited_energy = vertex_energies+tracks_and_spurs
        return deposited_energy

class renormalized_sim_weight(Label):
    """Custom label: renormalized sim_weight"""
    def __init__(
            self,
            db_size: int,
            sel_size: int,
            sim_weight: str = "sim_weight",
            key: str = "renorm_sim_weight",
            ):

        self._db_size = db_size
        self._sel_size = sel_size
        self._sim_weight = sim_weight

        # Base class constructor
        super().__init__(key=key)

    def __call__(self, graph: Data) -> torch.tensor:
        """Compute label for `graph`."""
        renorm_sim_weight = graph[self._sim_weight] * self._db_size / self._sel_size
        return renorm_sim_weight
    
def make_train_validation_test_dataloader(
    db: str,
    graph_definition: GraphDefinition,
    selection: Optional[List[int]],
    pulsemaps: Union[str, List[str]],
    features: List[str],
    truth: List[str],
    *,
    batch_size: int,
    database_indices: Optional[List[int]] = None,
    seed: int = 42,
    test_size: float = 0.1,
    num_workers: int = 10,
    persistent_workers: bool = True,
    node_truth: Optional[str] = None,
    truth_table: str = "truth",
    node_truth_table: Optional[str] = None,
    string_selection: Optional[List[int]] = None,
    loss_weight_column: Optional[str] = None,
    loss_weight_table: Optional[str] = None,
    index_column: str = "event_no",
    labels: Optional[Dict[str, Callable]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Construct train, validation and test `DataLoader` instances."""
    # Reproducibility
    rng = np.random.default_rng(seed=seed)
    # Checks(s)
    if isinstance(pulsemaps, str):
        pulsemaps = [pulsemaps]

    if selection is None:
        # If no selection is provided, use all events in dataset. 
        dataset: Dataset
        if db.endswith(".db"):
            dataset = SQLiteDataset(
                path=db,
                graph_definition=graph_definition,
                pulsemaps=pulsemaps,
                features=features,
                truth=truth,
                truth_table=truth_table,
                index_column=index_column,
            )
        else:
            dataset = ParquetDataset(
                path=db,
                graph_definition=graph_definition,
                pulsemaps=pulsemaps,
                features=features,
                truth=truth,
                truth_table=truth_table,
                index_column=index_column,
            )
        selection = dataset._get_all_indices()

    # Perform train/validation split
    if isinstance(db, list):
        df_for_shuffle = pd.DataFrame(
            {"event_no": selection, "db": database_indices}
        )
        shuffled_df = df_for_shuffle.sample(
            frac=1, replace=False, random_state=rng
        )
        trainval_df, test_df = train_test_split(
            shuffled_df, test_size=test_size, random_state=seed
        )
        training_df, validation_df = train_test_split(
            trainval_df, test_size=test_size, random_state=seed
        )
        training_selection = training_df.values.tolist()
        validation_selection = validation_df.values.tolist()
        test_selection = test_df.values.tolist()
    else:
        trainval_selection, test_selection = train_test_split(
            selection, test_size=test_size, random_state=seed
        )
        training_selection, validation_selection = train_test_split(
            trainval_selection, test_size=test_size, random_state=seed
        )

    # Create DataLoaders
    common_kwargs = dict(
        db=db,
        pulsemaps=pulsemaps,
        features=features,
        truth=truth,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        node_truth=node_truth,
        truth_table=truth_table,
        node_truth_table=node_truth_table,
        string_selection=string_selection,
        loss_weight_column=loss_weight_column,
        loss_weight_table=loss_weight_table,
        index_column=index_column,
        labels=labels,
        graph_definition=graph_definition,
    )

    training_dataloader = make_dataloader(
        shuffle=True,
        selection=training_selection,
        **common_kwargs,  # type: ignore[arg-type]
    )

    validation_dataloader = make_dataloader(
        shuffle=False,
        selection=validation_selection,
        **common_kwargs,  # type: ignore[arg-type]
    )
    
    test_dataloader = make_dataloader(
        shuffle=False,
        selection=test_selection,
        **common_kwargs,  # type: ignore[arg-type]
    )

    return (
        training_dataloader,
        validation_dataloader,
        test_dataloader,
    )

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
                        training_target_label: Label = CustomLabel_depoEnergy(),
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
                        training_target_label: Label = CustomLabel_depoEnergy(),
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

    training_ensemble = EnsembleDataset(trainingdatasets)
    print(f"Length training ensemble: {len(training_ensemble)}")

    validation_ensemble = EnsembleDataset(validationdatasets)
    print(f"Length validation ensemble: {len(validation_ensemble)}")

    testing_ensemble = EnsembleDataset(testingdatasets)
    print(f"Length testing ensemble: {len(testing_ensemble)}")

    return training_ensemble, validation_ensemble, testing_ensemble

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

# if __name__ == "__main__":