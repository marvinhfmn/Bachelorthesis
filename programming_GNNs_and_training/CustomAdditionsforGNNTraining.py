from typing import Dict, List, Optional, Tuple, Union, Callable
import os
import numpy as np
import pandas as pd
import sqlite3 as sql
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
          db:  Union[str, List[str]], 
          column: str = 'first_vertex_energy', 
          threshold: int = 10, #threshold energy in units of db entries
          ) -> List[float]:
    """
    Selects values of column from the db where they are above a certain threshold and returns a list of these values.
    """
    energy_values = []
    
    # Ensure that db is a list
    if isinstance(db, str):
        db = [db]
    
    for dbfile in db:
        con = sql.connect(dbfile)
        cur = con.cursor()
        query = f"SELECT {column} FROM truth WHERE {column} > {threshold}"
        cur.execute(query)
        values = cur.fetchall()
        energy_values += [float(value[0]) for value in values]
        con.close()
    
    return energy_values
     
#Idea: Create a custom dataset as e.g. a subdataset where the classification is specified and 
# the data is split into train, val and test sets
def CreateCustomDatasets_traininglabelinDataset(
                        path: Union[str, List[str]], 
                        graph_definition, 
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
        else: 
            logger.info("Path is an empty list ... Execution terminated")
            return
        
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
        logger.info("Path is not a string or list of strings ... Execution terminated")
        return
    
    #save dataset configuration doesn't work for ensembleDatasets
    # dataset.config.dump(f"savedDatasets/{datasetname}.yml")

def CreateCustomDatasets_CustomTrainingLabel(
                        path: Union[str, List[str]], 
                        graph_definition, 
                        features: List[str], 
                        truth: Union[str, List[str]], 
                        classifications: Union[int, List[int]] = [8, 9, 19, 20, 22, 23, 26, 27],
                        training_target_label: Label = CustomLabel_depoEnergy(),
                        deposited_energy_cols: List[str] = ["first_vertex_energy", "second_vertex_energy", "third_vertex_energy", "visible_track_energy", "visible_spur_energy"],
                        pulsemap: Union[str, List[str]] = 'InIceDSTPulses', 
                        truth_table: str = 'truth',
                        test_size: float = 0.1,
                        random_state = 42,
                        test_truth: Union[str, List[str]] = [],
                        logger: Logger = None, 
                        threshold: int = 10,
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
    - test_size: Proportion of data to reserve for testing.
    - random_state: Seed for random number generator to ensure reproducibility.
    - test_truth: Optional List of event level truths to add to the graph of only the testing dataset. list elements should be exclusive (no overlap with truth)
    - logger: Optional logger object for logging messages.
    - threshold: Threshold value for deposited energy columns to filter events.

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
        
    trainingdatasets = []
    validationdatasets = []
    testingdatasets = []

    #splitting into train, val, test
    for singledataset in path:
        #Get indices where classification fulfills requirements           
        database, VALIDINDICES = GetSelectionIndices_basedonClassification(
                                            db=singledataset, 
                                            classifications=classifications, 
                                            deposited_energy_cols=deposited_energy_cols,
                                            threshold=threshold
                                            )
        VALIDINDICES = VALIDINDICES[0]

        trainval_selection, test_selection = train_test_split(
        VALIDINDICES, test_size=test_size, random_state=random_state
        )
        training_selection, validation_selection = train_test_split(
        trainval_selection, test_size=test_size, random_state=random_state
        )
        training_dataset = CustomSQLiteDataset(path=singledataset, **trainval_kwargs, selection=training_selection)
        training_dataset.add_label(training_target_label)
        trainingdatasets.append(training_dataset)

        validation_dataset = CustomSQLiteDataset(path=singledataset, **trainval_kwargs, selection=validation_selection)
        validation_dataset.add_label(training_target_label)
        validationdatasets.append(validation_dataset)

        testing_dataset = CustomSQLiteDataset(path=singledataset, **test_kwargs, selection=test_selection)
        testing_dataset.add_label(training_target_label)
        testingdatasets.append(testing_dataset)

    return EnsembleDataset(trainingdatasets), EnsembleDataset(validationdatasets), EnsembleDataset(testingdatasets) 


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
    
class CustomIceCube86(Detector):
    """Custom `Detector` class for IceCube without pmt area."""

    geometry_table_path = os.path.join(
        ICECUBE_GEOMETRY_TABLE_DIR, "icecube86.parquet"
    )
    xyz = ["dom_x", "dom_y", "dom_z"]
    string_id_column = "string"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "dom_x": self._dom_xyz,
            "dom_y": self._dom_xyz,
            "dom_z": self._dom_xyz,
            "dom_time": self._dom_time,
            "charge": self._charge,
            "rde": self._rde,
            "hlc": self._identity,
        }
        return feature_map

    def _dom_xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 500.0

    def _dom_time(self, x: torch.tensor) -> torch.tensor:
        return (x - 1.0e04) / 3.0e4

    def _charge(self, x: torch.tensor) -> torch.tensor:
        return torch.log10(x)

    def _rde(self, x: torch.tensor) -> torch.tensor:
        return (x - 1.25) / 0.25


# if __name__ == "__main__":