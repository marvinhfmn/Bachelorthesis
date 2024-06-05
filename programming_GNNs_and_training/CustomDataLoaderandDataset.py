from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
import sqlite3 as sql
from glob import glob
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from graphnet.data.dataset import Dataset, SQLiteDataset, ParquetDataset, EnsembleDataset
from graphnet.utilities.logging import Logger
from graphnet.models.graphs import GraphDefinition
from graphnet.training.utils import make_dataloader

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
    """Construct train and test `DataLoader` instances."""
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

def DataloaderSelection(db, classification = 1.0, column = 'classification'):
    indices = []
    #ensure that db is a list
    if isinstance(db, str):
        db = [db]
    
    for dbfile in db:
            con = sql.connect(dbfile)
            cur = con.cursor()
            ids = cur.execute(f"SELECT event_no FROM truth WHERE {column}={classification}")
            ids = ids.fetchall()
            ids = [[dbfile, int(event_id[0])] for event_id in ids]
            indices += ids
    
    dbfilepaths = [path for [path, index] in indices]
    selection_indices = [ids for [path, ids] in indices]
    return dbfilepaths, selection_indices

def GetSelectionIndicesExcludeNULL(db:  Union[str, List[str]], column: str):
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


#Idea: Create a custom dataset as e.g. a subdataset where the classification is specified and 
# the data is split into train, val and test sets
def CreateCustomDatasets(path: Union[str, List[str]], 
                        graph_definition, 
                        features: List[str], 
                        truth: str, 
                        pulsemap: Union[str, List[str]] = 'InIceDSTPulses', 
                        truth_table: str = 'truth',
                        test_size: float = 0.1,
                        # shuffle = False,
                        random_state = 42,
                        INCLUDENULL = False, #NULL entries in databases are abandoned 
                        ):
      
    logger = Logger()
    if(INCLUDENULL):
        logger.info(f"Jokes on you, there is no functionality in setting INCLUDENULL to {INCLUDENULL}, except you don't get a dataset. Nice.")
        return 
    
    
    
    common_kwargs = dict(
                pulsemaps=pulsemap,
                features=features,
                truth=[truth], #truth has to be a list or tupel for line 283 in dataset.py: assert isinstance method to not fail
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
            #Get indices where the training parameter is not NULL
            temp, NOTNULLINDICES = GetSelectionIndicesExcludeNULL(db = singledataset, column=truth) 

            trainval_selection, test_selection = train_test_split(
            NOTNULLINDICES, test_size=test_size, random_state=random_state
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

        # print(training_ensemble, validation_ensemble, testing_ensemble)

        return training_ensemble, validation_ensemble, testing_ensemble
    
    else:
        logger.info("Path is not a string or list of strings ... Execution terminated")
        return
    
    #save dataset configuration doesn't work for ensembleDatasets
    # dataset.config.dump(f"savedDatasets/{datasetname}.yml")
    


if __name__ == "__main__":
    datasetpath = glob("/home/wecapstor3/capn/capn106h/l2_labeled/merged/*.db")
    training_parameter = 'first_vertex_energy'
    
    from graphnet.models.graphs import KNNGraph
    from graphnet.models.detector.icecube import IceCube86
    
    feature_names = ['dom_x', 'dom_y', 'dom_z', 'dom_time', 'charge', 'rde', 'pmt_area', 'hlc']
    graph_definition = KNNGraph(detector = IceCube86())
    mydataset = CreateCustomDatasets(datasetpath[0:3], graph_definition=graph_definition, features=feature_names, truth=training_parameter)
   
    for i in range(0,3):
        print(len(mydataset[i]))