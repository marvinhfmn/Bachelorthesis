from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from graphnet.data.dataset import Dataset, SQLiteDataset, ParquetDataset
from graphnet.models.graphs import GraphDefinition
from graphnet.training.utils import make_dataloader
from graphnet.training.labels import Label

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

# if __name__ == "__main__":