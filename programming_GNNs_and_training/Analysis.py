#Script to analyse and plot the results of GNN training 
from typing import Optional, List, Union, Tuple, Dict
import os
from glob import glob
import sqlite3 as sql 
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import iqr
import torch
from termcolor import colored
from tqdm import tqdm
from graphnet.data.dataloader import DataLoader
from graphnet.training.labels import Label
from graphnet.models import StandardModel, Model
from graphnet.utilities.config import ModelConfig
from graphnet.utilities.logging import Logger
import CustomAdditionsforGNNTraining as Custom

# plt.rcParams['text.usetex'] = True #produces errors?

#'Constants'
RUNFOLDER = "/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_and_saved_models"
RUNFOLDER_OLD = "/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_and_saved_models(old_datamergedate20240526)"
TESTRESULTNAME = "test_results.h5"

#load test results
def loadresults_from_subfolder(
        subfolder: str, 
        filename:str = "test_results.h5",
        ) -> pd.DataFrame:
    """
    Method to load the test result .h5 file from the specified subfolder.
    """
    try:
        result_df = pd.read_hdf(os.path.join(subfolder, filename))
    except:
        raise("Couldn't find test results dataframe.")
    return result_df

def calculate_deposited_energy(
        data: pd.DataFrame
        ) -> pd.DataFrame:
    """Calculate the deposited energy by summing relevant columns."""
    energy_columns = [
        'first_vertex_energy', 'second_vertex_energy', 'third_vertex_energy', 
        'visible_track_energy', 'visible_spur_energy'
    ]
    pd.set_option('future.no_silent_downcasting', True)
    # Convert object columns to numeric (this handles the downcasting issue)
    # data[energy_columns] = data[energy_columns].apply(pd.to_numeric, errors='coerce')
    
    # Replace NaN with 0
    data[energy_columns] = data[energy_columns].fillna(0).infer_objects(copy=False)
    
    # Sum over the deposited energy columns
    data['deposited_energy'] = data[energy_columns].sum(axis=1)

    # Ensure deposited_energy is never below zero (relevant for throughgoing(26) and stopping tracks (27))
    data['deposited_energy'] = data['deposited_energy'].clip(lower=0.1)

    # print(data[energy_columns + ['deposited_energy']])
    return data

def load_data_from_databases(
        database_paths: Union[str, List[str]],
        chunksize: int = 5000,
        selected_columns: Union[str, List[str]] = ["classification", "sim_weight"],
        deposited_energy_columns: Union[str, List[str]] = ["first_vertex_energy", "second_vertex_energy", "third_vertex_energy", "visible_track_energy", "visible_spur_energy"], 
        calculate_deposited_energy_onthefly: bool = True,
        ) -> pd.DataFrame:
    """Load data from multiple SQLite databases and concatenate into a single DataFrame."""
    if isinstance(selected_columns, str):
        selected_columns = [selected_columns]
    if isinstance(deposited_energy_columns, str):
        deposited_energy_columns = [deposited_energy_columns]
    
    all_columns = list(set(selected_columns + deposited_energy_columns))
    columns_str = ", ".join(all_columns)

    df_list = []
    for db_path in database_paths:
        conn = sql.connect(db_path)
        query = f"SELECT {columns_str} FROM truth"
        for chunk in pd.read_sql_query(query, conn, chunksize=chunksize):
            if calculate_deposited_energy_onthefly:
                chunk = calculate_deposited_energy(data=chunk)
            df_list.append(chunk)
        conn.close()

    return pd.concat(df_list, ignore_index=True)

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

def evaluateModel(
        model: Union[Model, StandardModel], 
        path_to_datasets: str,
        subfolder: str, 
        filename: str = 'test_results.h5',
        key: str ='test_results',
        gpus: Optional[Union[List[int], int]] = None,
        )-> pd.DataFrame:
    """
    If there is already a test result do nothing else create datasets and evaluate the model on the test dataset, save results
    """
    logger = Logger()
    if os.path.isfile(f"{subfolder}/{filename}"):
        logger.info(f"There is already a {filename} file in the specified folder. Skipping evaluate method.")
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
            test_results.to_hdf(os.path.join(f"{subfolder}", filename), key=key, mode='w')
            logger.info("Succesfully evaluated the model. Results can be found under {}".format(os.path.join(f"{subfolder}", filename)))
        except:
            logger.info("Saving test results unsuccesful.")
        
        return test_results

def get_classification_map() -> Dict[int, str]:

    classification_map = {
        8: 'hadr_cascade',
        9: 'em_hadr_cascade',
        19: 'starting_track',
        20: 'contained_track',
        22: 'double_bang_em',
        23: 'double_bang_hadr',
        26: 'throughgoing_track',
        27: 'stopping_track'
    }
    return classification_map

def plot_resultsashisto(
                test_results: pd.DataFrame, 
                subfolder: str, 
                backbone = "DynEdge", 
                bins = 100, 
                prediction: Union[str, List[str]] = ['energy_pred'], 
                target_label: Union[str, List[str]] = ['deposited_energy'],
                ) -> None:
    """
    Method to plot the difference between the true energy and the reconstructed/predicted energy as a histogramm.
    test_results: pandas dataframe that includes a prediction and a truth column 
    subfolder: Folder to save the histogram to 
    """
    # Map cosmic_primary_type to corresponding labels and titles
    type_labels = {
        12: ("NuE", r"$\nu_e$"),
        14: ("NuMu", r"$\nu_\mu$"),
        16: ("NuTau", r"$\nu_\tau$")
    }
   
    if isinstance(prediction, str):
        prediction = [prediction]
    if isinstance(target_label, str):
        target_label = [target_label]
    # Ensure prediction and target_label are lists with one element
    if len(prediction) != 1 or len(target_label) != 1:
        raise ValueError("prediction and target_label should each be a list with exactly one element.")
    
    prediction_col = prediction[0]
    target_col = target_label[0]

    # Filter out zero values
    zero_mask = (test_results[target_col] == 0) #Create mask for entries that are 0 

    # Check if there are any zero values
    # if zero_mask.any():
    #     print("There are zero values in the first_vertex_energy column.")
    # else:
    #     print("There are no zero values in the first_vertex_energy column.")
    
    filtered_results = test_results[~zero_mask]

    fig, ax = plt.subplots(figsize=(10,6))
    legend_elements = []
    # Define the bins using the range of the differences for the combined data
    log_predictions_all = np.log10(filtered_results[prediction_col])
    log_truth_all = np.log10(filtered_results[target_col])
    differences_all = log_predictions_all - log_truth_all
    bin_edges = np.histogram_bin_edges(differences_all, bins=bins)

    total_counts_from_flavors = np.zeros(len(bin_edges) - 1)

    for pdg_type, (flavor_label, flavor_title) in type_labels.items():
        flavor_mask = np.abs(filtered_results['cosmic_primary_type']) == pdg_type
        flavor_predictions = filtered_results[prediction_col][flavor_mask]
        flavor_truth = filtered_results[target_col][flavor_mask]

        if flavor_predictions.empty: #if empty skip this flavor
            break

        log_predictions = np.log10(flavor_predictions)
        log_truth = np.log10(flavor_truth)

        differences = log_predictions - log_truth

        mean_diff = np.mean(differences)
        std_diff = np.std(differences)

        hist = ax.hist(differences, weights = filtered_results['sim_weight'][flavor_mask], bins=bin_edges, histtype='step', label=f'{flavor_title} (mean={mean_diff:.2f}, std={std_diff:.2f})')
        color = hist[2][0].get_edgecolor()
        # Add legend entry
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10,
                                      label=f'{flavor_title} (mean={mean_diff:.2f}, std={std_diff:.2f})'))

        total_counts_from_flavors += hist[0]  # Sum the counts for each flavor

    if len(abs(filtered_results['cosmic_primary_type']).unique()) > 1: # if there are more than one neutrino flavor plot one histo for all cumulative 
        # Calculate mean and standard deviation for all combined differences
        mean_diff_all = np.mean(differences_all)
        std_diff_all = np.std(differences_all)

        hist_all = ax.hist(differences_all, weights = filtered_results['sim_weight'], bins=bin_edges, histtype='step', label=f'All flavors (mean={mean_diff_all:.2f}, std={std_diff_all:.2f})')
        color_all = hist_all[2][0].get_edgecolor()

        # Add legend entry for all flavors with the same color
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color_all, markersize=10,
                                    label=f'All flavors (mean={mean_diff_all:.2f}, std={std_diff_all:.2f})'))

        # Check if the sum of the counts from individual flavors matches the counts of the combined histogram
        # counts_all_flavors = hist_all[0]
        # if np.array_equal(total_counts_from_flavors, counts_all_flavors):
        #     print("The counts from individual flavors sum up to the counts in the combined histogram.")
        # else:
        #     print("There is a discrepancy between the counts from individual flavors and the combined histogram.")
        #     print(f"Sum of counts from individual flavors: {total_counts_from_flavors}")
        #     print(f"Counts from combined histogram: {counts_all_flavors}")


        plt.text(mean_diff_all + 1, plt.ylim()[1] * 0.9, f'Mean: {mean_diff_all:.2f}', color='red')
        plt.text(mean_diff_all + 2, plt.ylim()[1] * 0.9, f'Std Dev: {std_diff_all:.2f}', color='red')

    plt.axvline(0, ls='--', color="black", alpha=0.5)
    plt.title(f"Training parameter: {target_label}")
    plt.xlabel('Reco. Energy - True Energy [log10 GeV]', size=14)
    plt.ylabel('Amount', size=14)
    # ax.set_yscale('log')
    ax.legend(loc='upper left', handles=legend_elements)
    name = f"EnergypredvsEnergyfromtruth_{backbone}.png"
    try:
        os.makedirs(os.path.join(subfolder, 'plots'), exist_ok=True)
        temp = os.path.join(subfolder, 'plots', name)
        fig.savefig(temp, dpi=300)
        message = colored("SUCCESSFUL", "green") + f': histogram plotted to {temp}'
    except Exception as e:
        message = colored("UNSUCCESSFUL", "red") + f': Could not plot the histogram due to {e}'
    logger = Logger()
    logger.info(message)

def plot_lossesandlearningrate(
        subfolder: str
        ) -> None:
    """
    Method to plot the losses and the learning rate during the GNN training 
    subfolder: Folder to save plots to
    """
    logger = Logger()
    timestamp = subfolder.split("/")[-1]
    metric_files = glob(subfolder + "/losses/version_*/metrics.csv")
    metric_files.sort()

    #merge all metrics in subfolder
    metrics = pd.DataFrame([])
    for file in metric_files:
        metric = pd.read_csv(file, index_col=False)
        metrics = pd.concat([metrics, metric], ignore_index=True)

    metrics.to_csv(subfolder + "/losses/allmetrics.csv", index=False)
    # losses = pd.read_csv(f"{subfolder}/losses/version_0/metrics.csv")

    losses = pd.read_csv(os.path.join(subfolder, 'losses/allmetrics.csv'))

    if "train_loss" in losses.columns and "val_loss" in losses.columns:
        # Drop NaN values but retain the epoch indices
        train_loss = losses[["epoch", "train_loss"]].dropna()
        val_loss = losses[["epoch", "val_loss"]].dropna()

        # Get the corresponding epochs
        train_epochs = train_loss.index
        val_epochs = val_loss.index

        plt.figure(dpi=500)
        plt.plot(train_loss["epoch"], train_loss["train_loss"], marker=".", color="navy", label="training loss")
        plt.plot(val_loss["epoch"], val_loss["val_loss"], marker=".", color="crimson", label="validation loss")
        plt.grid()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(f"Losses \n {timestamp.replace('_', ' ')}")
        plt.legend()
        os.makedirs(os.path.join(subfolder, 'plots'), exist_ok =True)
        temp = os.path.join(subfolder, 'plots', 'Losses.png')
        plt.savefig(temp)
        plt.close()
        message = colored("SUCCESSFUL", "green") + f": losses plotted, saved to {temp}"
        logger.info(message)
    else:
        message = colored("ERROR", "red") + ": train_loss or val_loss not found"
        logger.info(message)

    if "lr" in losses.columns:
        lr = np.array(losses["lr"].dropna())

        plt.figure(dpi=500)
        plt.plot(lr, marker=".", color="crimson")
        plt.grid()
        plt.xlabel("step")
        plt.ylabel("learning rate")
        plt.title(f"learning rate \n {timestamp.replace('_', ' ')}")
        plt.tight_layout()
        os.makedirs(os.path.join(subfolder, 'plots'), exist_ok =True)
        temp = os.path.join(subfolder, 'plots', 'learning_rate.png')
        plt.savefig(temp)
        plt.close()
        message = colored("SUCCESSFUL", "green") + f": learning rate plotted, saved to {temp}"
        logger.info(message)
    else:
        message = colored("ERROR", "red") + ": learning rate not found"
        logger.info(message)
 
#2d histogram plotting true energy vs reconstructed energy 
#normalise every column of the true energy 
def plotEtruevsEreco(
        dataframe: pd.DataFrame, 
        subfolder: str, 
        normalise: Union[str, List[str]] = ['E_true']
        ) -> None:
    """
    Method to plot the true energy vs the reconstructed/predicted energy as an unweighted 2d histogram or heatmap
    dataframe: pandas dataframe that includes a prediction and a truth column 
    subfolder: folder to save plots to
    normalise: parameter to determine in what dimension the data should be normalised
                'E_true': Adds up every E_pred value for a E_true bin and uses that sum to normalise the histogram 
                        -> Interpretation: Probability distribution of what energy is reconstructed for a given true energy  
                'E_reco' or 'E_pred': Adds up every E_true value for a E_pred bin and uses that sum to normalise the histogram 
                        -> Interpretation: Probability distribution of what would be the true energy for a given predicted energy 
                not specification: no normalisation, total counts shown
    """
    if isinstance(normalise, str):
        normalise = [normalise]

    timeofrun = subfolder.split("/")[-1]
    logger = Logger()
    prediction = dataframe.keys()[0]
    truth = dataframe.keys()[1]

    energy_prediction = dataframe[prediction]
    energy_true = dataframe[truth]
    
    bins_Epred = np.geomspace(energy_prediction.min(), energy_prediction.max(), num=100)
    bins_Etrue= np.geomspace(energy_true.min(), energy_true.max(), num=101)
    
    for i in range(len(normalise)):
        fig, ax = plt.subplots()

        #adding line which showcases a perfect match in prediction and truth // identity line
        min_energy = min(energy_true.min(), energy_prediction.min())
        max_energy = max(energy_true.max(), energy_prediction.max())
        ax.plot([min_energy, max_energy], [min_energy, max_energy], 'r--', label='Identity Line')

        if (normalise[i]=='E_true'):
            filename = "2dHisto_EtruevsEreco_unweighted_normalisedalongEtrue.png" 
            #Create 2d histogram
            hist, xedges, yedges = np.histogram2d(energy_true, energy_prediction, bins=[bins_Etrue, bins_Epred])
            #Normalize histogram (columnwise)
            Etrue_sums = hist.sum(axis=1, keepdims=True)
            Etrue_sums[Etrue_sums == 0] = 1  # Prevent division by zero
            hist_normalized = hist / Etrue_sums

            pc = plt.pcolormesh(xedges, yedges, hist_normalized.T, norm=colors.LogNorm(), 
                    cmap='viridis')
            
            # Adding a colorbar
            cbar = plt.colorbar(pc, ax=ax)
            cbar.ax.set_ylabel('Normalized Counts')
            ax.set_title(f"unweighted but normalised (with respect to true energy) plot \n of true energy vs predicted energy \n Training parameter: {truth} \n {timeofrun.replace('_', ' ')}")

        elif (normalise[i]=='E_pred' or normalise[i]=='E_reco'):
            filename = "2dHisto_EtruevsEreco_unweighted_normalisedalongEreco.png" 
            #Create 2d histogram
            hist, xedges, yedges = np.histogram2d(energy_true, energy_prediction, bins=[bins_Etrue, bins_Epred])
            #Normalize histogram (columnwise)
            Ereco_sums = hist.sum(axis=0, keepdims=True)
            Ereco_sums[Ereco_sums == 0] = 1  # Prevent division by zero
            hist_normalized = hist / Ereco_sums

            pc = plt.pcolormesh(xedges, yedges, hist_normalized.T, norm=colors.LogNorm(), 
                    cmap='viridis')
            
            # Adding a colorbar
            cbar = plt.colorbar(pc, ax=ax)
            cbar.ax.set_ylabel('Normalized Counts')
            ax.set_title(f"unweighted but normalised (with respect to reconstructed energy) plot \n of true energy vs predicted energy \n Training parameter: {truth} \n {timeofrun.replace('_', ' ')}")
        else:
            filename = "2dHisto_EtruevsEreco_unweighted_notnormalised.png"

            hist = ax.hist2d(energy_true, energy_prediction, bins=[bins_Etrue, bins_Epred], 
                    norm=colors.LogNorm(), 
                    cmap='viridis'
                    )

            #Adding a colorbar
            cbar = plt.colorbar(hist[3], ax=ax)
            cbar.ax.set_ylabel('Counts')   

            ax.set_title(f"unweighted and not normalised plot of true energy vs predicted energy \n Training parameter: {truth} \n {timeofrun.replace('_', ' ')}")
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r'True Energy $E_{true} [GeV]$')
        ax.set_ylabel(r'Predicted Energy $E_{pred} [GeV]$')

        #adding a grid
        ax.grid(True, which="both", linestyle='--', linewidth=0.5)

        #adding a legend
        ax.legend(loc='upper left')

        #try to save the figure 
        try:             
            os.makedirs(os.path.join(subfolder, 'plots'), exist_ok =True)
            save_path = os.path.join(subfolder, 'plots', filename)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            message = colored("Successful", "green") + f": Saved figure to {save_path}"
            logger.info(message)
        except:
            message = colored("ERROR", "red") + ": Couldn't save figure."
            logger.info(message)

        plt.show()
        plt.close(fig)

def plotEtruedepositedvsEtruecosmicprimary(data, classification, savefolder=RUNFOLDER, bins=100):
    """
    Plot heatmaps of deposited energy vs cosmic primary energy for specified classification.
    """

    flavors = {
        12: [r'$\nu_e, \overline{\nu}_e$', 'Blues'],
        14: [r'$\nu_\mu, \overline{\nu}_\mu$', 'Reds'],
        16: [r'$\nu_\tau, \overline{\nu}_\tau$', 'Greens']
    }

    classification_map = {
        8: 'hadr_cascade',
        9: 'em_hadr_cascade',
        19: 'starting_track',
        20: 'contained_track',
        22: 'double_bang_em',
        23: 'double_bang_hadr',
        26: 'throughgoing_track',
        27: 'stopping_track'
    }

    filtered_data = data[data['classification'] == classification]
    print(f'Length of filtered data: {len(filtered_data)}')

    non_empty_flavors = [(particle_type, (label, cmap)) for particle_type, (label, cmap) in flavors.items() if not filtered_data[np.abs(filtered_data['cosmic_primary_type']) == particle_type].empty]
    amount_of_uniqueentries = len(non_empty_flavors)
    print(f'Amount of unique entries {amount_of_uniqueentries}')
    
    fig, axs = plt.subplots(1, amount_of_uniqueentries, figsize=(18, 6), sharex=True, sharey=True)

    totalcounts = 0

    # Define the global bins for the combined histogram
    global_min_cpe = filtered_data['cosmic_primary_energy'].min()
    global_max_cpe = filtered_data['cosmic_primary_energy'].max()
    global_min_de = filtered_data['deposited_energy'].min()
    global_max_de = filtered_data['deposited_energy'].max()

    print(f'Global min/max cosmic primary energy: {global_min_cpe}/{global_max_cpe}')
    print(f'Global min/max deposited energy: {global_min_de}/{global_max_de}')

    bins_cpe = np.geomspace(global_min_cpe, global_max_cpe, num=bins)
    bins_de = np.geomspace(global_min_de, global_max_de, num=bins)
    
    min_energy = min(global_min_cpe, global_min_de)
    max_energy = max(global_max_cpe, global_max_de)

    colormaps = ['Blues', 'Greens', 'Reds']

    for i, (particle_type, (label, cmap)) in enumerate(non_empty_flavors):
        
        flavor_data = filtered_data[np.abs(filtered_data['cosmic_primary_type']) == particle_type]

        if flavor_data.empty:
            print(f'Flavor data is empty for {particle_type}')
            continue

        cosmic_primary_energy = flavor_data['cosmic_primary_energy'].values
        deposited_energy = flavor_data['deposited_energy'].values

        heatmap, xedges, yedges = np.histogram2d(
            cosmic_primary_energy, deposited_energy, bins=(bins_cpe, bins_de), weights=flavor_data['sim_weight']
        )
        # Check for NaN or infinite values
        if np.any(np.isnan(xedges)) or np.any(np.isnan(yedges)):
            raise ValueError("xedges or yedges contain NaN values")

        if np.any(np.isinf(xedges)) or np.any(np.isinf(yedges)):
            raise ValueError("xedges or yedges contain infinite values")


        assert isinstance(xedges, np.ndarray), "xedges should be a numpy array"
        assert isinstance(yedges, np.ndarray), "yedges should be a numpy array"
    

        totalcounts += heatmap.sum()

        pcm = axs[i].pcolormesh(xedges, yedges, heatmap.T, norm = colors.LogNorm(), cmap=cmap)
        axs[i].set_title(f'Cosmic primary type: {label}')
        fig.colorbar(pcm, ax=axs[i], label='weighted counts')

        axs[i].plot([min_energy, max_energy], [min_energy, max_energy], 'r--', label='Identity Line')
        
        axs[i].set_xscale("log")
        axs[i].set_yscale("log")
        axs[i].set_xlabel('Cosmic Primary Energy')
        axs[i].set_ylabel('Deposited Energy')
        axs[i].legend()
        

    print(f'Total weighted counts: {totalcounts}')

    if classification in classification_map:
        title = f'Heatmaps for classification {classification} ({classification_map[classification]})'
        filename = f'EtruedepovsEtruecep_classif_{classification_map[classification]}'
    else:
        title = f'Heatmaps for classification {classification}'
        filename = f'EtruedepovsEtruecep_classif_{classification}'

    fig.suptitle(title)

    os.makedirs(os.path.join(savefolder, 'plots'), exist_ok=True)
    save_path = os.path.join(savefolder, 'plots', filename)
    plt.savefig(save_path)
    plt.show()

#Calculate Quantiles (eg middle 68% qunatile) with scipy.stats.iqr for the reconstructed energies (one iqr per bin) and plot iqr vs true energy
#iqr: width of the E reco distribution for a E true bin
def plotIQRvsEtrue(
        dataframe: pd.DataFrame, 
        subfolder:str, 
        scale: Union[str, List[str]] = ['linear', 'log'], 
        bins: int = 30,
        cosmic_primary_type: int = -1,
        ) -> None: 
    """
    Method to plot the interquartile range vs the true energy.
    The IQR (here middle 68 percentile) is here the width of the E_reco distribution for a true energy bin and acts as a 
    measure for reconstruction uncertainty.
    possible plots are looking at the IQR in logspace (applied the logarithmn before calculating the IQR) or 
    in linear space (IQR calculated for data without logarithmn)
    Binning for true energy is calculated as np.linspace(rounded_log_Etrue_min, rounded_log_Etrue_max, bins) either way 
    because that concerns the other axis 
    
    Args:
        dataframe (pd.DataFrame): Input dataframe with predictions and true values.
        subfolder (str): Directory to save the plots.
        scale (Union[str, List[str]]): Scale to use for the plot ('linear' and/or 'log').
        bins (int): Number of bins to use for the plot.
        cosmic_primary_type (int): Cosmic primary type to filter by (absolute value).
    """
    # Map cosmic_primary_type to corresponding labels and titles
    type_labels = {
        -1: ('allflavors', 'all flavors'),
        12: ("NuE", r"$\nu_e$"),
        14: ("NuMu", r"$\nu_\mu$"),
        16: ("NuTau", r"$\nu_\tau$")
    }
    
    type_label, type_title = type_labels.get(cosmic_primary_type, ("allflavors", "all flavors"))
    #make sure scale is a list to iterate over it
    if isinstance(scale, str):
        scale = [scale]
        
    prediction = dataframe.keys()[0] #should be energy_pred
    truth = dataframe.keys()[1] #should be training target label


    if cosmic_primary_type == -1:
        energy_prediction = dataframe[prediction]
        energy_true = dataframe[truth]
    else:
        filtered_dataframe = dataframe[dataframe['cosmic_primary_type'].abs().isin([cosmic_primary_type])]
        if filtered_dataframe.empty:
            return
        # print(filtered_dataframe)
        energy_prediction = filtered_dataframe[prediction]
        energy_true = filtered_dataframe[truth]


    #Get the logarithmic values for the enrgy prediction and truth 
    energy_prediction_log = np.log10(energy_prediction)
    energy_true_log = np.log10(energy_true)
    
    #min value, max value -> define bins
    rounded_log_Etrue_min = np.floor(min(energy_true_log))
    rounded_log_Etrue_max = np.ceil(max(energy_true_log))
    e_log_bins = np.linspace(rounded_log_Etrue_min, rounded_log_Etrue_max, bins)
    # Calculate the middle points of the bins for plotting
    e_log_bin_centers = (e_log_bins[:-1] + e_log_bins[1:]) / 2
    
    for s in scale:
        if s =='log':
            #in logspace:
            fig, ax = plt.subplots(figsize=(10, 6))
            all_iqr = []
            # Calculate the IQR for each bin
            for i in range(len(e_log_bins) - 1):
                e_lower, e_upper = e_log_bins[i], e_log_bins[i + 1]
                
                # Mask to select reconstructed energies within the current bin of true energy
                events_mask = (energy_true_log > e_lower) & (energy_true_log < e_upper)
                
                # Calculate IQR for the reconstructed energies in the current bin
                if np.any(events_mask):
                    iqr_value = iqr(energy_prediction_log[events_mask], rng=(16, 84))
                    all_iqr.append(iqr_value)
                else:
                    all_iqr.append(np.nan)  # In case there are no events in the bin, append NaN

            plt.plot(e_log_bin_centers, all_iqr, marker='o', ls='', color='b') #TODO: map color to neutrino flavor 
            plt.xlabel('True Energy (log GeV)')
            plt.ylabel('IQR of Reconstructed Energy (log GeV)')
            plt.title(f'IQR of Reconstructed Energy vs True Energy for {type_title}')
            plt.grid()
            plt.tight_layout()
            filename = f'IQRvsEtrue_logarithmicEreco_{type_label}_{bins}bins'
            os.makedirs(os.path.join(subfolder, 'plots'), exist_ok =True)
            save_path = os.path.join(subfolder, 'plots', filename)
            plt.savefig(save_path)
            plt.show()

        elif s =='linear':      
            fig, ax = plt.subplots(figsize=(10, 6))     
            #linear in y axis and divided by E_true
            all_iqr = []
            for i in range(len(e_log_bins) - 1):
                e_lower, e_upper = e_log_bins[i], e_log_bins[i + 1]
                
                # Mask to select reconstructed energies within the current bin of true energy
                events_mask = (energy_true_log > e_lower) & (energy_true_log < e_upper)
                
                # Calculate IQR for the reconstructed energies in the current bin
                if np.any(events_mask):
                    iqr_value = iqr(energy_prediction[events_mask], rng=(16, 84))
                    all_iqr.append(iqr_value)
                else:
                    all_iqr.append(np.nan)  # In case there are no events in the bin, append NaN

            
            plt.plot(e_log_bin_centers, all_iqr/np.power(10, e_log_bin_centers), marker='o', ls='', color='b')
            plt.xlabel('True Energy (log GeV)')
            plt.ylabel('IQR of Reconstructed Energy(GeV)/ True Energy at bin center(GeV) [arb. unit]')
            plt.title(f'normalized IQR of Reconstructed Energy vs True Energy for {type_title}')
            plt.grid()
            plt.tight_layout()
            filename = f'IQRvsEtrue_linearEreco_{type_label}_{bins}bins_normalized'
            os.makedirs(os.path.join(subfolder, 'plots'), exist_ok =True)
            save_path = os.path.join(subfolder, 'plots', filename)
            plt.savefig(save_path)
            plt.show()

def plot_iqr_vs_true_energy(
    filtered_dataframe: pd.DataFrame, 
    energy_prediction: np.ndarray, 
    energy_true: np.ndarray, 
    runfolder: str, 
    type_title: str, 
    type_label: str, 
    bins: int,
    classification: int,
    scale: List[str] = ['linear'], 
) -> None:
    """
    Plot the interquartile range (IQR) of reconstructed energy vs true energy.

    Args:
        filtered_dataframe (pd.DataFrame): Input dataframe with predictions and true values.
        energy_prediction (np.ndarray): Predicted energy values.
        energy_true (np.ndarray): True energy values.
        subfolder (str): Directory to save the plots.
        type_title (str): Title describing the type of particle or event.
        type_label (str): Label describing the type of particle or event.
        scale (List[str]): List of scales to use for the plot ('linear' and/or 'log').
        bins (int): Number of bins to use for the plot.
    """
    classification_map = get_classification_map()
    class_name = classification_map.get(classification, "unknown")
    if classification ==-1:
        class_name = "all_considered_class"
    
    energy_prediction_log = np.log10(energy_prediction)
    energy_true_log = np.log10(energy_true)
    
    rounded_log_Etrue_min = np.floor(min(energy_true_log))
    rounded_log_Etrue_max = np.ceil(max(energy_true_log))
    e_log_bins = np.linspace(rounded_log_Etrue_min, rounded_log_Etrue_max, bins)
    e_log_bin_centers = (e_log_bins[:-1] + e_log_bins[1:]) / 2
    
    for s in scale:
        fig, ax = plt.subplots(figsize=(10, 6))
        all_iqr = []
        for i in range(len(e_log_bins) - 1):
            e_lower, e_upper = e_log_bins[i], e_log_bins[i + 1]
            events_mask = (energy_true_log > e_lower) & (energy_true_log < e_upper)
            
            if np.any(events_mask):
                if s == 'log':
                    iqr_value = iqr(energy_prediction_log[events_mask], rng=(16, 84))
                else:
                    iqr_value = iqr(energy_prediction[events_mask], rng=(16, 84))
                    iqr_value /= np.power(10, e_log_bin_centers[i])
                all_iqr.append(iqr_value)
            else:
                all_iqr.append(np.nan)

        plt.plot(e_log_bin_centers, all_iqr, marker='o', ls='', color='b')
        plt.xlabel('True Energy (log GeV)')
        ylabel = 'IQR of Reconstructed Energy (log GeV - log GeV)' if s == 'log' else 'IQR of Reconstructed Energy (GeV) / True Energy (GeV)'
        plt.ylabel(ylabel)
        plt.title(f'IQR of Reconstructed Energy vs True Energy for {type_title} ({class_name})')
        plt.grid()
        plt.tight_layout()
        filename = f'IQRvsEtrue_{s}_Ereco_{type_label}_{class_name}_{bins}bins'
        os.makedirs(os.path.join(runfolder, 'plots', 'iqr'), exist_ok=True)
        save_path = os.path.join(runfolder, 'plots', 'iqr', filename)
        plt.savefig(save_path)
        plt.show()
        plt.close()

def plot_iqr_vs_variable(
    filtered_dataframe: pd.DataFrame, 
    energy_prediction: np.ndarray,  
    energy_true: np.ndarray, 
    variable: np.ndarray, 
    variable_name: str, 
    runfolder: str, 
    type_title: str, 
    type_label: str, 
    scale: List[str], 
    bins: int, 
    classification: int = -1,
    normalize: bool = False,
) -> None:
    """
    Plot the interquartile range (IQR) of reconstructed energy vs a given variable.

    Args:
        filtered_dataframe (pd.DataFrame): Input dataframe with predictions and true values.
        energy_prediction (np.ndarray): Predicted energy values.
        variable (np.ndarray): Variable to plot against.
        variable_name (str): Name of the variable.
        subfolder (str): Directory to save the plots.
        type_title (str): Title describing the type of particle or event.
        type_label (str): Label describing the type of particle or event.
        scale (List[str]): List of scales to use for the plot ('linear' and/or 'log').
        bins (int): Number of bins to use for the plot.
        normalize (bool): Whether to normalize the IQR by the true energy 
            (bin center value calculated like plot_iqr_vs_true_energy for comparison purposes)

        method no longer in use
    """
    classification_map = get_classification_map()
    class_name = classification_map.get(classification, "unknown")
    if classification ==-1:
        class_name = "all_considered_class"

    # print(type(variable))
    # print(variable)
    min_val = np.floor(min(variable))
    max_val = np.ceil(max(variable))
    bins_edges = np.linspace(min_val, max_val, bins)
    bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
    
    for s in scale:
        if s == 'log' and normalize:
            raise ValueError("Normalization with log scale is not allowed.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        all_iqr = []
        for i in range(len(bins_edges) - 1):
            lower, upper = bins_edges[i], bins_edges[i + 1]
            events_mask = (variable > lower) & (variable < upper)
            
            if np.any(events_mask):
                if s == 'log':
                    iqr_value = iqr(np.log10(energy_prediction[events_mask]), rng=(16, 84))
                else:
                    iqr_value = iqr(energy_prediction[events_mask], rng=(16, 84))
                if normalize:
                    energy_true_log = np.log10(energy_true)

                    rounded_log_Etrue_min = np.floor(min(energy_true_log))
                    rounded_log_Etrue_max = np.ceil(max(energy_true_log))
                    e_log_bins = np.linspace(rounded_log_Etrue_min, rounded_log_Etrue_max, bins)
                    e_log_bin_centers = (e_log_bins[:-1] + e_log_bins[1:]) / 2
                    all_iqr.append(iqr_value / np.power(10, e_log_bin_centers[i]))
                else:
                    all_iqr.append(iqr_value)
            else:
                all_iqr.append(np.nan)

        plt.plot(bin_centers, all_iqr, marker='o', ls='', color='b')
        plt.xlabel(f'true vertex {variable_name} (m)')

        if normalize:
            ylabel = 'IQR of Reconstructed Energy (GeV) / True Energy (GeV)'
        else:
            ylabel = 'IQR of Reconstructed Energy'

            if s == 'log':
                ylabel += ' (log GeV - log GeV)'
            else:
                ylabel += ' (GeV)'
        plt.ylabel(ylabel)
        ax.set_yscale('log')

        title = f'IQR of Reconstructed Energy vs true vertex {variable_name} for {type_title} ({class_name})'
        if normalize:
            title += " (normalized to true energy)"
        plt.title(title)
        plt.grid()
        plt.tight_layout()
        filename = f'IQRvs{variable_name}_{type_label}_{class_name}_{bins}bins_{s}'
        if normalize:
            filename += "_normalized"
        os.makedirs(os.path.join(runfolder, 'plots', 'iqr'), exist_ok=True)
        save_path = os.path.join(runfolder, 'plots', 'iqr', filename)
        plt.savefig(save_path)
        plt.show()
        plt.close()

def plot_relerror_vs_variable(
    filtered_dataframe: pd.DataFrame, 
    energy_prediction: np.ndarray,  
    energy_true: np.ndarray, 
    variable: np.ndarray, 
    variable_name: str, 
    runfolder: str, 
    type_title: str, 
    type_label: str, 
    bins: int, 
    classification: int = -1,
) -> None:
    """
    Plot the interquartile range (IQR) of reconstructed energy vs a given variable.

    Args:
        filtered_dataframe (pd.DataFrame): Input dataframe with predictions and true values.
        energy_prediction (np.ndarray): Predicted energy values.
        variable (np.ndarray): Variable to plot against.
        variable_name (str): Name of the variable.
        subfolder (str): Directory to save the plots.
        type_title (str): Title describing the type of particle or event.
        type_label (str): Label describing the type of particle or event.
        scale (List[str]): List of scales to use for the plot ('linear' and/or 'log').
        bins (int): Number of bins to use for the plot.
        normalize (bool): Whether to normalize the IQR by the true energy 
            (bin center value calculated like plot_iqr_vs_true_energy for comparison purposes)
    """
    classification_map = get_classification_map()
    class_name = classification_map.get(classification, "unknown")
    if classification ==-1:
        class_name = "all_considered_class"

    # print(type(variable))
    # print(variable)
    min_val = np.floor(min(variable))
    max_val = np.ceil(max(variable))
    bins_edges = np.linspace(min_val, max_val, bins)
    bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    all_relerror = []
    for i in range(len(bins_edges) - 1):
        lower, upper = bins_edges[i], bins_edges[i + 1]
        events_mask = (variable > lower) & (variable < upper)
        
        if np.any(events_mask):
            relerror_values = (np.abs(energy_prediction[events_mask]- energy_true[events_mask])/energy_true[events_mask])
            all_relerror.append(np.mean(relerror_values))
        else:
            all_relerror.append(np.nan)

    plt.plot(bin_centers, all_relerror, marker='o', ls='', color='b')
    plt.xlabel(f'true vertex {variable_name} (m)')
    ylabel = 'relative error of reconstructed deposited Eenergy'
    plt.ylabel(ylabel)
    ax.set_yscale('log')

    title = f'relative error of reconstructed deposited energy vs true vertex {variable_name} for {type_title} ({class_name})'
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    filename = f'IQRvs{variable_name}_{type_label}_{class_name}_{bins}bins'
    os.makedirs(os.path.join(runfolder, 'plots', 'relerror'), exist_ok=True)
    save_path = os.path.join(runfolder, 'plots', 'relerror', filename)
    plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_IQR_and_relerror(
    dataframe: pd.DataFrame, 
    runfolder: str, 
    plot_type: str = 'energy',
    bins: int = 30, 
    cosmic_primary_type: int = -1,
    classification: int = -1, 
) -> None:
    """
    Plot the interquartile range (IQR, here middle 68 percentile) vs the true energy or the relative error vs true vertex position in polar coordinates.
    Calls plot_relerror_vs_variable or plot_iqr_vs_true_energy for actual plotting functionality 
    
    Args:
        dataframe (pd.DataFrame): Input dataframe with predictions and true values.
        subfolder (str): Directory to save the plots.
        plot_type (str): Type of plot ('energy', 'r', 'z').
        bins (int): Number of bins to use for the plot.
        cosmic_primary_type (int): Cosmic primary type to filter by (absolute value).
    """
    type_labels = {
        -1: ('allflavors', 'all flavors'),
        12: ("NuE", r"$\nu_e$"),
        14: ("NuMu", r"$\nu_\mu$"),
        16: ("NuTau", r"$\nu_\tau$")
    }
    
    type_label, type_title = type_labels.get(cosmic_primary_type, ("allflavors", "all flavors"))
        
    prediction = dataframe.keys()[0]  # Should be energy_pred
    truth = dataframe.keys()[1]  # Should be training target label

    # Filter by classification 
    if classification != -1:
        dataframe = dataframe[dataframe['classification'] == classification]


    if cosmic_primary_type == -1:
        filtered_dataframe = dataframe
        energy_prediction = dataframe[prediction]
        energy_true = dataframe[truth]
        vertex_x = dataframe['first_vertex_x']
        vertex_y = dataframe['first_vertex_y']
        vertex_z = dataframe['first_vertex_z']
    else:
        filtered_dataframe = dataframe[dataframe['cosmic_primary_type'].abs().isin([cosmic_primary_type])]
        if filtered_dataframe.empty:
            return
        energy_prediction = filtered_dataframe[prediction]
        energy_true = filtered_dataframe[truth]
        vertex_x = filtered_dataframe['first_vertex_x']
        vertex_y = filtered_dataframe['first_vertex_y']
        vertex_z = filtered_dataframe['first_vertex_z']

    vertex_r = np.sqrt(vertex_x**2 + vertex_y**2)
    
    if plot_type == 'energy':
        plot_iqr_vs_true_energy(filtered_dataframe, energy_prediction, energy_true, runfolder, type_title, type_label, bins, classification)
    if plot_type == 'r':
        plot_relerror_vs_variable(filtered_dataframe, energy_prediction, energy_true, vertex_r, 'r', runfolder, type_title, type_label, bins, classification)
    elif plot_type == 'z':
        plot_relerror_vs_variable(filtered_dataframe, energy_prediction, energy_true, vertex_z, 'z', runfolder, type_title, type_label, bins, classification)

def AnalyiseDatasetEnergyStructure(
        db: Union[str, List[str]],
        label: str = 'deposited_energy',
        flavor: str = 'All', 
        threshold: float = 10,
        bins: int = 10,
        savefolder: str = RUNFOLDER,
        dbversion: str = 'unknown',
        scale: str = 'log'
)-> None:
    """
    Plots the structure of a given dataset or datasets as a histogram.
    """
    energy_values = np.log10(Custom.GetTrainingLabelEntries(db, column=label, threshold=threshold))
    # print(energy_values)

    filenames = [os.path.basename(path) for path in db]
    fig, ax = plt.subplots()
    # Plot the histogram
    plt.hist(energy_values, bins=bins, edgecolor='black')
    plt.title(f"Histogram of {label} for {flavor} \n {', '.join(filenames)}")
    plt.xlabel(f'{label} (log10 GeV)')
    plt.ylabel('Counts')
    if scale=='log':
        ax.set_yscale('log')
        filename = f"EnergyStructure_for_{label}_{flavor}_{'_'.join([filename.split('_')[0] for filename in filenames])}_v{dbversion}_logscale"
    else:
        filename = f"EnergyStructure_for_{label}_{flavor}_{'_'.join([filename.split('_')[0] for filename in filenames])}_v{dbversion}"
        
    os.makedirs(os.path.join(savefolder, 'plots'), exist_ok=True)
    save_path = os.path.join(savefolder, 'plots', filename)
    plt.savefig(save_path)
    plt.show()
    plt.close()

def GetDatabasePaths(dbversion: str = 'newest')-> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Temporary method to get the datasetpaths of the old datsets (mergedate 26.05.24) and split them into flavors
    """
    #Paths
    if dbversion == 'newest':
        datasetpaths = glob("/home/wecapstor3/capn/capn106h/l2_labeled/merged/*.db")
    else:
        datasetpaths = glob("/home/wecapstor3/capn/capn108h/l2_labeled/merged/*.db")
    datasetpaths.sort()

    NuE_datasetIDs = {'22614', '22613', '22612'}
    NuEfiles = [path for path in datasetpaths if (path.split("/")[-1]).split("_")[0] in NuE_datasetIDs]

    NuMu_datasetIDs = {'22646', '22645', '22644'}
    NuMufiles = [path for path in datasetpaths if (path.split("/")[-1]).split("_")[0] in NuMu_datasetIDs]

    NuTau_datasetIDs = {'22633', '22634', '22635'}
    NuTaufiles = [path for path in datasetpaths if (path.split("/")[-1]).split("_")[0] in NuTau_datasetIDs]

    Corsika_datasetIDs = {'22615'}
    CorsikaSimfiles = [path for path in datasetpaths if (path.split("/")[-1]).split("_")[0] in Corsika_datasetIDs]
    return NuEfiles, NuMufiles, NuTaufiles, CorsikaSimfiles

def DatasetEnergyStructureAnalysisCalls(dbversion: str = 'new', scale: str='log') -> None:
    """
    Plotting calls for the Structure of the datasets
    """
    NuEfiles, NuMufiles, NuTaufiles, CorsikaSimfiles = GetDatabasePaths(dbversion)

    # Determine the flavor for each set of files
    file_groups = [
        (NuEfiles, 'NuE'),
        (NuMufiles, 'NuMu'),
        (NuTaufiles, 'NuTau')
    ]
    # print(type(file_groups))
    cpe = 'cosmic_primary_energy'
    for files, flavor in file_groups:
        AnalyiseDatasetEnergyStructure(files, bins=100, flavor=flavor, dbversion=dbversion, scale=scale)
        AnalyiseDatasetEnergyStructure(files, label=cpe, bins=100, flavor=flavor, dbversion=dbversion, scale=scale)

def PlottingCalls():
    sub_folder = '/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_and_saved_models/run_from_2024.07.11_16:34:51_UTC'
    df = loadresults_from_subfolder(subfolder=sub_folder)
    # plotEtruevsEreco(df, subfolder=sub_folder, normalise=['E_true', 'E_reco', 'notnorm'])
    # plot_lossesandlearningrate(sub_folder)
    # plot_resultsashisto(df, subfolder=sub_folder, target_label=['deposited_energy'])
    classifications = [8, 9, 19, 20, 22, 23, 26, 27]

    for i in classifications:
        plot_IQR_and_relerror(dataframe=df, runfolder=sub_folder, plot_type='energy', classification=i)
        plot_IQR_and_relerror(dataframe=df, runfolder=sub_folder, plot_type='r', classification=i)
        plot_IQR_and_relerror(dataframe=df, runfolder=sub_folder, plot_type='z', classification=i)  

def main():
    PlottingCalls()
    quit()
    NuEfiles, NuMufiles, NuTaufiles, temp = GetDatabasePaths(dbversion='newest')
    all_databasepaths =  NuEfiles + NuMufiles + NuTaufiles + temp

    classifications = [8, 9, 19, 20, 22, 23, 26, 27]

    subfolder = '/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_and_saved_models(merdedate_20240612)/run_from_2024.07.01_19:40:10_UTC'

    model = LoadModel(
        path_to_modelconfig='/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_and_saved_models(merdedate_20240612)/run_from_2024.07.01_19:40:10_UTC/GNN_DynEdge_mergedNuE_NuMu_NuTau.yml',
        path_to_statedict_or_ckpt='/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_and_saved_models(merdedate_20240612)/run_from_2024.07.01_19:40:10_UTC/GNN_DynEdge_mergedNuE_NuMu_NuTau.pth',
        )
    df = evaluateModel(model=model, path_to_datasets=all_databasepaths, 
                  subfolder=subfolder,
                   filename= 'test_result_withclassific.h5')
    
    plotfolder = '/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_and_saved_models(merdedate_20240612)/run_from_2024.07.01_19:40:10_UTC/plots'


if __name__ == "__main__":
        main()  