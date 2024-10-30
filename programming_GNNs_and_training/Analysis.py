#Script to analyse and plot the results of GNN training 
from typing import Optional, List, Union, Tuple, Dict
import os
from glob import glob
import sqlite3 as sql 
from matplotlib import colors
import matplotlib.pyplot as plt

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

import matplotlib as mpl
# Define custom colors
custom_colors = ['#332288', '#88CCEE', '#44AA99', '#117733', 
                 '#999933', '#DDCC77', '#CC6677', '#882255', 
                 '#AA4499']

# custom_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
#                 '#ff7f00', '#ffff33', '#a65628', '#f781bf']
# Set the default color cycle
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=custom_colors)

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
import utilities as utils
import CustomAdditionsforGNNTraining as Custom

# plt.rcParams['text.usetex'] = True #produces errors?

#'Constants'
RUNFOLDER = "/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_and_saved_models"
# RUNFOLDER_OLD = "/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_and_saved_models(old_datamergedate20240526)"
TESTRESULTNAME = "test_results.h5"

#load test results
def loadresults_from_subfolder(
        subfolder: str, 
        filename: str = "test_results.h5",
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
        -1: 'all_considered_classes',
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
                yscale: str = 'linear',
                ) -> None:
    """
    Method to plot the difference between the true deposited energy and the reconstructed/predicted energy as a histogramm.
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

        # Get the color cycle
        colors = [color for color in plt.rcParams['axes.prop_cycle'].by_key()['color']]
        if pdg_type == 12:  # Assuming you want to use the first color for pdg_type 11
            hist_color = colors[2]  # Use the first color
        elif pdg_type == 14:
            hist_color = colors[0]
        elif pdg_type == 16:
            hist_color = colors[4]

        hist = ax.hist(
            differences, 
            weights = filtered_results['sim_weight'][flavor_mask], 
            bins=bin_edges, 
            histtype='step', 
            label=f'{flavor_title} (mean={np.round(mean_diff, decimals=2):.2f}, std={np.round(std_diff, decimals=2):.2f})',
            color = hist_color,
        )
        color = hist[2][0].get_edgecolor()
        # Add legend entry
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10,
                                      label=f'{flavor_title} (mean={mean_diff:.2f}, std={std_diff:.2f})'))

        total_counts_from_flavors += hist[0]  # Sum the counts for each flavor

    if len(abs(filtered_results['cosmic_primary_type']).unique()) > 1: # if there are more than one neutrino flavor plot one histo for all cumulative 
        # Calculate mean and standard deviation for all combined differences
        mean_diff_all = np.mean(differences_all)
        std_diff_all = np.std(differences_all)

        hist_all = ax.hist(
            differences_all, 
            weights = filtered_results['sim_weight'], 
            bins=bin_edges, 
            histtype='step', 
            label=f'All flavors (mean={np.round(mean_diff_all, decimals=2):.2f}, std={np.round(std_diff_all, decimals=2):.2f})',
            # color = custom_colors[0],
            color = '#88223d',
            alpha = 0.8,
        )
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


        # plt.text(mean_diff_all + 1, plt.ylim()[1] * 0.9, f'Mean: {mean_diff_all:.2f}', color='red')
        # plt.text(mean_diff_all + 2, plt.ylim()[1] * 0.9, f'Std Dev: {std_diff_all:.2f}', color='red')

    plt.axvline(0, ls='--', color="black", alpha=0.5)
    #plt.title(f"weighted histogram \n Training parameter: {target_label}")
    plt.xlabel(r'$log_{10}(E_{pred})$ - $log_{10}(E_{true})$ [arb. unit]', size=14)
    plt.ylabel(r'weighted counts $\left[\frac{1}{s}\right]$', size=14)
    ax.legend(loc='upper right', handles=legend_elements)
    name = f"EnergypredvsEnergyfromtruth_{backbone}.png"
    if yscale in ['log']:
        ax.set_yscale('log')
        name = f"EnergypredvsEnergyfromtruth_{backbone}_log.png"
    
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
        plt.plot(train_loss["epoch"], train_loss["train_loss"], marker=".", ls='', color="navy", label="training loss")
        plt.plot(val_loss["epoch"], val_loss["val_loss"], marker=".", ls='', color="crimson", label="validation loss")
        plt.grid()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        #plt.title(f"Loss curves")
        plt.legend()
        os.makedirs(os.path.join(subfolder, 'plots'), exist_ok =True)
        temp = os.path.join(subfolder, 'plots', 'Losses.png')
        plt.savefig(temp, dpi=300)
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
        #plt.title(f"learning rate \n {timestamp.replace('_', ' ')}")
        plt.tight_layout()
        os.makedirs(os.path.join(subfolder, 'plots'), exist_ok =True)
        temp = os.path.join(subfolder, 'plots', 'learning_rate.png')
        plt.savefig(temp, dpi=300)
        plt.close()
        message = colored("SUCCESSFUL", "green") + f": learning rate plotted, saved to {temp}"
        logger.info(message)
    else:
        message = colored("ERROR", "red") + ": learning rate not found"
        logger.info(message)
 
#2d histogram plotting true deposited energy vs reconstructed energy 
#normalise every column of the true deposited energy 
def plotEtruevsEreco(
        dataframe: pd.DataFrame, 
        subfolder: str, 
        normalise: Union[str, List[str]] = ['E_true']
        ) -> None:
    """
    Method to plot the true deposited energy vs the reconstructed/predicted energy as an unweighted 2d histogram or heatmap
    dataframe: pandas dataframe that includes a prediction and a truth column 
    subfolder: folder to save plots to
    normalise: parameter to determine in what dimension the data should be normalised
                'E_true': Adds up every E_pred value for a E_true bin and uses that sum to normalise the histogram 
                        -> Interpretation: Probability distribution of what energy is reconstructed for a given true deposited energy  
                'E_reco' or 'E_pred': Adds up every E_true value for a E_pred bin and uses that sum to normalise the histogram 
                        -> Interpretation: Probability distribution of what would be the true deposited energy for a given predicted energy 
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
        fig, ax = plt.subplots(figsize=(10,10))

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
            cax = fig.add_axes([ax.get_position().x0,ax.get_position().y1+0.01,ax.get_position().width, 0.02])
            cbar = plt.colorbar(pc, cax=cax, location='top')
            cbar.ax.set_title('Normalized Counts', size = 16)

            #Set phantom y-label to match 
            ax.set_ylabel(r'predicted Energy $E_{pred} [GeV]$', 
                          color='none',
                          )

            #ax.set_title(f"unweighted but normalised (with respect to true deposited energy) plot \n of true deposited energy vs predicted energy")

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
            cax = fig.add_axes([ax.get_position().x0,ax.get_position().y1+0.01,ax.get_position().width, 0.02])
            cbar = plt.colorbar(pc, cax=cax, location='top')
            cbar.ax.set_title('Normalized Counts')

            #Set phantom y-label to match 
            ax.set_ylabel(r'predicted Energy $E_{pred} [GeV]$', 
                        #   color='none',
                          )
            #ax.set_title(f"unweighted but normalised (with respect to predicted energy) plot \n of true deposited energy vs predicted energy")
        else:
            filename = "2dHisto_EtruevsEreco_unweighted_notnormalised.png"

            hist = ax.hist2d(energy_true, energy_prediction, bins=[bins_Etrue, bins_Epred], 
                    norm=colors.LogNorm(), 
                    cmap='viridis'
                    )

            #Adding a colorbar
            cax = fig.add_axes([ax.get_position().x0,ax.get_position().y1+0.01,ax.get_position().width, 0.02])
            cbar = plt.colorbar(hist[3], cax=cax, location='top')
            cbar.ax.set_title('Counts', size = 16)   
            ax.set_ylabel(r'predicted Energy $E_{pred} [GeV]$')

            #ax.set_title(f"unweighted and not normalised plot of true deposited energy vs predicted energy")
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r'true deposited energy $E_{true} [GeV]$', size = 16)       

        #adding a grid
        ax.grid(True, which="both", linestyle='--', linewidth=0.5)

        #adding a legend
        ax.legend(loc='upper left')
        ax.set_xlim(right=max(energy_prediction)) #match x and y figure dim

        ax.set_aspect('equal', adjustable='box') 
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        # plt.tight_layout()
        plt.subplots_adjust(top=0.85)


        #try to save the figure 
        try:             
            os.makedirs(os.path.join(subfolder, 'plots'), exist_ok =True)
            save_path = os.path.join(subfolder, 'plots', filename)
            plt.savefig(save_path,
                        bbox_inches = 'tight', 
                        dpi=300,
                        )
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
    plt.savefig(save_path, dpi=300)
    plt.show()

#Calculate Quantiles (eg middle 68% qunatile) with scipy.stats.iqr for the reconstructed energies (one iqr per bin) and plot iqr vs true deposited energy
#iqr: width of the E reco distribution for a E true bin
def plotIQRvsEtrue(
        dataframe: pd.DataFrame, 
        subfolder:str, 
        scale: Union[str, List[str]] = ['linear', 'log'], 
        bins: int = 30,
        cosmic_primary_type: int = -1,
        ) -> None: 
    """
    Method to plot the interquartile range vs the true deposited energy.
    The IQR (here middle 68 percentile) is here the width of the E_reco distribution for a true deposited energy bin and acts as a 
    measure for reconstruction uncertainty.
    possible plots are looking at the IQR in logspace (applied the logarithmn before calculating the IQR) or 
    in linear space (IQR calculated for data without logarithmn)
    Binning for true deposited energy is calculated as np.linspace(rounded_log_Etrue_min, rounded_log_Etrue_max, bins) either way 
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
                
                # Mask to select reconstructed energies within the current bin of true deposited energy
                events_mask = (energy_true_log > e_lower) & (energy_true_log < e_upper)
                
                # Calculate IQR for the reconstructed energies in the current bin
                if np.any(events_mask):
                    iqr_value = iqr(energy_prediction_log[events_mask], rng=(16, 84))
                    #put it to 1sigma
                    iqr_value = iqr_value/2 
                    all_iqr.append(iqr_value)
                else:
                    all_iqr.append(np.nan)  # In case there are no events in the bin, append NaN

            plt.plot(e_log_bin_centers, all_iqr, marker='o', ls='', color='b') #TODO: map color to neutrino flavor 
            plt.xlabel(r'true deposited energy $E_{true} [log(GeV)]')
            plt.ylabel('IQR of reconstructed energy [log(GeV)]')
            #plt.title(f'IQR of reconstructed energy vs true deposited energy for {type_title}')
            plt.grid()
            plt.tight_layout()
            filename = f'IQRvsEtrue_logarithmicEreco_{type_label}_{bins}bins'
            os.makedirs(os.path.join(subfolder, 'plots'), exist_ok =True)
            save_path = os.path.join(subfolder, 'plots', filename)
            plt.savefig(save_path, dpi=300)
            plt.show()

        elif s =='linear':      
            fig, ax = plt.subplots(figsize=(10, 6))     
            #linear in y axis and divided by E_true
            all_iqr = []
            for i in range(len(e_log_bins) - 1):
                e_lower, e_upper = e_log_bins[i], e_log_bins[i + 1]
                
                # Mask to select reconstructed energies within the current bin of true deposited energy
                events_mask = (energy_true_log > e_lower) & (energy_true_log < e_upper)
                
                # Calculate IQR for the reconstructed energies in the current bin
                if np.any(events_mask):
                    iqr_value = iqr(energy_prediction[events_mask], rng=(16, 84))/2
                    all_iqr.append(iqr_value)
                else:
                    all_iqr.append(np.nan)  # In case there are no events in the bin, append NaN

            
            plt.plot(e_log_bin_centers, all_iqr/np.power(10, e_log_bin_centers) * 100, marker='o', ls='', color='b')
            plt.xlabel(r'true deposited energy $E_{true}$[log(GeV)]')
            plt.ylabel('IQR of reconstructed energy(GeV)/ true deposited energy at bin center(GeV) [%]')
            ax.set_ylim(bottom=0, top=150)
            #plt.title(f'normalized IQR of reconstructed energy vs true deposited energy for {type_title}')
            plt.grid()
            plt.tight_layout()
            filename = f'IQRvsEtrue_linearEreco_{type_label}_{bins}bins_normalized'
            os.makedirs(os.path.join(subfolder, 'plots'), exist_ok =True)
            save_path = os.path.join(subfolder, 'plots', filename)
            plt.savefig(save_path, dpi=300)
            plt.show()

def calculate_iqr_and_counts(
    energy_true_filtered: np.ndarray, 
    energy_prediction_filtered: np.ndarray, 
    e_log_bins: np.ndarray, 
    e_log_bin_centers: np.ndarray
) -> Tuple[List[float], List[int]]:
    """
    Calculate the IQR and event counts for each energy bin.
    
    Args:
        energy_true_log_filtered (np.ndarray): Logarithm of the true energy values after filtering.
        energy_prediction_filtered (np.ndarray): Filtered predicted energy values.
        e_log_bins (np.ndarray): The bin edges for log energy.
        e_log_bin_centers (np.ndarray): The center points of each bin.
    
    Returns:
        Tuple[List[float], List[int]]: IQR values and event counts for each bin.
    """
    all_iqr = []
    event_counts = []

    energy_true_log_filtered = np.log10(energy_true_filtered)
    for i in range(len(e_log_bins) - 1):
        e_lower, e_upper = e_log_bins[i], e_log_bins[i + 1]
        events_mask = (energy_true_log_filtered > e_lower) & (energy_true_log_filtered <= e_upper)

        # Count the number of events in the bin
        event_count = np.sum(events_mask)
        event_counts.append(event_count)

        if event_count > 10:
            relerror_in_energy = (energy_prediction_filtered[events_mask]-energy_true_filtered[events_mask])/energy_true_filtered[events_mask]
            iqr_value = iqr(relerror_in_energy, rng=(16, 84))
            # iqr_value = iqr(energy_prediction_filtered[events_mask], rng=(16, 84))
            # iqr_value /= np.power(10, e_log_bin_centers[i])  # Relative resolution
            iqr_value = iqr_value / 2  # Convert to 1σ
        else:
            iqr_value = np.nan

        all_iqr.append(iqr_value)

    return all_iqr, event_counts

def calculate_iqr_and_counts_v2(
    energy_true_log_filtered: np.ndarray, 
    energy_prediction_log_filtered: np.ndarray, 
    e_log_bins: np.ndarray, 
    e_log_bin_centers: np.ndarray
) -> Tuple[List[float], List[int]]:
    """
    Calculate the IQR and event counts for each energy bin.
    
    Args:
        energy_true_log_filtered (np.ndarray): Logarithm of the true energy values after filtering.
        energy_prediction_filtered (np.ndarray): Filtered predicted energy values.
        e_log_bins (np.ndarray): The bin edges for log energy.
        e_log_bin_centers (np.ndarray): The center points of each bin.
    
    Returns:
        Tuple[List[float], List[int]]: IQR values and event counts for each bin.
    """
    all_iqr = []
    event_counts = []

    for i in range(len(e_log_bins) - 1):
        e_lower, e_upper = e_log_bins[i], e_log_bins[i + 1]
        events_mask = (energy_true_log_filtered > e_lower) & (energy_true_log_filtered <= e_upper)

        # Count the number of events in the bin
        event_count = np.sum(events_mask)
        event_counts.append(event_count)

        if event_count > 10:
            residual = energy_prediction_log_filtered[events_mask] - energy_true_log_filtered[events_mask]
            iqr_value = iqr(residual, rng=(16, 84))
            # iqr_value /= np.power(10, e_log_bin_centers[i])  # Relative resolution
            iqr_value = iqr_value / 2  # Convert to 1σ
        else:
            iqr_value = np.nan

        all_iqr.append(iqr_value)


    return all_iqr, event_counts

def weighted_percentile(data, weights, percentiles):
    if isinstance(data, pd.Series):
        data = data.to_numpy()  # Convert to NumPy array
    if isinstance(weights, pd.Series):
        weights = weights.to_numpy()  # Convert to NumPy array

    sorted_idx = np.argsort(data)
    sorted_data = data[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cumulative_weights = np.cumsum(sorted_weights)
    total_weight = cumulative_weights[-1]
    
    return np.interp(
        percentiles * total_weight / 100.0, cumulative_weights, sorted_data
    )

def weighted_iqr(data, weights, rng=(16, 84)):
    # Compute weighted IQR using weighted percentiles
    p16 = weighted_percentile(data, weights, rng[0])
    p84 = weighted_percentile(data, weights, rng[1])
    return p84 - p16

def calculate_iqr_and_counts_weighted(
    energy_true_log_filtered: np.ndarray, 
    energy_prediction_log_filtered: np.ndarray, 
    e_log_bins: np.ndarray, 
    e_log_bin_centers: np.ndarray,
    weights: np.ndarray = None  # Add weights parameter
) -> Tuple[List[float], List[float]]:
    """
    Calculate the IQR and weighted event counts for each energy bin.
    
    Args:
        energy_true_log_filtered (np.ndarray): Logarithm of the true energy values after filtering.
        energy_prediction_log_filtered (np.ndarray): Filtered predicted energy values.
        e_log_bins (np.ndarray): The bin edges for log energy.
        e_log_bin_centers (np.ndarray): The center points of each bin.
        weights (np.ndarray): Weights for each event.
    
    Returns:
        Tuple[List[float], List[float]]: IQR values and weighted event counts for each bin.
    """

    all_iqr = []
    weighted_event_counts = []

    # Ensure weights are provided, otherwise default to uniform weights
    if weights is None:
        weights = np.ones_like(energy_true_log_filtered)

    for i in range(len(e_log_bins) - 1):
        e_lower, e_upper = e_log_bins[i], e_log_bins[i + 1]
        events_mask = (energy_true_log_filtered > e_lower) & (energy_true_log_filtered <= e_upper)

        # Select the weights and events that fall in this bin
        weights_in_bin = weights[events_mask]
        weighted_event_count = np.sum(weights_in_bin)  # Sum of weights as the event count
        weighted_event_counts.append(weighted_event_count)

        if np.sum(events_mask) > 10:
            residual = energy_prediction_log_filtered[events_mask] - energy_true_log_filtered[events_mask]
            # Use the custom weighted IQR
            
            iqr_value = weighted_iqr(residual, weights[events_mask], rng=(16, 84))
            iqr_value = iqr_value/2
            # weighted_percentiles = np.percentile(residual, [16, 84], method='inverted_cdf', weights=weights[events_mask])
            # iqr_value = (weighted_percentile[1]-weighted_percentile[0]) / 2  # Convert to 1σ
        else:
            iqr_value = np.nan

        all_iqr.append(iqr_value)

    return all_iqr, weighted_event_counts

def plot_iqr_vs_true_energy_classspecific(
    filtered_dataframe: pd.DataFrame, 
    energy_prediction: np.ndarray, 
    energy_true: np.ndarray, 
    runfolder: str, 
    type_title: str, 
    type_label: str, 
    bins: int,
    classification: int,
    scale: List[str] = ['linear'], 
    filter: str= 'nofilter',
    literature_comparison: bool = True,
) -> None:
    """
    Plot the interquartile range (IQR) of reconstructed energy vs true deposited energy.

    Args:
        filtered_dataframe (pd.DataFrame): Input dataframe with predictions and true values.
        energy_prediction (np.ndarray): Predicted energy values.
        energy_true (np.ndarray): true deposited energy values.
        subfolder (str): Directory to save the plots.
        type_title (str): Title describing the type of particle or event.
        type_label (str): Label describing the type of particle or event.
        scale (List[str]): List of scales for which to compute the IQR for the plot ('linear' and/or 'log').
        bins (int): Number of bins to use for the plot.
        filter (str): options: 'CascadeFilter_13', 'MuonFilter_13', 'OnlineL2Filter_17', 'HESEFilter_15', 'MESEFilter_15', 'HighQFilter_17', nofilter, allevents,
    """
    classification_map = get_classification_map()
    class_name = classification_map.get(classification, "unknown")
    # if classification ==-1:
    #     class_name = "all_considered_class"
    
    # Filter the events based on the filter argument
    filter_list = ['CascadeFilter_13', 'MuonFilter_13', 'OnlineL2Filter_17', 'HESEFilter_15', 'MESEFilter_15', 'HighQFilter_17']
    if filter == 'nofilter':
        filter_mask = filtered_dataframe[filter_list].sum(axis=1) == 0
    elif filter == 'allevents':
        filter_mask = np.ones(len(filtered_dataframe), dtype=bool)
    elif filter in filter_list:
        filter_mask = filtered_dataframe[filter] == 1
    else: 
        raise("Filter has to be a supported IceCube filter")

    energy_prediction = energy_prediction[filter_mask]
    energy_true = energy_true[filter_mask]
    filtered_dataframe = filtered_dataframe[filter_mask]

    energy_prediction_log = np.log10(energy_prediction)
    energy_true_log = np.log10(energy_true)
    
    rounded_log_Etrue_min = np.floor(min(energy_true_log))
    rounded_log_Etrue_max = np.ceil(max(energy_true_log))
    # e_log_bins = np.linspace(rounded_log_Etrue_min, rounded_log_Etrue_max, bins)# Leads to issues when trying to divide iqr for different classif
    e_log_bins = np.linspace(1, 8, bins)
    e_log_bin_centers = (e_log_bins[:-1] + e_log_bins[1:]) / 2
    
    # print(f"filter: {filter}, classif: {class_name}")

    for s in scale:
        fig, ax = plt.subplots(figsize=(10, 6))
        all_iqr = []
        event_counts = []  # To store the number of events in each bin

        for i in range(len(e_log_bins) - 1):
            e_lower, e_upper = e_log_bins[i], e_log_bins[i + 1]
            events_mask = (energy_true_log > e_lower) & (energy_true_log <= e_upper)

            # Count the number of events where the true energy is in this bin
            event_count = np.sum(events_mask)
            event_counts.append(event_count)
            
            if event_count > 10:
                if s == 'log':
                    iqr_value = iqr(energy_prediction_log[events_mask], rng=(16, 84))
                else:
                    iqr_value = iqr(energy_prediction[events_mask], rng=(16, 84))
                    iqr_value /= np.power(10, e_log_bin_centers[i])
                
                #Divide by 2 to get 1sigma instead of +-1 sigma
                iqr_value = iqr_value/2
            else:
                iqr_value = np.nan

            all_iqr.append(iqr_value)
            # print(f"bin center: {e_log_bin_centers[i]}, event count: {event_count}, IQR: {iqr_value}")
            
        # print('\n')
        # Filter out zero values
        all_iqr = np.array(all_iqr)
        valid_mask = all_iqr != 0

        # Total number of events
        total_events = np.sum(event_counts)

        if classification in [8,9]:
            output_data = pd.DataFrame({
                'e_log_bin_centers': e_log_bin_centers,
                'all_iqr': all_iqr
            })
            output_file = f"all_iqr_data_{class_name}_{filter}.csv"
            os.makedirs(os.path.join(runfolder, 'iqr_files'), exist_ok=True)
            output_data.to_csv(os.path.join(runfolder, 'iqr_files', output_file), index=False)
            # print(f"Saved data for classification {classification} and filter {filter} to {os.path.join(runfolder, 'iqr_files', output_file)}")

        if s== 'log':
            plt.plot(e_log_bin_centers[valid_mask], all_iqr[valid_mask], marker='o', ls='', color='b')
        else:
            plt.plot(e_log_bin_centers[valid_mask], all_iqr[valid_mask]*100, marker='o', ls='', color='b', label=f"calculated rel. resolution for {class_name}")
        plt.axvline(x=1, color='black', linestyle='-', label='lower energy threshold for selected events')

        if literature_comparison:
            def load_and_process_csv(filepath):
                data = pd.read_csv(filepath, delimiter=';', decimal=',', header=None, names=['energy', 'value'])
                data['energy'] = pd.to_numeric(data['energy'], errors='coerce')
                data['value'] = pd.to_numeric(data['value'], errors='coerce')
                data = data.dropna()
                # print(f"Processed data from {filepath}:\n{data}\n")
                return data.sort_values(by='energy')

            # Low energy cascades
            low_energy_cascades_csv = "/home/saturn/capn/capn108h/programming_GNNs_and_training/webplotdigitizer_data/lowEnergyCascades.csv"
            low_energy_casc_data = load_and_process_csv(low_energy_cascades_csv)
            ax.plot(low_energy_casc_data['energy'], low_energy_casc_data['value'], ls='--', alpha=0.8, color = 'C7', label='resolution, low energy cascades literature')

            # Low energy tracks
            low_energy_tracks_csv = "/home/saturn/capn/capn108h/programming_GNNs_and_training/webplotdigitizer_data/lowEnergyTracks.csv"
            low_energy_tracks_data = load_and_process_csv(low_energy_tracks_csv)
            ax.plot(low_energy_tracks_data['energy'], low_energy_tracks_data['value'], ls='--', alpha=0.8, color = 'C8', label='resolution, low energy tracks literature')

            # High energy nu_mu
            high_energy_numu_csv = "/home/saturn/capn/capn108h/programming_GNNs_and_training/webplotdigitizer_data/highEnergyEdepoNuMu.csv"
            high_energy_numu_data = load_and_process_csv(high_energy_numu_csv)
            ax.plot(np.log10(high_energy_numu_data['energy']), high_energy_numu_data['value'], ls=':', alpha=0.8, color = 'C9', label=r'resolution, high energy $\nu_{\mu}$ literature')

            # High energy nu_e
            resolution_nue_csv = "/home/saturn/capn/capn108h/programming_GNNs_and_training/webplotdigitizer_data/EdepoNUE.csv"
            resolution_nue_data = load_and_process_csv(resolution_nue_csv)
            ax.plot(np.log10(resolution_nue_data['energy']), resolution_nue_data['value'], ls=':', alpha=0.8, color = 'C10', label=r'resolution, $\nu_{e}$ literature')
  

        plt.xlabel(r'true deposited energy $E_{true}$ [log(GeV)]')
        ylabel = 'realtive resolution (IQR/2) (log GeV - log GeV)' if s == 'log' else 'relative resolution [%]'
        #relative resolution: resolution (IQR/2) (GeV) / true deposited energy (GeV) [%]

        plt.ylabel(ylabel)
        ax.set_ylim(bottom=0, top=150)
        # plt.title(f'relative resolution of reconstructed energy vs true deposited energy for {type_title} ({class_name}) \n passed filter: {filter} \n Total events: {total_events}) 
        plt.title(f'passed filter: {filter}, total events: {total_events} \n resolution only calculated for bins with above 10 events') 
        plt.grid()
        # ax.set_yscale('log')
        ax.legend()

        # # Annotate event counts on the plot
        for center, count in zip(e_log_bin_centers, event_counts):
            plt.text(center, 0.5, f'{count}', fontsize=8, ha='center', va='bottom', color='blue', rotation=45)

        plt.tight_layout()
        filename = f'IQRvsEtrue_{s}_Ereco_{type_label}_{class_name}_{bins}bins_{filter}_ylim'

        save_dir_iqr = os.path.join(runfolder, 'plots', 'iqr')
        os.makedirs(save_dir_iqr, exist_ok=True)

        if classification ==-1:
            temp = os.path.join(save_dir_iqr, 'overall')
            os.makedirs(temp, exist_ok=True)
            save_path = os.path.join(temp, filename)
        else:
            save_path = os.path.join(save_dir_iqr, filename)
        plt.savefig(save_path, dpi=300)
        plt.show()
        plt.close()

def plot_iqr_vs_true_energy_classspecific_v2(
    filtered_dataframe: pd.DataFrame, 
    energy_prediction: np.ndarray, 
    energy_true: np.ndarray, 
    runfolder: str, 
    type_title: str, 
    type_label: str, 
    bins: int,
    classification: int,
    filter_list: List[str] = ['CascadeFilter_13', 'MuonFilter_13', 'OnlineL2Filter_17', 'HESEFilter_15', 'MESEFilter_15', 'HighQFilter_17'],
    filter: str = 'allevents',
    literature_comparison: bool = True,
) -> None:
    """
    Plots the interquartile range (IQR) of reconstructed energy vs true deposited energy
    for each filter (into one plot), and below it plots the event count histograms, sharing the x-axis.
    
    Args:
        filtered_dataframe (pd.DataFrame): Input dataframe with predictions and true values.
        energy_prediction (np.ndarray): Predicted energy values.
        energy_true (np.ndarray): True deposited energy values.
        runfolder (str): Directory to save the plots.
        type_title (str): Title describing the type of particle or event.
        type_label (str): Label describing the type of particle or event.
        bins (int): Number of bins to use for the plot.
        classification (int): Classification of the event.
        filter_list (List[str]): List of filters to consider.
        filter (str): Filter name to use ('CascadeFilter_13', 'MuonFilter_13', etc.).
        literature_comparison (bool): Whether to include literature comparison in the plot.
    """
    
    classification_map = get_classification_map()
    class_name = classification_map.get(classification, "unknown")

    e_log_bins = np.linspace(1, 8, bins)
    e_log_bin_centers = (e_log_bins[:-1] + e_log_bins[1:]) / 2

    # Dictionary to hold IQR values and event counts for each filter
    all_iqr_dict = {}
    event_count_dict = {}

    for current_filter in filter_list + ['nofilter']:
        if current_filter == 'nofilter':
            filter_mask = filtered_dataframe[filter_list].sum(axis=1) == 0
        else:
            filter_mask = filtered_dataframe[current_filter] == 1
        
        energy_true_log = np.log10(energy_true[filter_mask])
        energy_prediction_filtered = energy_prediction[filter_mask]

        # Calculate IQR for each bin
        all_iqr = []
        event_counts = []

        # print(f"filter: {current_filter}, classif: {class_name}")

        for i in range(len(e_log_bins) - 1):
            e_lower, e_upper = e_log_bins[i], e_log_bins[i + 1]
            events_mask = (energy_true_log > e_lower) & (energy_true_log <= e_upper)

            # Count the number of events in the bin
            event_count = np.sum(events_mask)
            event_counts.append(event_count)

            if event_count > 10:
                iqr_value = iqr(energy_prediction_filtered[events_mask], rng=(16, 84))
                iqr_value /= np.power(10, e_log_bin_centers[i])  # Relative resolution
                iqr_value = iqr_value / 2  # Convert to 1σ
            else:
                iqr_value = np.nan

            all_iqr.append(iqr_value)
        #     print(f"bin center: {e_log_bin_centers[i]}, event count: {event_count}, IQR: {iqr_value}")
        # print("\n")

        # Store IQR and event counts
        all_iqr_dict[current_filter] = np.array(all_iqr)
        event_count_dict[current_filter] = np.array(event_counts)

    for filter_name, event_counts in event_count_dict.items():
        print(f"{filter_name}: {event_counts}")
        print("\n")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot the IQR for each filter
    for current_filter in filter_list + ['nofilter']:
        valid_mask = ~np.isnan(all_iqr_dict[current_filter]) & (all_iqr_dict[current_filter] != 0)
        ax1.plot(e_log_bin_centers[valid_mask], all_iqr_dict[current_filter][valid_mask] * 100, 
                 marker='o', ls='', label=f"rel. resolution for {current_filter}")
    
    # Literature comparison
    if literature_comparison:
        def load_and_process_csv(filepath):
            data = pd.read_csv(filepath, delimiter=';', decimal=',', header=None, names=['energy', 'value'])
            data['energy'] = pd.to_numeric(data['energy'], errors='coerce')
            data['value'] = pd.to_numeric(data['value'], errors='coerce')
            data = data.dropna()
            # print(f"Processed data from {filepath}:\n{data}\n")
            return data.sort_values(by='energy')

        # Low energy cascades
        low_energy_cascades_csv = "/home/saturn/capn/capn108h/programming_GNNs_and_training/webplotdigitizer_data/lowEnergyCascades.csv"
        low_energy_casc_data = load_and_process_csv(low_energy_cascades_csv)
        ax1.plot(low_energy_casc_data['energy'], low_energy_casc_data['value'], ls='--', alpha=0.8, color = 'C7', label='resolution, low energy cascades literature')

        # Low energy tracks
        low_energy_tracks_csv = "/home/saturn/capn/capn108h/programming_GNNs_and_training/webplotdigitizer_data/lowEnergyTracks.csv"
        low_energy_tracks_data = load_and_process_csv(low_energy_tracks_csv)
        ax1.plot(low_energy_tracks_data['energy'], low_energy_tracks_data['value'], ls='--', alpha=0.8, color = 'C8', label='resolution, low energy tracks literature')

        # High energy nu_mu
        high_energy_numu_csv = "/home/saturn/capn/capn108h/programming_GNNs_and_training/webplotdigitizer_data/highEnergyEdepoNuMu.csv"
        high_energy_numu_data = load_and_process_csv(high_energy_numu_csv)
        ax1.plot(np.log10(high_energy_numu_data['energy']), high_energy_numu_data['value'], ls=':', alpha=0.8, color = 'C9', label=r'resolution, high energy $\nu_{\mu}$ literature')

        # High energy nu_e
        resolution_nue_csv = "/home/saturn/capn/capn108h/programming_GNNs_and_training/webplotdigitizer_data/EdepoNUE.csv"
        resolution_nue_data = load_and_process_csv(resolution_nue_csv)
        ax1.plot(np.log10(resolution_nue_data['energy']), resolution_nue_data['value'], ls=':', alpha=0.8, color = 'C10', label=r'resolution, $\nu_{e}$ literature')
    
    ax1.set_ylabel('relative resolution [%]')
    ax1.set_ylim(bottom=0, top=150)
    ax1.axvline(x=1, color='black', linestyle='-', label='lower energy threshold for selected events')
    ax1.set_title(f"Relative resolution of reconstructed energy vs true energy for {type_title} ({class_name})")
    ax1.legend()
    ax1.grid()

    # Plot the event count histogram for each filter
    for current_filter in filter_list + ['nofilter']:
        # ax2.bar(e_log_bin_centers, event_count_dict[current_filter], width=0.1, alpha=0.5, label=f"Counts for {current_filter}")
        ax2.step(e_log_bins[:-1], event_count_dict[current_filter], where='mid', label=f"Counts for {current_filter}")

    ax2.set_yscale("log")
    ax2.set_ylabel('Event counts')
    ax2.set_xlabel(r'true deposited Energy $E_{true}$ [log(GeV)]')
    ax2.grid()
    # ax2.legend()

    # Save the plots
    plt.tight_layout()
    temp = os.path.join(runfolder, 'combined_plot_iqr_event_counts')
    os.makedirs(temp, exist_ok=True)
    save_path = os.path.join(temp, f'combined_plot_iqr_event_counts_{class_name}.png')
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

def plot_iqr_vs_true_energy_classspecific_v3(
    filtered_dataframe: pd.DataFrame, 
    energy_prediction: np.ndarray, 
    energy_true: np.ndarray, 
    runfolder: str, 
    type_title: str, 
    type_label: str, 
    bins: int,
    classification: int,
    filter_list: List[str] = ['CascadeFilter_13', 'MuonFilter_13', 'OnlineL2Filter_17', 'HESEFilter_15', 'MESEFilter_15', 'HighQFilter_17'],
    filter: str = 'allevents',
    literature_comparison: bool = True,
) -> None:
    """
    Plots the interquartile range (IQR) of reconstructed energy vs true deposited energy
    for each filter (into one plot), and below it plots the event count histograms, sharing the x-axis.
    
    If classification == -1, creates a second plot with all events combined (no filters).
    
    Args:
        filtered_dataframe (pd.DataFrame): Input dataframe with predictions and true values.
        energy_prediction (np.ndarray): Predicted energy values.
        energy_true (np.ndarray): True deposited energy values.
        runfolder (str): Directory to save the plots.
        type_title (str): Title describing the type of particle or event.
        type_label (str): Label describing the type of particle or event.
        bins (int): Number of bins to use for the plot.
        classification (int): Classification of the event.
        filter_list (List[str]): List of filters to consider.
        filter (str): Filter name to use ('CascadeFilter_13', 'MuonFilter_13', etc.).
        literature_comparison (bool): Whether to include literature comparison in the plot.
    """
    
    classification_map = get_classification_map()
    class_name = classification_map.get(classification, "unknown")

    e_log_bins = np.linspace(1, 8, bins)
    e_log_bin_centers = (e_log_bins[:-1] + e_log_bins[1:]) / 2

    # Dictionary to hold IQR values and event counts for each filter
    all_iqr_dict = {}
    event_count_dict = {}

    for current_filter in filter_list + ['nofilter', 'allevents']:
        if current_filter == 'nofilter':
            filter_mask = filtered_dataframe[filter_list].sum(axis=1) == 0
        elif current_filter == 'allevents': 
            filter_mask = np.ones(len(filtered_dataframe), dtype=bool)
        else:
            filter_mask = filtered_dataframe[current_filter] == 1
            # print(f'filter mask for {current_filter}: {filter_mask}')
        
        energy_true_filtered = energy_true[filter_mask]
        energy_true_log_filtered = np.log10(energy_true_filtered)
        # energy_prediction_filtered = energy_prediction[filter_mask]
        # # Calculate IQR for each bin
        # all_iqr_forfilter, event_counts_forfilter = calculate_iqr_and_counts(energy_true_log_filtered, energy_prediction_filtered, e_log_bins, e_log_bin_centers)

        energy_prediction_filtered = energy_prediction[filter_mask]
        energy_prediction_log_filtered = np.log(energy_prediction[filter_mask])
        all_iqr_forfilter, event_counts_forfilter = calculate_iqr_and_counts_v2(energy_true_log_filtered, energy_prediction_log_filtered, e_log_bins, e_log_bin_centers)

        # all_iqr_forfilter, event_counts_forfilter = calculate_iqr_and_counts(energy_true_filtered, energy_prediction_filtered, e_log_bins, e_log_bin_centers)


        # Store IQR and event counts
        all_iqr_dict[current_filter] = np.array(all_iqr_forfilter)
        event_count_dict[current_filter] = np.array(event_counts_forfilter)

        # if current_filter == 'allevents':
        #     print(f'classification: {class_name} -- resolution: {all_iqr_dict[current_filter]}')
        #     print('\n')
        # if class_name == 'all_considered_classes':
        #     print(f'filter: {current_filter} -- resolution: {all_iqr_dict[current_filter]}')
        #     print('\n')

    # Plotting function
    def plot_iqr_and_counts(title_suffix, iqr_data, count_data, filter_list_call):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        # Plot the IQR for each filter
        for current_filter in filter_list_call:
            filter_name=current_filter
            if current_filter == 'nofilter':
                filter_name = 'no filters passed'
            valid_mask = ~np.isnan(iqr_data[current_filter]) & (iqr_data[current_filter] != 0)
            ax1.plot(e_log_bin_centers[valid_mask], iqr_data[current_filter][valid_mask] * 100, 
                     marker='o', ls='', label=f"{filter_name}")
        
        # Literature comparison
        if literature_comparison:
            def load_and_process_csv(filepath):
                data = pd.read_csv(filepath, delimiter=';', decimal=',', header=None, names=['energy', 'value'])
                data['energy'] = pd.to_numeric(data['energy'], errors='coerce')
                data['value'] = pd.to_numeric(data['value'], errors='coerce')
                data = data.dropna()
                # print(f"Processed data from {filepath}:\n{data}\n")
                return data.sort_values(by='energy')

            # Low energy cascades
            low_energy_cascades_csv = "/home/saturn/capn/capn108h/programming_GNNs_and_training/webplotdigitizer_data/lowEnergyCascades.csv"
            low_energy_casc_data = load_and_process_csv(low_energy_cascades_csv)
            ax1.plot(low_energy_casc_data['energy'], low_energy_casc_data['value'], ls='--', alpha=0.8, color = 'C0', label='low energy shower-like events, DynEdge, literature')

            # Low energy tracks
            low_energy_tracks_csv = "/home/saturn/capn/capn108h/programming_GNNs_and_training/webplotdigitizer_data/lowEnergyTracks.csv"
            low_energy_tracks_data = load_and_process_csv(low_energy_tracks_csv)
            ax1.plot(low_energy_tracks_data['energy'], low_energy_tracks_data['value'], ls='--', alpha=0.8, color = 'C2', label='low energy track-like events, DynEdge, literature')

            # High energy nu_mu
            high_energy_numu_csv = "/home/saturn/capn/capn108h/programming_GNNs_and_training/webplotdigitizer_data/highEnergyEdepoNuMu.csv"
            high_energy_numu_data = load_and_process_csv(high_energy_numu_csv)
            ax1.plot(np.log10(high_energy_numu_data['energy']), high_energy_numu_data['value'], ls='-.', alpha=0.8, color = 'C4', label=r'high energy $\nu_{\mu}$ events, literature')

            # High energy nu_e
            resolution_nue_csv = "/home/saturn/capn/capn108h/programming_GNNs_and_training/webplotdigitizer_data/EdepoNUE.csv"
            resolution_nue_data = load_and_process_csv(resolution_nue_csv)
            ax1.plot(np.log10(resolution_nue_data['energy']), resolution_nue_data['value'], ls='-.', alpha=0.8, color = 'C6', label=r'$\nu_{e}$ events, literature')
    
        ax1.set_ylabel('relative resolution [%]')

        minor_ticks = np.arange(0, 8, 0.25)
        ax1.set_xticks(minor_ticks, minor=True)

        ax1.set_ylim(bottom=0, top=150)
        ax1.axvline(x=1, color='black', linestyle='-', label='lower energy threshold for selected events')
        ax1.set_title(f"Relative resolution of reconstructed energy vs true energy for {class_name} {title_suffix.replace('_', '')} \n rel. resolution only calculated for bins with above 10 events")
        ax1.legend()
        ax1.grid()

        # Plot the event count histogram for each filter
        for current_filter in filter_list_call:
            ax2.step(e_log_bins, np.concatenate([[count_data[current_filter][0]], count_data[current_filter]]), where='pre', label=f"Counts for {filter_name}")
            # ax2.hist(
            #     e_log_bins[:-1],  # Bin edges, excluding the last edge because hist uses the edges
            #     bins=e_log_bins,  # The full array of bin edges
            #     weights=count_data[current_filter],  # The counts for each bin
            #     histtype='step',  # Step-style histogram
            #     label=f"Counts for {current_filter}"
            # )

        ax2.axvline(x=1, color='black', linestyle='-', label='lower energy threshold for selected events')
        ax2.set_yscale("log")
        ax2.set_ylabel('Event counts')
        ax2.set_xlabel(r'true deposited Energy $E_{true}$ [$log_{10}$(GeV)]')
        ax2.grid()
        # ax2.legend()

        # Save the plots
        plt.tight_layout()
        temp = os.path.join(runfolder, 'plots', 'combined_plot_iqr_event_counts')
        os.makedirs(temp, exist_ok=True)
        save_path = os.path.join(temp, f'combined_plot_iqr_event_counts_{class_name}{title_suffix}.png')
        plt.savefig(save_path, dpi=400)
        plt.show()
        plt.close()

    plot_iqr_and_counts("", all_iqr_dict, event_count_dict, filter_list + ['nofilter'])
    plot_iqr_and_counts("_allevents", all_iqr_dict, event_count_dict, ['allevents'])

def plot_iqr_vs_true_energy_classspecific_weighted(
    filtered_dataframe: pd.DataFrame, 
    energy_prediction: np.ndarray, 
    energy_true: np.ndarray, 
    runfolder: str, 
    type_title: str, 
    type_label: str, 
    bins: int,
    classification: int,
    weights: np.ndarray,  # New parameter for weights
    filter_list: List[str] = ['CascadeFilter_13', 'MuonFilter_13', 'OnlineL2Filter_17', 'HESEFilter_15', 'MESEFilter_15', 'HighQFilter_17'],
    filter: str = 'allevents',
    literature_comparison: bool = True,
) -> None:
    """
    Plots the interquartile range (IQR) of reconstructed energy vs true deposited energy
    for each filter (into one plot), and below it plots the event count histograms, sharing the x-axis.
    
    If classification == -1, creates a second plot with all events combined (no filters).
    
    Args:
        filtered_dataframe (pd.DataFrame): Input dataframe with predictions and true values.
        energy_prediction (np.ndarray): Predicted energy values.
        energy_true (np.ndarray): True deposited energy values.
        runfolder (str): Directory to save the plots.
        type_title (str): Title describing the type of particle or event.
        type_label (str): Label describing the type of particle or event.
        bins (int): Number of bins to use for the plot.
        classification (int): Classification of the event.
        event_weights (np.ndarray): Array of weights for each event.
        filter_list (List[str]): List of filters to consider.
        filter (str): Filter name to use ('CascadeFilter_13', 'MuonFilter_13', etc.).
        literature_comparison (bool): Whether to include literature comparison in the plot.
    """
    
    classification_map = get_classification_map()
    class_name = classification_map.get(classification, "unknown")

    e_log_bins = np.linspace(1, 8, bins)
    e_log_bin_centers = (e_log_bins[:-1] + e_log_bins[1:]) / 2

    # Dictionary to hold IQR values and event counts for each filter
    all_iqr_weighted_dict = {}
    event_count_weighted_dict = {}

    for current_filter in filter_list + ['nofilter', 'allevents']:
        if current_filter == 'nofilter':
            filter_mask = filtered_dataframe[filter_list].sum(axis=1) == 0
        elif current_filter == 'allevents': 
            filter_mask = np.ones(len(filtered_dataframe), dtype=bool)
        else:
            filter_mask = filtered_dataframe[current_filter] == 1
        
        # Apply filter to true energy, predicted energy, and weights
        energy_true_log_filtered = np.log10(energy_true[filter_mask])
        energy_prediction_log_filtered = np.log(energy_prediction[filter_mask])
        weights_filtered = weights[filter_mask]  # NEW: filter weights

        # Call modified IQR calculation function
        all_iqr_forfilter_weighted, event_counts_forfilter_weighted = calculate_iqr_and_counts_weighted(
            energy_true_log_filtered, energy_prediction_log_filtered, e_log_bins, e_log_bin_centers, weights_filtered
        )

        # Store IQR and event counts
        all_iqr_weighted_dict[current_filter] = np.array(all_iqr_forfilter_weighted)
        event_count_weighted_dict[current_filter] = np.array(event_counts_forfilter_weighted)

        if class_name == 'all_considered_classes':
            print(f'filter: {current_filter} -- resolution: {all_iqr_weighted_dict[current_filter]}')
            print('\n')

    # Plotting function
    def plot_iqr_and_counts_weighted(title_suffix, iqr_data, count_data, filter_list_call):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        # Plot the IQR for each filter
        for current_filter in filter_list_call:
            valid_mask = ~np.isnan(iqr_data[current_filter]) & (iqr_data[current_filter] != 0)
            ax1.plot(e_log_bin_centers[valid_mask], iqr_data[current_filter][valid_mask] * 100, 
                     marker='o', ls='', label=f"{current_filter}")
        
        # Literature comparison
        if literature_comparison:
            def load_and_process_csv(filepath):
                data = pd.read_csv(filepath, delimiter=';', decimal=',', header=None, names=['energy', 'value'])
                data['energy'] = pd.to_numeric(data['energy'], errors='coerce')
                data['value'] = pd.to_numeric(data['value'], errors='coerce')
                data = data.dropna()
                # print(f"Processed data from {filepath}:\n{data}\n")
                return data.sort_values(by='energy')

            # Low energy cascades
            low_energy_cascades_csv = "/home/saturn/capn/capn108h/programming_GNNs_and_training/webplotdigitizer_data/lowEnergyCascades.csv"
            low_energy_casc_data = load_and_process_csv(low_energy_cascades_csv)
            ax1.plot(low_energy_casc_data['energy'], low_energy_casc_data['value'], ls='--', alpha=0.8, color = 'C0', label='low energy cascade-like events, literature')

            # Low energy tracks
            low_energy_tracks_csv = "/home/saturn/capn/capn108h/programming_GNNs_and_training/webplotdigitizer_data/lowEnergyTracks.csv"
            low_energy_tracks_data = load_and_process_csv(low_energy_tracks_csv)
            ax1.plot(low_energy_tracks_data['energy'], low_energy_tracks_data['value'], ls='--', alpha=0.8, color = 'C2', label='low energy track-like events, literature')

            # High energy nu_mu
            high_energy_numu_csv = "/home/saturn/capn/capn108h/programming_GNNs_and_training/webplotdigitizer_data/highEnergyEdepoNuMu.csv"
            high_energy_numu_data = load_and_process_csv(high_energy_numu_csv)
            ax1.plot(np.log10(high_energy_numu_data['energy']), high_energy_numu_data['value'], ls='-.', alpha=0.8, color = 'C4', label=r'high energy $\nu_{\mu}$ events, literature')

            # High energy nu_e
            resolution_nue_csv = "/home/saturn/capn/capn108h/programming_GNNs_and_training/webplotdigitizer_data/EdepoNUE.csv"
            resolution_nue_data = load_and_process_csv(resolution_nue_csv)
            ax1.plot(np.log10(resolution_nue_data['energy']), resolution_nue_data['value'], ls='-.', alpha=0.8, color = 'C6', label=r'$\nu_{e}$ events, literature')
    
        ax1.set_ylabel('relative resolution (weighted) [%]')
        ax1.set_ylim(bottom=0, top=150)
        ax1.axvline(x=1, color='black', linestyle='-', label='lower energy threshold for selected events')
        ax1.set_title(f"Relative resolution (with weights) of reconstructed energy vs true energy for {class_name} {title_suffix.replace('_', '')} \n rel. resolution only calculated for bins with above 10 events")
        ax1.legend()
        ax1.grid()

        # Plot the event count histogram for each filter
        for current_filter in filter_list_call:
            ax2.step(e_log_bins, np.concatenate([[count_data[current_filter][0]], count_data[current_filter]]), where='pre', label=f"weighted counts for {current_filter}")
            # ax2.hist(
            #     e_log_bins[:-1],  # Bin edges, excluding the last edge because hist uses the edges
            #     bins=e_log_bins,  # The full array of bin edges
            #     weights=count_data[current_filter],  # The counts for each bin
            #     histtype='step',  # Step-style histogram
            #     label=f"Counts for {current_filter}"
            # )

        ax2.axvline(x=1, color='black', linestyle='-', label='lower energy threshold for selected events')
        ax2.set_yscale("log")
        ax2.set_ylabel('weighted event counts')
        ax2.set_xlabel(r'true deposited Energy $E_{true}$ [$log_{10}$(GeV)]')
        ax2.grid()
        # ax2.legend()

        # Save the plots
        plt.tight_layout()
        temp = os.path.join(runfolder, 'weighted_plots', 'combined_plot_iqr_event_counts')
        os.makedirs(temp, exist_ok=True)
        save_path = os.path.join(temp, f'combined_plot_iqr_event_counts_{class_name}{title_suffix}.png')
        plt.savefig(save_path, dpi=400)
        plt.show()
        plt.close()

    plot_iqr_and_counts_weighted("", all_iqr_weighted_dict, event_count_weighted_dict, filter_list + ['nofilter'])
    plot_iqr_and_counts_weighted("_allevents", all_iqr_weighted_dict, event_count_weighted_dict, ['allevents'])

def format_threshold(threshold):
    if threshold.is_integer():
        formatted_str = f"{int(threshold):.0e}"
        # Remove the plus sign and leading zero from the exponent
        parts = formatted_str.split('e')
        exponent = parts[1].replace('+', '').lstrip('0')
        return f"{parts[0]}e{exponent}"
    else:
        formatted_str = f"{threshold:.0e}"
        # Remove the plus sign and leading zero from the exponent
        parts = formatted_str.split('e')
        exponent = parts[1].replace('+', '').lstrip('0')
        return f"{parts[0]}e{exponent}"
            
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
    threshold: float = 1e4,
) -> None:
    """
    Plot the half interquartile range (IQR) of reconstructed energy (rng= 16,84) vs a given variable. 
    Corresponds to 1 sigma if the energy was gaussian distributed. 

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
        threshold (float): Energy threshold to separate the plots.
    """
    if threshold < 10 and threshold > 1e8:
        raise("Threshold should be between 10 and 1e8 GeV")
    classification_map = get_classification_map()
    class_name = classification_map.get(classification, "unknown")
    # if classification ==-1:
    #     class_name = "all_considered_class"

    min_val = np.floor(min(variable))
    max_val = np.ceil(max(variable))
    bins_edges = np.linspace(min_val, max_val, bins)
    bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2

    for part in ['low', 'high', 'full']:
        fig, ax = plt.subplots(figsize=(10, 6))
        all_relerror = []
        event_counts = []  # To store the number of events in each bin

        for i in range(len(bins_edges) - 1):
            lower, upper = bins_edges[i], bins_edges[i + 1]
            events_mask = (variable > lower) & (variable <= upper)

            if np.any(events_mask):
                energy_in_bin = energy_true[events_mask]

                # Separate events based on energy threshold
                if part == 'low':
                    energy_mask = (energy_in_bin < threshold)
                elif part == 'high':
                    energy_mask = (energy_in_bin >= threshold)
                else:  # part == 'all'
                    energy_mask = np.ones_like(energy_in_bin, dtype=bool)

                combined_mask = np.zeros_like(events_mask, dtype=bool)
                combined_mask[events_mask] = energy_mask

                event_counts.append(np.sum(combined_mask))  # Count the events in this bin

                if np.any(combined_mask):
                    relerror_values = (np.abs(energy_prediction[combined_mask] - energy_true[combined_mask]) / energy_true[combined_mask])
                    all_relerror.append(np.mean(relerror_values))
                else:
                    all_relerror.append(np.nan)

            else:
                all_relerror.append(np.nan)
        
        all_relerror = np.array(all_relerror)
        valid_mask = all_relerror != 0
            
        total_events = sum(event_counts)
        
        formatted_threshold = format_threshold(threshold)

        energy_range = (
            "full $E_{{depo,true}}$ range" if part == 'full' else 
            f"10 GeV < $E_{{depo,true}}$ <= {formatted_threshold} GeV" if part == 'low' else 
            f"{formatted_threshold} GeV < $E_{{depo,true}}$ <= 1e8 GeV"
        )
        
        ylabel = 'relative error of reconstructed deposited energy [%]'
        ax.set_ylabel(ylabel)

        all_relerror_percent = np.array(all_relerror)*100

        plotting_x_value = None

        if variable_name in ['r', 'z']:
            plotting_x_value = bin_centers[valid_mask]
            plt.plot(plotting_x_value, all_relerror_percent[valid_mask], marker='o', ls='', color='b')
            plt.xlabel(f'true vertex {variable_name} (m)')
    
            if variable_name == 'z':
                plt.axvline(x=524.56, color='r', linestyle='--', label='first DOM position')
                plt.axvline(x=-512.82, color='g', linestyle='--', label='last DOM position')
                plt.axvspan(-150, -50, color='grey', alpha=0.3, label='dust layer')

                # Plot secondary y-axis for absorption length
                absorption_length_csv = "/home/saturn/capn/capn108h/programming_GNNs_and_training/webplotdigitizer_data/AbsorptionLengthinIce.csv"
                absorption_data = pd.read_csv(absorption_length_csv, delimiter=';', decimal=',', header=None, names=['depth', 'absorption_length'])
                absorption_data['depth'] = pd.to_numeric(absorption_data['depth'], errors='coerce')
                absorption_data['absorption_length'] = pd.to_numeric(absorption_data['absorption_length'], errors='coerce')
                absorption_data = absorption_data.dropna()
                absorption_z = 1950 - absorption_data['depth']
                absorption_length = absorption_data['absorption_length']

                ax2 = ax.twinx()
                ax2.plot(absorption_z, absorption_length, ls='-.', alpha=0.5, color='r', label='Absorption Length in Ice')
                ax2.set_ylabel('Absorption Length (m)', color='r')
                ax2.tick_params(axis='y', labelcolor='r')

                # Combine legends
                handles1, labels1 = ax.get_legend_handles_labels()
                handles2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(handles=handles1 + handles2, labels=labels1 + labels2, loc='best')


            elif variable_name == 'r':
                plt.axvline(x=601.18, color='r', linestyle='--', label='maximal string distance')
                plt.axvline(x=12.86, color='g', linestyle='--', label='minimal string distance')
                ax.legend()

            title = f'relative error of reconstructed deposited energy vs true vertex {variable_name} for {type_title} ({class_name}) \n Energy range: {energy_range} \n Total events: {total_events}'
            
        elif variable_name in ['cos_zenith', 'azimuth']:
            if variable_name == 'cos_zenith':
                plotting_x_value = np.cos(bin_centers[valid_mask])
                plt.plot(plotting_x_value, all_relerror_percent[valid_mask], marker='o', ls='', color='b')
                plt.xlabel(f'cos({variable_name})')

                title = f'relative error of reconstructed deposited energy vs cosine true {variable_name} for {type_title} ({class_name}) \n Energy range: {energy_range} \n Total events: {total_events}'
            elif variable_name == 'azimuth':
                tick_positions = np.arange(0, 2 * np.pi + np.pi/4, np.pi/4)
                tick_labels = [r"$0$", r"${\pi}/{4}$", r"${\pi}/{2}$", r"${3\pi}/{4}$", r"$\pi$", r"${5\pi}/{4}$", r"${3\pi}/{2}$", r"${7\pi}/{4}$", r"$2\pi$" ]
                plt.xticks(tick_positions, tick_labels)
                plotting_x_value = bin_centers[valid_mask]
                plt.plot(plotting_x_value, all_relerror_percent[valid_mask], marker='o', ls='', color='b')
                plt.xlabel(f'true {variable_name} (rad)')
                
                title = f'relative error of reconstructed deposited energy vs true {variable_name} for {type_title} ({class_name}) \n Energy range: {energy_range} \n Total events: {total_events}'

        # Annotate event counts on the plot
        for center, count in zip(plotting_x_value, event_counts):
            plt.text(center, 0.5, f'{count}', fontsize=12, ha='center', va='bottom', color='blue', rotation=90)

        # ax.set_yscale('log')
        plt.title(f'Total events: {total_events}')

        if classification == 27:
            ax.set_ylim(bottom=0, top=min(5000, np.nanmax(all_relerror_percent)))
        else:
            ax.set_ylim(bottom=0, top=100)
        plt.grid()
        plt.tight_layout()

        filename = f'relerrorvs{variable_name}_{type_label}_{class_name}_{bins}bins_{part}'

        os.makedirs(os.path.join(runfolder, 'plots', 'relerror'), exist_ok=True)
        save_path = os.path.join(runfolder, 'plots', 'relerror', filename)
        plt.savefig(save_path, dpi=300)
        plt.show()
        plt.close()

def plot_relerror_vs_variable_v2(
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
    weights: np.ndarray = None,
    threshold: float = 1e4,
) -> None:
    """
    Plot the half interquartile range (IQR) of reconstructed energy (rng= 16,84) vs a given variable. 
    Corresponds to 1 sigma if the energy was gaussian distributed.

    Args:
        filtered_dataframe (pd.DataFrame): Input dataframe with predictions and true values.
        energy_prediction (np.ndarray): Predicted energy values.
        energy_true (np.ndarray): True energy values.
        variable (np.ndarray): Variable to plot against.
        variable_name (str): Name of the variable.
        runfolder (str): Directory to save the plots.
        type_title (str): Title describing the type of particle or event.
        type_label (str): Label describing the type of particle or event.
        bins (int): Number of bins to use for the plot.
        threshold (float): Energy threshold to separate the plots.
    """
    if threshold < 10 or threshold > 1e8:
        raise ValueError("Threshold should be between 10 and 1e8 GeV")
        
    classification_map = get_classification_map()
    class_name = classification_map.get(classification, "unknown")

    min_val = np.floor(min(variable))
    max_val = np.ceil(max(variable))
    bins_edges = np.linspace(min_val, max_val, bins)
    bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2            
    bin_width = (max_val - min_val) / bins

    fig, (ax, ax_hist) = plt.subplots(2, 1, figsize=(10, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    all_relerror = {'low': [], 'high': [], 'full': []}
    event_counts = {'low': [], 'high': [], 'full': []}  # To store the number of events in each bin
    
    event_count_weighted_dict = {'low': [], 'high': [], 'full': []}

    # print(f'weights: {weights}')

    for part in ['full', 'low', 'high']:
        for i in range(len(bins_edges) - 1):
            lower, upper = bins_edges[i], bins_edges[i + 1]
            events_mask = (variable > lower) & (variable <= upper)

            if np.any(events_mask):
                energy_in_bin = energy_true[events_mask]

                # Separate events based on energy threshold
                if part == 'low':
                    energy_mask = (energy_in_bin < threshold)
                elif part == 'high':
                    energy_mask = (energy_in_bin >= threshold)
                else:  # part == 'full'
                    energy_mask = np.ones_like(energy_in_bin, dtype=bool)

                combined_mask = np.zeros_like(events_mask, dtype=bool)
                combined_mask[events_mask] = energy_mask

                event_count = np.sum(combined_mask)  # Count the events in this bin
                event_counts[part].append(event_count)  # Append event count

                # print(f'weights: {weights}')
                # event_count_weighted = np.sum(weights[combined_mask])
                # event_count_weighted_dict[part].append(event_count_weighted)


                if event_count > 0:
                    relerror_values = (np.abs(energy_prediction[combined_mask] - energy_true[combined_mask]) / energy_true[combined_mask])
                    all_relerror[part].append(np.mean(relerror_values))
                else:
                    all_relerror[part].append(np.nan)  # Append NaN if no valid relerror
            else:
                event_counts[part].append(0)  # Append zero if no events in the bin
                all_relerror[part].append(np.nan)  # Append NaN if no valid relerror
        all_relerror[part] = np.array(all_relerror[part])

    formatted_threshold = format_threshold(threshold)

    energy_ranges = ["full $E_{{depo,true}}$ range", 
                    f"1e1 GeV < $E_{{depo,true}}$ <= {formatted_threshold} GeV",
                    f"{formatted_threshold} GeV < $E_{{depo,true}}$ <= 1e8 GeV",
                    ]  
    
    # Plot relative errors
    for part, color, label in zip(['full', 'low', 'high'], ['#88223d', 'C1', 'C2'], energy_ranges):
        # Set zorder based on the part
        if part == 'full':
            zorder_val = 10  # Bring 'full' plot to the front
        else:
            zorder_val = 1  
        
        valid_mask = ~np.isnan(all_relerror[part]) & (all_relerror[part] != 0)
        
        ax.plot(bin_centers[valid_mask], all_relerror[part][valid_mask] * 100, marker='o', ls='', label=label, color = color, zorder=zorder_val)
        # Plot event counts as a histogram
        ax_hist.step(bins_edges, np.concatenate([[event_counts[part][0]], event_counts[part]]), where='pre', label=f'{part.capitalize()} Count', color = color, zorder=zorder_val)
        
        #Potentially weighting counts
        # ax_hist.step(bins_edges, np.concatenate([[event_count_weighted_dict[part][0]], event_count_weighted_dict[part]]), where='pre',label=f'{part.capitalize()} weighted event count', color = color, zorder=zorder_val)

    if variable_name == 'z':
        for axes in [ax, ax_hist]:
            axes.axvline(x=524.56, color='r', linestyle='--', label='first DOM position')
            axes.axvline(x=-512.82, color='g', linestyle='--', label='last DOM position')
            axes.axvspan(-150, -50, color='grey', alpha=0.3, label='dust layer')

        # Plot secondary y-axis for absorption length
        absorption_length_csv = "/home/saturn/capn/capn108h/programming_GNNs_and_training/webplotdigitizer_data/AbsorptionLengthinIce.csv"
        absorption_data = pd.read_csv(absorption_length_csv, delimiter=';', decimal=',', header=None, names=['depth', 'absorption_length'])
        absorption_data['depth'] = pd.to_numeric(absorption_data['depth'], errors='coerce')
        absorption_data['absorption_length'] = pd.to_numeric(absorption_data['absorption_length'], errors='coerce')
        absorption_data = absorption_data.dropna()
        absorption_z = 1950 - absorption_data['depth']
        absorption_length = absorption_data['absorption_length']

        ax2 = ax.twinx()
        ax2.plot(absorption_z, absorption_length, ls='-.', alpha=0.4, color='r', label='Absorption Length in Ice')
        ax2.set_ylim([0, 140 * 4])  # 140% * 4 meters per percent
        ax2.set_ylabel('Absorption Length (m)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        ax_hist.set_xlabel(f'True Vertex {variable_name} (m)')
        # ax.set_xlabel(f'True Vertex {variable_name} (m)')

        # Combine legends
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(handles=handles1 + handles2, labels=labels1 + labels2, loc='best')
        
        title = f'relative error of reconstructed deposited energy vs true vertex {variable_name} for {class_name}'

    elif variable_name == 'r':
        for axes in [ax, ax_hist]:
            axes.axvline(x=601.18, color='r', linestyle='--', label='maximal string distance')
            axes.axvline(x=12.86, color='g', linestyle='--', label='minimal string distance')
        ax_hist.set_xlabel(f'True Vertex {variable_name} (m)')
        # ax.set_xlabel(f'True Vertex {variable_name} (m)')
        ax.legend()
        title = f'relative error of reconstructed deposited energy vs true vertex {variable_name} for {class_name}'

    elif variable_name == 'cos_zenith':
        ax_hist.set_xlabel(f'cos(zenith)')
        ax.legend()
        title = f'relative error of reconstructed deposited energy vs cosine true zenith for {class_name}'
        
    elif variable_name == 'azimuth':
        tick_positions = np.arange(0, 2 * np.pi + np.pi/4, np.pi/4)
        tick_labels = [r"$0$", r"${\pi}/{4}$", r"${\pi}/{2}$", r"${3\pi}/{4}$", r"$\pi$", r"${5\pi}/{4}$", r"${3\pi}/{2}$", r"${7\pi}/{4}$", r"$2\pi$" ]
        ax.set_xticks(tick_positions, tick_labels)
        ax_hist.set_xlabel(f'true {variable_name} (rad)')
        # plt.xlabel(f'true {variable_name} (rad)')
        ax.legend()
        
        title = f'relative error of reconstructed deposited energy vs true {variable_name} for {class_name}'
    
    ylabel = 'relative error of reconstructed deposited energy [%]'
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.set_title(title)

    ax_hist.set_ylabel('Event counts')
    ax_hist.set_yscale('log')
    # ax_hist.legend()
    ax_hist.grid()

    if classification in [26,27]:
        ax.set_yscale('log')
    else:
        ax.set_ylim(bottom=0, top=100)
    
    plt.tight_layout()
    
        
    filename = f'relerror_and_counts_vs_{variable_name}_{class_name}_{bins}bins'
    temp = os.path.join(runfolder, 'plots', 'combined_relerror_eventcount')
    os.makedirs(temp, exist_ok=True)
    save_path = os.path.join(temp, filename)
    plt.savefig(save_path, dpi=300)
    # print(save_path)
    plt.show()
    plt.close()

def plot_IQR_and_relerror(
    dataframe: pd.DataFrame, 
    runfolder: str, 
    plot_type: str = 'energy',
    bins: int = 35, 
    cosmic_primary_type: int = -1,
    classification: int = -1, 
    filter: str = 'allevents',
    prediction:str = 'energy_pred',
    truth_training_target_label:str = 'deposited_energy',
) -> None:
    """
    Plot the interquartile range (IQR, here middle 68 percentile) vs the true deposited energy or the relative error vs true vertex position in polar coordinates.
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
        
    # prediction = dataframe.keys()[0]  # Should be energy_pred
    # truth = dataframe.keys()[1]  # Should be training target label

    # Filter by classification 
    if classification != -1:
        dataframe = dataframe[dataframe['classification'] == classification]

    if cosmic_primary_type == -1:
        filtered_dataframe = dataframe
    else:
        filtered_dataframe = dataframe[dataframe['cosmic_primary_type'].abs().isin([cosmic_primary_type])]
        if filtered_dataframe.empty:
            return
    
    energy_prediction = filtered_dataframe[prediction]
    energy_true = filtered_dataframe[truth_training_target_label]
    vertex_x = filtered_dataframe['first_vertex_x']
    vertex_y = filtered_dataframe['first_vertex_y']
    vertex_z = filtered_dataframe['first_vertex_z']
    true_cos_zenith = np.cos(filtered_dataframe['primary_zenith'])
    true_azimuth = filtered_dataframe['primary_azimuth']
    weights = filtered_dataframe['sim_weight']


    vertex_r = np.sqrt(vertex_x**2 + vertex_y**2)
    
    if plot_type == 'energy':
        # plot_iqr_vs_true_energy_classspecific(filtered_dataframe, energy_prediction, energy_true, runfolder, type_title, type_label, bins, classification, filter=filter)
        plot_iqr_vs_true_energy_classspecific_v3(filtered_dataframe, energy_prediction, energy_true, runfolder, type_title, type_label, bins, classification)
        # plot_iqr_vs_true_energy_classspecific_weighted(filtered_dataframe, energy_prediction, energy_true, runfolder, type_title, type_label, bins, classification, weights)
        
    elif plot_type == 'r':
        plot_relerror_vs_variable_v2(filtered_dataframe, energy_prediction, energy_true, vertex_r, 'r', runfolder, type_title, type_label, bins, classification)
    elif plot_type == 'z':
        plot_relerror_vs_variable_v2(filtered_dataframe, energy_prediction, energy_true, vertex_z, 'z', runfolder, type_title, type_label, bins, classification)
    elif plot_type == 'cos_zenith':
        plot_relerror_vs_variable_v2(filtered_dataframe, energy_prediction, energy_true, true_cos_zenith, 'cos_zenith', runfolder, type_title, type_label, bins, classification)
        # plot_relerror_vs_variable_v2(filtered_dataframe, energy_prediction, energy_true, true_cos_zenith, 'cos_zenith', runfolder, type_title, type_label, bins, classification, weights)
    elif plot_type == 'azimuth':
        plot_relerror_vs_variable_v2(filtered_dataframe, energy_prediction, energy_true, true_azimuth, 'azimuth', runfolder, type_title, type_label, bins, classification)

def plot_lightyield_ratio(
     runfolder: str,   
):
    # Path to your data files
    data_path = os.path.join(runfolder, 'iqr_files')

    # Read all files related to em_hadr_cascade and hadr_cascade
    em_hadr_files = glob(os.path.join(data_path, 'all_iqr_data_em_hadr_cascade_*.csv'))
    hadr_files = glob(os.path.join(data_path, 'all_iqr_data_hadr_cascade_*.csv'))

    # Define a function to extract the filter name from the file path
    def get_filter_name(file_path):
        base_name = os.path.basename(file_path)
        return base_name.replace('all_iqr_data_em_hadr_cascade_', '').replace('all_iqr_data_hadr_cascade_', '').replace('.csv', '')

    # Read the data into DataFrames
    em_hadr_dfs = {get_filter_name(f): pd.read_csv(f) for f in em_hadr_files}
    hadr_dfs = {get_filter_name(f): pd.read_csv(f) for f in hadr_files}

    # Calculate the ratio and store in a dictionary
    ratios = {}
    max_ratio_value = 0
    for filter_name in em_hadr_dfs.keys():
        if filter_name in hadr_dfs:
            em_hadr_df = em_hadr_dfs[filter_name]
            hadr_df = hadr_dfs[filter_name]
            
            # Merge DataFrames on e_log_bin_centers
            merged_df = pd.merge(em_hadr_df, hadr_df, on='e_log_bin_centers', suffixes=('_em', '_hadr'), how='outer')
            
            # Handle missing values
            merged_df.fillna(0, inplace=True)  # Fill NaNs with 0
            
            # Calculate the ratio with proper handling for zero values
            merged_df['ratio'] = np.where(
                (merged_df['all_iqr_em'] == 0) & (merged_df['all_iqr_hadr'] == 0), 
                np.nan,  # If both are zero, set ratio to NaN
                merged_df['all_iqr_hadr'] / merged_df['all_iqr_em']
            )
            
            # Drop rows where the ratio is NaN or zero
            merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['ratio'])
            merged_df = merged_df[merged_df['ratio'] != 0]
            
            # Store the result
            ratios[filter_name] = merged_df[['e_log_bin_centers', 'ratio']]
            
            # Update the maximum ratio value
            if not merged_df['ratio'].empty:
                max_ratio_value = max(max_ratio_value, merged_df['ratio'].max())

    # Ensure max_ratio_value is valid for y-axis limits
    if np.isfinite(max_ratio_value) and max_ratio_value > 0:
        y_max_limit = max_ratio_value * 1.1
    else:
        y_max_limit = 10  # Default value in case of invalid max_ratio_value

    # Plot the ratios
    plt.figure(figsize=(12, 8))
    for filter_name, ratio_df in ratios.items():
        plt.plot(ratio_df['e_log_bin_centers'], ratio_df['ratio'], marker='o', label=filter_name)

    plt.xlabel('e_log_bin_centers')
    plt.ylabel('Light Yield Ratio (hadr_cascade / em_hadr_cascade)')
    plt.title('Light Yield Ratio for Different Filters')
    plt.legend(title='Filters', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, y_max_limit)  # Set y-axis limits with a bit of margin
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return

def plot_bias(
    dataframe: pd.DataFrame,
    runfolder: str,
    prediction:str = 'energy_pred',
    truth_training_target_label:str = 'deposited_energy',
    bins: int = 35,
):
    """
    plots biases as mean - true

    Args:
        dataframe (pd.DataFrame): _description_
        runfolder (str): _description_
        prediction (str, optional): _description_. Defaults to 'energy_pred'.
        truth_training_target_label (str, optional): _description_. Defaults to 'deposited_energy'.
        bins (int, optional): _description_. Defaults to 35.
    """    
    energy_prediction = dataframe[prediction]
    energy_true = dataframe[truth_training_target_label] 

    energy_true_log = np.log10(energy_true)
    energy_prediction_log = np.log10(energy_prediction)

    rounded_log_Etrue_min = np.floor(min(energy_true_log))
    rounded_log_Etrue_max = np.ceil(max(energy_true_log))

    e_log_bins = np.linspace(rounded_log_Etrue_min, rounded_log_Etrue_max, bins)
    e_log_bin_centers = (e_log_bins[:-1] + e_log_bins[1:]) / 2

    all_biases = []
    all_std = []

    for i in range(len(e_log_bins) - 1):
        e_lower, e_upper = e_log_bins[i], e_log_bins[i + 1]
        events_mask = (energy_true_log > e_lower) & (energy_true_log < e_upper)
        # energy_median = np.median(energy_prediction_log[events_mask]) 

        residual = energy_prediction_log[events_mask] - energy_true_log[events_mask]

        # energy_log_mean = np.mean(energy_prediction_log[events_mask]) 
        # energy_log_std = np.std(energy_prediction_log[events_mask]) 
        # energy_bias = energy_log_mean - e_log_bin_centers[i]

        energy_bias = np.mean(residual)
        energy_bias_std = np.std(residual, ddof=1)

        all_biases.append(energy_bias)
        all_std.append(energy_bias_std)

    fig, ax = plt.subplots()
    plt.plot(e_log_bin_centers, all_biases, marker='o', ls='',  label='calculated biases')
    plt.errorbar(e_log_bin_centers, all_biases, yerr=all_std, label=r'$\sigma$-standard error', ls='',  capsize=3, ecolor='C0')
    plt.xlabel(r"true deposited energy $E_{true}$ [$log_{10}$ (GeV)]")
    plt.ylabel(r"energy bias [arb. unit]")
    # plt.ylabel("median of log predicted energy - log true deposited energy [log(GeV)]")
    # plt.title("Calculated biases as median of the predicted energy - true deposited energy \n as function of true deposited energy")
    print(f'bin centers: {e_log_bin_centers} -- biases: {all_biases}')
    ax.legend()
    plt.grid()
    plt.tight_layout()
    filename = "EnergyBias_vs_Etruedeposited"
    os.makedirs(os.path.join(runfolder, 'plots'), exist_ok=True)
    save_path = os.path.join(runfolder, 'plots', filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

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
    #plt.title(f"Histogram of {label} for {flavor} \n {', '.join(filenames)}")
    plt.xlabel(f'{label} (log10 GeV)')
    plt.ylabel('Counts')
    if scale=='log':
        ax.set_yscale('log')
        filename = f"EnergyStructure_for_{label}_{flavor}_{'_'.join([filename.split('_')[0] for filename in filenames])}_v{dbversion}_logscale"
    else:
        filename = f"EnergyStructure_for_{label}_{flavor}_{'_'.join([filename.split('_')[0] for filename in filenames])}_v{dbversion}"
        
    os.makedirs(os.path.join(savefolder, 'plots'), exist_ok=True)
    save_path = os.path.join(savefolder, 'plots', filename)
    plt.savefig(save_path, dpi=300)
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

def PlottingRoutine(
              results: pd.DataFrame = None, 
              subfolder: str = None, 
              target_label: Union[str, List[str]] = ['deposited_energy'],
              config: dict = {}, 
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
                # print(f"Metrics file created at: {metrics_path}")
                plot_lossesandlearningrate(subfolder)
        else:
                message = colored("ERROR", "red") + f": Metrics file not found under {subfolder}."
                logger.info(message)

        #if no result dataframe is handed over, load the dataframe from the subfolder
        if results is None:
               results = loadresults(subfolder=subfolder)

        tracklike_events = [19,20,26,27]
        showerlike_events = [8,9,22,23]
        filter_list = ['CascadeFilter_13', 'MuonFilter_13', 'OnlineL2Filter_17', 'HESEFilter_15', 'MESEFilter_15', 'HighQFilter_17', 'nofilter', 'allevents']

        #All the other plotting calls
        plot_resultsashisto(results, subfolder=subfolder, target_label=target_label, backbone=config.get('backbone', 'DynEdge'))
        plot_resultsashisto(results, subfolder=subfolder, target_label=target_label, backbone=config.get('backbone', 'DynEdge'), yscale='log')
        plot_bias(dataframe=results, runfolder=subfolder)
        
        
        plotEtruevsEreco(results, subfolder=subfolder , normalise=['E_true', 'E_reco', 'nonormalisation'])
        # quit()
        
        for i in config.get('classifications_to_train_on', [8, 9, 19, 20, 22, 23, 26, 27]):

                # for j in filter_list: 
                #     plot_IQR_and_relerror(dataframe=results, runfolder=subfolder, plot_type='energy', classification=-1, filter=j)
                #     plot_IQR_and_relerror(dataframe=results, runfolder=subfolder, plot_type='energy', classification=i, filter=j)
                
                plot_IQR_and_relerror(dataframe=results, runfolder=subfolder, plot_type='energy', classification=-1)
                plot_IQR_and_relerror(dataframe=results, runfolder=subfolder, plot_type='energy', classification=i)

                if i in showerlike_events:
                    plot_IQR_and_relerror(dataframe=results, runfolder=subfolder, plot_type='r', classification=i)
                    plot_IQR_and_relerror(dataframe=results, runfolder=subfolder, plot_type='z', classification=i)

                if i in tracklike_events:
                    plot_IQR_and_relerror(dataframe=results, runfolder=subfolder, plot_type='cos_zenith', classification=i)
                    plot_IQR_and_relerror(dataframe=results, runfolder=subfolder, plot_type='azimuth', classification=i)
                
        
        plot_lightyield_ratio(runfolder=subfolder)

def main():
    logger = Logger()
    subfolder = '/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_and_saved_models/run_EnergyReco'
    filename = 'test_result_epoch41.h5'
    # ckpt_path = '/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_and_saved_models/run_jobid_867417/checkpoints/DynEdge-epoch=31-val_loss=0.09-train_loss=0.09.ckpt'
    # pth_path = '/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_and_saved_models/run_jobid_867417/GNN_DynEdge_mergedNuE_NuMu_NuTau.pth'
    results = loadresults_from_subfolder(subfolder=subfolder, filename=filename)
    config_path = '/home/saturn/capn/capn108h/programming_GNNs_and_training/config.yaml'
    config= utils.read_yaml(config_path=config_path)

    PlottingRoutine( 
                    results=results, 
                    subfolder=subfolder, 
                    config=config,
                    target_label=config.get('training_parameter', ['deposited_energy']), 
                    logger=logger
                )
    # results = loadresults_from_subfolder(subfolder=subfolder)
    # PlottingRoutine(results=results, subfolder=subfolder)

if __name__ == "__main__":
        main()  