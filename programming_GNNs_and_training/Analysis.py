#Script to analyse and plot the results of GNN training 
import os
from glob import glob
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import iqr

from termcolor import colored
from graphnet.utilities.logging import Logger

# plt.rcParams['text.usetex'] = True #produces errors?

import argparse

# def parse_args():
#     argparse.parser()
#     parser.known_args


#'Constants'
RUNFOLDER = "/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_and_saved_models"
TIMEOFRUN = "run_from_2024.06.11_15:59:19_UTC" 
SUBFOLDER = os.path.join(RUNFOLDER, TIMEOFRUN)
TESTRESULTNAME = "test_results.h5"


#load test results
def loadresults(subfolder, filename=TESTRESULTNAME):
    try:
        result_df = pd.read_hdf(os.path.join(subfolder, filename))
    except:
        raise("Couldn't find test results dataframe.")
    return result_df

def plot_resultsashisto(test_results, subfolder, backbone = "DynEdge", bins =100, prediction = 'energy_pred', truth = 'first_vertex_energy'):
    """
    Method to plot the difference between the true energy and the reconstructed/predicted energy as a histogramm.
    test_results: pandas dataframe that includes a prediction and a truth column 
    subfolder: Folder to save the histogram to 
    """
    # Filter out zero values
    zero_mask = (test_results[truth] == 0) #Create mask for entries that are 0 
    filtered_predictions = test_results[prediction][~zero_mask]
    filtered_truth = test_results[truth][~zero_mask]

    log_predictions = np.log10(filtered_predictions)
    log_truth = np.log10(filtered_truth)

    fig = plt.figure()
    plt.hist(log_predictions - log_truth, bins = bins, histtype = "step")
    plt.axvline(0, ls = '--', color = "black", alpha = 0.5)
    plt.title(f"Training parameter: {truth}")
    plt.xlabel('Reco. Energy - True Energy [log10 GeV]', size = 14)
    plt.ylabel('Amount', size = 14)
    name = f"EnergypredvsEnergyfromtruth_{backbone}.png"
    logger = Logger()
    try:
        temp = os.path.join(subfolder, name)
        fig.savefig(temp, dpi=300)
        message = colored("SUCCESSFUL:", "green") + f'histogram plotted to {temp}'
    except:
        message = colored("UNSUCCESSFUL:", "red") + 'Could not plot the histogram'

def plot_lossesandlearningrate(subfolder=SUBFOLDER):
    """
    Method to plot the losses and the learning rate during the GNN training 
    subfolder: Folder to save plots to
    """
    logger = Logger()
    timestamp = subfolder.split("/")[-1]
    # losses = pd.read_csv(f"{subfolder}/losses/version_0/metrics.csv")
    losses = pd.read_csv(os.path.join(subfolder, 'losses/version_0/metrics.csv'))

    if "train_loss" in losses.columns and "val_loss" in losses.columns:
        loss = np.array(losses["train_loss"].dropna())
        val_loss = np.array(losses["val_loss"].dropna())

        plt.figure(dpi=500)
        plt.plot(loss, color="navy", label="training loss")
        plt.plot(val_loss, color="crimson", label="validation loss")
        plt.grid()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(f"Losses \n {timestamp.replace('_', ' ')}")
        plt.legend()
        temp = os.path.join(subfolder, 'Losses.png')
        plt.savefig(temp)
        plt.close()
        message = colored("SUCCESSFUL:", "green") + f"losses plotted, saved to {temp}"
        logger.info(message)
    else:
        message = colored("ERROR:", "red") + "train_loss or val_loss not found"
        logger.info(message)

    if "lr" in losses.columns:
        lr = np.array(losses["lr"].dropna())

        plt.figure(dpi=500)
        plt.plot(lr, color="crimson")
        plt.grid()
        plt.xlabel("step")
        plt.ylabel("learning rate")
        plt.title(f"learning rate \n {timestamp.replace('_', ' ')}")
        plt.tight_layout()
        temp = os.path.join(subfolder, 'learning_rate.png')
        plt.savefig(temp)
        plt.close()
        message = colored("SUCCESSFUL:", "green") + f"learning rate plotted, saved to {temp}"
        logger.info(message)
    else:
        message = colored("ERROR:", "red") + "learning rate not found"
        logger.info(message)
 
#2d histogram plotting true energy vs reconstructed energy 
#normalise every column of the true energy 
def plotEtruevsEreco(dataframe, subfolder, normalise=['E_true']):
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
    fig, ax = plt.subplots()

    #adding line which showcases a perfect match in prediction and truth // identity line
    min_energy = min(energy_true.min(), energy_prediction.min())
    max_energy = max(energy_true.max(), energy_prediction.max())
    ax.plot([min_energy, max_energy], [min_energy, max_energy], 'r--', label='Identity Line')
    
    for i in range(len(normalise)):
        if (normalise[i]=='E_true'):
            filename = "2dHisto_EtruevsEreco_unweighted_normalisedalongEtrue.png" 
            #Create 2d histogram
            hist, xedges, yedges = np.histogram2d(energy_true, energy_prediction, bins=[bins_Etrue, bins_Epred])
            #Normalize histogram (columnwise)
            hist_normalized = hist / hist.sum(axis=1, keepdims=True)
            #TODO: Handle Zero division warnings

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
            hist_normalized = hist / hist.sum(axis=0, keepdims=True)
            #TODO: Handle Zero division warnings

            pc = plt.pcolormesh(xedges, yedges, hist_normalized.T, norm=colors.LogNorm(), 
                    cmap='viridis')
            
            # Adding a colorbar
            cbar = plt.colorbar(pc, ax=ax)
            cbar.ax.set_ylabel('Normalized Counts')
            ax.set_title(f"unweighted but normalised (with respect to reconstructed energy) plot \n of true energy vs predicted energy \n Training parameter: {truth} \n {timeofrun.replace('_', ' ')}")
        elif normalise[i] is None:
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
            save_path = os.path.join(subfolder, filename)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            message = colored("Successful:", "green") + f"Saved figure to {save_path}"
            logger.info(message)
        except:
            message = colored("ERROR:", "red") + "Couldn't save figure."
            logger.info(message)

        plt.show()

#Calculate Quantiles (eg middle 68% qunatile) with scipy.stats.iqr for the reconstructed energies (one iqr per bin) and plot iqr vs true energy
#iqr: width of the E reco distribution for a E true bin
def plotIQRvsEtrue(dataframe, subfolder, scale = ['linear', 'log'], bins=30):
    """
    Method to plot the interquartile range vs the true energy.
    The IQR (here middle 68 percentile) is here the width of the E_reco distribution for a true energy bin and acts as a 
    measure for reconstruction uncertainty.
    possible plots are looking at the IQR in logspace (applied the logarithmn before calculating the IQR) or 
    in linear space (IQR calculated for data without logarithmn)
    Binning for true energy is calculated as np.linspace(rounded_log_Etrue_min, rounded_log_Etrue_max, bins) either way 
    because that concerns the other axis 
    """
    prediction = dataframe.keys()[0]
    truth = dataframe.keys()[1]

    energy_prediction = dataframe[prediction]
    energy_true = dataframe[truth]
    #Get the logarithmic values for the enrgy prediction and truth 
    energy_prediction_log = np.log10(dataframe[prediction])
    energy_true_log = np.log10(dataframe[truth])
    
    #min value, max value -> define bins
    rounded_log_Etrue_min = np.floor(min(energy_true_log))
    rounded_log_Etrue_max = np.ceil(max(energy_true_log))
    e_log_bins = np.linspace(rounded_log_Etrue_min, rounded_log_Etrue_max, bins)

    #make sure scale is a list to iterate over it
    if isinstance(scale, str):
        scale = [scale]
    
    for s in scale:
        if s =='log':
            #in logspace:
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

            # Calculate the middle points of the bins for plotting
            e_log_bin_centers = (e_log_bins[:-1] + e_log_bins[1:]) / 2

            plt.figure(figsize=(10, 6))
            plt.plot(e_log_bin_centers, all_iqr, marker='o', ls='', color='b')
            plt.xlabel('True Energy (log GeV)')
            plt.ylabel('IQR of Reconstructed Energy (log GeV)')
            plt.title('IQR of Reconstructed Energy vs True Energy')
            plt.grid()
            filename = f'IQRvsEtrue_logarithmicEreco_{bins}bins'
            save_path = os.path.join(subfolder, filename)
            plt.savefig(save_path)
            plt.show()

        elif s =='linear':           
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

            # Calculate the middle points of the bins for plotting
            e_log_bin_centers = (e_log_bins[:-1] + e_log_bins[1:]) / 2
            plt.figure(figsize=(10, 6))
            plt.plot(e_log_bin_centers, all_iqr/np.power(10, e_log_bin_centers), marker='o', ls='', color='b')
            plt.xlabel('True Energy (log GeV)')
            plt.ylabel('IQR of Reconstructed Energy(GeV)/ True Energy at bin center(GeV) [arb. unit]')
            plt.title('normalized IQR of Reconstructed Energy vs True Energy')
            plt.grid()
            filename = f'IQRvsEtrue_linearEreco_{bins}bins_normalized'
            save_path = os.path.join(subfolder, filename)
            plt.savefig(save_path)
            plt.show()

def IQRvstruevertexpos():
    """
    Method to plot the performance of the GNN, measured as IQR (normalised with respect tp E_true or E_reco???), 
    vs the true vertex position to get a sense for the positional dependency of the reconstruction.
    Plots true vertex position in polar koordinates vs IQR
    """
    return

def main():  
    sub_folder = '/home/saturn/capn/capn108h/programming_GNNs_and_training/runs_and_saved_models/run_from_2024.06.16_10:30:38_UTC'
    df = loadresults(subfolder=sub_folder)
    # training_parameter = df.keys()[1]
    # plotEtruevsEreco(df, subfolder=sub_folder, normalise=None)
    # plotEtruevsEreco(df, subfolder=sub_folder, normalise='E_reco')
    # plot_lossesandlearningrate(sub_folder)
    # plot_resultsashisto(df, subfolder=sub_folder, truth=training_parameter)
    plotIQRvsEtrue(df, subfolder=sub_folder)



if __name__ == "__main__":
        main() 