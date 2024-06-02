import numpy as np 
import matplotlib.pyplot as plt
from termcolor import colored
import os
import pandas as pd


def plot_result(test_results, folder, backbone = "DynEdge", bins =100, prediction = 'energy_pred', reco = 'primary_energy'):
        fig = plt.figure()
        plt.hist(np.log10(test_results[prediction])-np.log10(test_results[reco]), bins = bins, histtype = "step")
        plt.axvline(0, ls = '--', color = "black", alpha = 0.5)
        plt.xlabel('Reco. Energy - True Energy [log10 GeV]', size = 14)
        plt.ylabel('Amount', size = 14)
        name = f"EnergypredvsEnergyprim_{backbone}.png"
        fig.savefig(os.path.join(folder, name), dpi=300)



def plot_losses(folder):
    run=folder.split("/")[-1]

    losses=pd.read_csv(f"{folder}/losses/version_0/metrics.csv")

    try:
        loss=np.array(losses["train_loss"])
        loss=loss[~np.isnan(loss)]

        val_loss=np.array(losses["val_loss"])
        val_loss=val_loss[~np.isnan(val_loss)]

        plt.figure(dpi=500)
        plt.plot(loss,color="navy",label="training loss")
        plt.plot(val_loss,color="crimson",label="validation loss")
        plt.grid()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(f"Losses \n {run}")
        plt.legend()
        plt.savefig(f"{folder}/Losses.png")

        print(colored("SUCCESSFUL:","green"),f"losses plotted, saved to {run}/Losses.png")
    except:
        print(colored("ERROR:","red"),"test_loss not found")
    
    try:
        lr=np.array(losses["lr"])
        lr=lr[~np.isnan(lr)]

        plt.figure(dpi=500)
        plt.plot(lr,color="crimson")
        plt.grid()
        plt.xlabel("step")
        plt.ylabel("learning rate")
        plt.title(f"learning rate \n {run}")
        plt.tight_layout()
        plt.savefig(f"{folder}/learning_rate.png")

        print(colored("SUCCESSFUL:","green"),f"learning rate plotted, saved to {run}/learning_rate.png")
    except:
        print(colored("ERROR:","red"),"learning rate not found")

