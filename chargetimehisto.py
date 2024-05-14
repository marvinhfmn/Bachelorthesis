#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
params = {'text.latex.preamble' :  r'\usepackage{amsmath}'}
plt.rcParams.update(params)
from IPython.display import display, Latex, Math, Markdown
import sqlite3 as sql
from glob import glob
import pandas as pd
import random
import os
from time import gmtime, strftime
import pickle 
from tqdm import tqdm
import graphnet
import warnings
import tables as pytables
from pathlib import Path

bla = 1

all_files = glob("/home/wecapstor3/capn/capn106h/sim_databases/*/*/*.db")
corsika_files = glob("/home/wecapstor3/capn/capn106h/sim_databases/22615/*/*.db")
nugen_files = [file for file in all_files if file not in corsika_files]


#%%
#simple histogram plot: charge pro bin vs time for one DOM and one event for throughgoing
@profile # important for terminal command kernprof -vl chargetimehisto.py 
def getListofChargeTimeDataframes(dbfiles, table = "InIceDSTPulses", truth_table = "truth", classification = 1, E_threshold = 1e7, sumcharge_threshold = 1000):
    indices = []
    dflist = []
    path = "/home/hpc/capn/capn108h/programs/"
    filename = "PandasDataframes_Ethresh{:.0e}_sumchargethresh{}.h5".format(E_threshold, sumcharge_threshold)
    with pd.HDFStore(os.path.join(path, filename), 'w') as store:
        for dbfile in tqdm(dbfiles):
            con = sql.connect(dbfile)
            cur = con.cursor()
            if classification is not None:
                ids = cur.execute(f"SELECT DISTINCT(event_no) FROM {truth_table} WHERE classification_50m={classification} AND primary_energy>={E_threshold}")
            else:
                ids = cur.execute(f"SELECT DISTINCT(event_no) FROM {truth_table} WHERE primary_energy>={E_threshold}")

            ids = ids.fetchall()
            ids = [[dbfile, int(event_id[0])] for event_id in ids]
            indices += ids
            
            for (databasefile, event_id) in ids:

                # print(f"{event_id} {dbfile}")
                event_df = pd.read_sql_query(f"SELECT dom_number, SUM(charge) FROM {table} WHERE event_no={event_id} AND (is_errata_dom = -1.0 OR is_errata_dom = 0.0) GROUP BY dom_number", con)
                temp = event_df['SUM(charge)'].idxmax() #Select index of event with the most deposited charge
                dom_numb = event_df['dom_number'][temp] #Select DOM with the most deposited charge
                timecharge_df = pd.read_sql_query(f"SELECT dom_time, charge FROM {table} WHERE event_no={event_id} AND (is_errata_dom = -1.0 OR is_errata_dom = 0.0) AND dom_number = {dom_numb}", con) #get times and charges at one specific DOM for event 
                # print(timecharge_df, "\n\n")
                temp = np.max(np.array(event_df['SUM(charge)'], dtype=np.float64))

                # print(temp)
                # dflist.append((temp, timecharge_df)) #results in a list of tuples of (sum(charge) and pd dataframes)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', pytables.NaturalNameWarning)
                    store[f'{dbfile}_id{event_id}'] = timecharge_df
        # print(dflist)
        # dflist_final = [timecharge_df for (sumcharge, timecharge_df) in dflist if sumcharge >= sumcharge_threshold] #generates a list of dataframes where the total charge exceed a defined threshold charge
        #save that dflist maybe? so that it doesn't take ages to load the next time 
   

def plotChargeTimeHistogram(choice, seed):
    totalcharge = choice[0]
    timecharge_dataframe = choice[1]
    print([timecharge_dataframe['charge']])
   
    # Look for specific values of sum(charge) (1e4) out of the dflist (list comprehension or dictionary) and choose a random event of that subset -> plot it as histogram
    
    #ChargeTime-Histogramm: Showcases something approximating the waveform of the signal: adding up the pulses
    fig, ax = plt.figure()
    plt.grid()
    plt.xlabel(r'time [ns]')
    plt.ylabel("charge [PE]")
    plt.title(f"Charge/time Histogramm for one DOM \n total charge: {np.round(totalcharge, decimals=3)}")
    timebins= np.linspace(timecharge_dataframe['dom_time'].min(), timecharge_dataframe['dom_time'].max(), num= 30) 
    plt.hist(timecharge_dataframe['dom_time'], weights=timecharge_dataframe['charge'], bins=timebins)

    # dbstring = dbfile.split("/")[-1]
    plot_folder = "/home/hpc/capn/capn108h/programs/plots/"
    now=strftime("%d.%m.%Y_%H:%M:%S",gmtime())
    filename = f"ChargeTimeHistogram_for_{now}.png"
    plt.savefig(os.path.join(plot_folder, filename))
    plt.close()

    #Scatterplot:  showcases pulses 
    fig = plt.figure()
    plt.grid()
    plt.xlabel(r'relative time [ns]')
    plt.ylabel("charge [PE]")
    plt.title(f"Charge/time Scatterplot for one DOM \n total charge: {np.round(totalcharge, decimals=3)}")
    relativetimes = timecharge_dataframe['dom_time'] - timecharge_dataframe['dom_time'].min()
    # plt.scatter(relativetimes,timecharge_dataframe['charge'], alpha = 0.5)
    plt.vlines(relativetimes,ymin = 0, ymax = timecharge_dataframe['charge'], alpha = 0.5, color="gray")
    # dbstring = dbfile.split("/")[-1]
    plot_folder = "/home/hpc/capn/capn108h/programs/plots/"
    now=strftime("%d.%m.%Y_%H:%M:%S",gmtime())
    filename = f"ChargeTimeScatterplot_for_{now}.png"
    plt.savefig(os.path.join(plot_folder, filename))
    plt.close()



def main():
    classification = 1
    E_threshold = 1e7 #GeV
    sumcharge_threshold = 1000 #PE
    seed = 10 

    getListofChargeTimeDataframes(all_files, classification = classification, E_threshold = E_threshold, sumcharge_threshold = sumcharge_threshold)
    filename = Path(f"/home/hpc/capn/capn108h/programs/PandasDataframes_Ethresh{E_threshold:.0e}_sumchargethresh{sumcharge_threshold}.h5")

    # define parameters -> check wheter h5 table exists for parameters -> if not: call getListofChargeTimeDataframes method |
    # else choose key and call plot method
    choice = random.choice(dflist)
    # my_file = Path("/path/to/file")
    if filename.is_file():
        plotChargeTimeHistogram()
        # file exists   
    # filename = "/home/hpc/capn/capn108h/programs/PandasDataframes_Ethresh10000000.0_sumchargethresh1000.h5"
    # with pd.HDFStore(filename, 'r') as store:
    #     keys = store.keys()
    #     print(keys[:10])
    #random.seed(10)
    #random.choice(keys)

if __name__ == "__main__":
    main()
# %%
