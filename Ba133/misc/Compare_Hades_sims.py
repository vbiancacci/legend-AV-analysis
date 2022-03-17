import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime
import json
import argparse
import os
from scipy import optimize
from scipy import stats


def main(): 

    detector = "V08682A"
    source_z = "88z"
    FCCD = 0.0
    DLF = 1.0
    MC_id=detector+"-ba_HS4-top-0r-"+source_z+"_g_notl_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_fracFCCDbore0.5"

    #Normal sim
    sim_path = "/lfs/l1/legend/users/aalexander/legend-g4simple-simulation/simulations/"+detector+"/ba_HS4/top_0r_"+source_z+"/hdf5/AV_processed/"+MC_id+".hdf5"
    df =  pd.read_hdf(sim_path, key="procdf")
    energy_MC = df['energy']
    print(energy_MC.shape)

    #new Ba source
    sim_path = "/lfs/l1/legend/users/aalexander/HADES_test_sims/new_ba_source/legend-g4simple-simulation/simulations/"+detector+"/ba_HS4/top_0r_"+source_z+"/hdf5/AV_processed/"+MC_id+".hdf5"
    df_newBa =  pd.read_hdf(sim_path, key="procdf")
    energy_MC_newBa = df['energy']
    print(energy_MC_newBa.shape)
    

    #Ba 360 degrees - might need to x2
    sim_path = "/lfs/l1/legend/users/aalexander/HADES_test_sims/ba_360_angle/legend-g4simple-simulation/simulations/"+detector+"/ba_HS4/top_0r_"+source_z+"/hdf5/AV_processed/"+MC_id+".hdf5"
    df_360 =  pd.read_hdf(sim_path, key="procdf")
    energy_MC_360 = df['energy']
    print(energy_MC_360.shape)

    binwidth = 0.1 #keV
    bins = np.arange(0,450,binwidth)


    fig = plt.figure()
    counts_MC, bins, bars = plt.hist(energy_MC, bins = bins, label = "MC baseline", histtype = 'step', linewidth = '0.35')
    counts_MC_newBa, bins, bars = plt.hist(energy_MC_newBa, bins = bins, label = "MC new Ba source", histtype = 'step', linewidth = '0.35')
    # counts_MC_360, bins, bars = ax0.hist(energy_MC_360, bins = bins, label = "MC Ba 360 deg", histtype = 'step', linewidth = '0.35')

    plt.xlabel("Energy [keV]")
    plt.ylabel("counts")
    plt.yscale("log")
    plt.legend()
    plt.xlim(0,450)
    plt.title(detector+", No FCCD")

    fig2 = plt.figure()
    counts_MC, bins, bars = plt.hist(energy_MC, bins = bins, label = "MC baseline", histtype = 'step', linewidth = '0.35')
    # counts_MC_newBa, bins, bars = plt.hist(energy_MC_newBa, bins = bins, label = "MC new Ba source", histtype = 'step', linewidth = '0.35')
    counts_MC_360, bins, bars = plt.hist(energy_MC_360, bins = bins, label = "MC Ba 360 deg", histtype = 'step', linewidth = '0.35')

    plt.xlabel("Energy [keV]")
    plt.ylabel("counts")
    plt.yscale("log")
    plt.legend()
    plt.xlim(0,450)
    plt.title(detector+", No FCCD")

    fig3 = plt.figure()
    counts_MC, bins, bars = plt.hist(energy_MC, bins = bins, label = "MC baseline", histtype = 'step', linewidth = '0.35')
    # counts_MC_newBa, bins, bars = plt.hist(energy_MC_newBa, bins = bins, label = "MC new Ba source", histtype = 'step', linewidth = '0.35')
    # counts_MC_360, bins, bars = plt.hist(energy_MC_360, bins = bins, label = "MC Ba 360 deg", histtype = 'step', linewidth = '0.35')

    plt.xlabel("Energy [keV]")
    plt.ylabel("counts")
    plt.yscale("log")
    plt.legend()
    plt.xlim(0,450)
    plt.title(detector)



    plt.show()



if __name__ == "__main__":
    main()