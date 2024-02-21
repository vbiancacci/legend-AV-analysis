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

from GammaLine_Counting_Ba133 import read_all_dsp_lh5

#Script to plot spectra of MC (best fit FCCD) and data

def main():


    if(len(sys.argv) != 11):
        print('Example usage: python AV_postproc.py <detector> <MC_id> <sim_path> <FCCD> <DLF> <data_path> <calibration> <cuts> <run>')
        sys.exit()

    detector = sys.argv[1] #raw MC folder path - e.g. "/lfs/l1/legend/users/aalexander/legend-g4simple-simulation/legend/simulations/V02160A/ba_HS4/top_0r_78z/hdf5/"
    MC_id = sys.argv[2]     #file id, including fccd config - e.g. ${detector}-ba_HS4-top-0r-78z_${smear}_${TL_model}_FCCD${FCCD}mm_DLF${DLF}_fracFCCDbore${frac_FCCDbore}
    sim_path = sys.argv[3] #path to AV processed sim, e.g. /lfs/l1/legend/users/aalexander/legend-g4simple-simulation/legend/simulations/${detector}/ba_HS4/top_0r_78z/hdf5/AV_processed/${MC_id}.hdf5
    FCCD = sys.argv[4] #FCCD of MC - e.g. 0.69
    DLF = sys.argv[5] #DLF of MC - e.g. 1.0
    data_path = sys.argv[6] #path to data
    calibration = sys.argv[7] #path to data calibration
    energy_filter = sys.argv[8] #energy filter - e.g. trapEftp
    cuts = sys.argv[9] #e.g. False
    run = int(sys.argv[10]) #data run, e.g. 1 or 2

    print("detector: ", detector)
    print("MC_id: ", MC_id)
    print("sim_path: ", sim_path)
    print("FCCD: ", str(FCCD))
    print("DLF: ", str(DLF))
    print("data_path: ", data_path)
    print("calibration: ", calibration)
    print("energy_filter: ", energy_filter)
    print("applying data cuts: ", cuts)
    print("run: ", run)

    if cuts == "False":
        cuts = False
    else:
        cuts = True

    dir=os.path.dirname(os.path.realpath(__file__))
    print("working directory: ", dir)

    #initialise directories to save spectra
    if not os.path.exists(dir+"/Spectra/"+detector+"/"):
        os.makedirs(dir+"/Spectra/"+detector+"/")

    print("start...")


    #GET DATA
    #Get data and concoatonate into df
    if cuts == False:
        df_total_lh5 = read_all_dsp_lh5(data_path,cuts=cuts,run=run)
    else:
        df_total_lh5 = read_all_dsp_lh5(data_path,cuts=cuts,run=run)

    energy_filter_data = df_total_lh5[energy_filter]

    #Get Calibration
    with open(calibration) as json_file:
        calibration_coefs = json.load(json_file)
    # m = calibration_coefs[energy_filter]["Calibration_pars"][0]
    # c = calibration_coefs[energy_filter]["Calibration_pars"][1]
    m = calibration_coefs[energy_filter]["calibration"][0]
    c = calibration_coefs[energy_filter]["calibration"][1]

    # energy_data = (energy_filter_data-c)/m
    energy_data = energy_filter_data*m + c

    #GET MC
    df =  pd.read_hdf(sim_path, key="procdf")
    energy_MC = df['energy']
    print("opened MC")

    #Get peak counts C_356 for scaling
    if cuts == False:
        PeakCounts_data = dir+"/PeakCounts/"+detector+"/PeakCounts_data_"+detector+"_"+energy_filter+"_run"+str(run)+".json"
    else:
        PeakCounts_data = dir+"/PeakCounts/"+detector+"/PeakCounts_data_"+detector+"_cuts_"+energy_filter+"_run"+str(run)+".json"

    PeakCounts_MC = dir+"/PeakCounts/"+detector+"/new/PeakCounts_sim_"+MC_id+".json"

    with open(PeakCounts_data) as json_file:
        PeakCounts = json.load(json_file)
        C_356_data = PeakCounts['C_356']

    with open(PeakCounts_MC) as json_file:
        PeakCounts = json.load(json_file)
        C_356_MC = PeakCounts['C_356']

    print("got peak counts")

    #Plot data and scaled MC
    binwidth = 0.1 #keV
    bins = np.arange(0,450,binwidth)

    fig, ax =plt.subplots()
    #fig = plt.figure()
    #gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    #ax0 = plt.subplot(gs[0])
    #ax1 = plt.subplot(gs[1], sharex = ax0)

    counts_data, bins, bars_data = ax.hist(energy_data, bins=bins,  label = "Data", histtype = 'step', linewidth = '0.35')
    counts_MC, bins, bars = ax.hist(energy_MC, bins = bins, weights=(C_356_data/C_356_MC)*np.ones_like(energy_MC), label = "G4simple simulation", histtype = 'step', linewidth = '0.35')

    print("basic histos complete")

    Data_MC_ratios = []
    Data_MC_ratios_err = []
    for index, bin in enumerate(bins[1:]):
        data = counts_data[index]
        MC = counts_MC[index] #This counts has already been scaled by weights
        if MC == 0:
            ratio = 0.
            error = 0.
        else:
            try:
                ratio = data/MC
                try:
                    error = np.sqrt(1/data + 1/MC)
                except:
                    error = 0.
            except:
                ratio = 0 #if MC=0 and dividing by 0
        Data_MC_ratios.append(ratio)
        Data_MC_ratios_err.append(error)

    print("errors")

    #ax1.errorbar(bins[1:], Data_MC_ratios, yerr=Data_MC_ratios_err,color="green", elinewidth = 1, fmt='x', ms = 1.0, mew = 1.0)
    #ax1.hlines(1, 0, 450, colors="gray", linestyles='dashed')


    plt.xlabel("Energy [keV]")
    ax.set_ylabel("Counts / 0.1keV")
    ax.set_yscale("log")
    ax.legend(frameon=False, loc = "lower left")
    plt.show()
    #ax0.set_title(detector)
    #ax1.set_ylabel("data/MC")
    #ax1.set_yscale("log")
    #ax1.set_xlim(0,450)
    #ax0.set_xlim(0,450)

    # plt.subplots_adjust(hspace=.0)

    if cuts == False:
        plt.savefig(dir+"/Spectra/"+detector+"/DataMC_"+MC_id+"_"+energy_filter+"_run"+str(run)+".pdf")
    else:
        plt.savefig(dir+"/Spectra/"+detector+"/DataMC_"+MC_id+"_"+energy_filter+"_run"+str(run)+"_cuts.png")

    print("done")



if __name__ == "__main__":
    main()
