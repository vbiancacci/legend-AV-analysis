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

CodePath=dir=os.path.dirname(os.path.realpath(__file__))

def main():

    detector = "V05266A"
    FCCD = 1.5

    #==========DATA==========
    #Parameters
    data_path="/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/gen/"+detector+"/tier2/ba_HS4_top_dlt/"
    energy_filter="cuspEmax_ctc"
    cuts="True"
    if (detector[2] == "7") or (detector[2] =="8"):
        run = 2
    else:
        run=1
    calibration = CodePath+"/data_calibration/"+detector+"/calibration_run"+str(run)+"_cuts.json"

    #get data
    df_total_lh5 = read_all_dsp_lh5(data_path,cuts=cuts,run=run)
    energy_filter_data = df_total_lh5[energy_filter]
    #calibrate data
    with open(calibration) as json_file:
        calibration_coefs = json.load(json_file)
    m = calibration_coefs[energy_filter]["calibration"][0]
    c = calibration_coefs[energy_filter]["calibration"][1]
    energy_data = energy_filter_data*m + c

    #get peak counts
    PeakCounts_data = dir+"/PeakCounts/"+detector+"/PeakCounts_data_"+detector+"_cuts_"+energy_filter+"_run"+str(run)+".json"
    with open(PeakCounts_data) as json_file:
        PeakCounts = json.load(json_file)
    C_356_data = PeakCounts['C_356']

    #initialise histogram:
    fig = plt.figure()
    binwidth = 0.1 #keV
    bins = np.arange(0,450,binwidth)
    counts_data, bins, bars_data = plt.hist(energy_data, bins=bins,  label = "Data", histtype = 'step', linewidth = '0.35')


    #===========MC============
    #paramaters
    #DLF_list = [0.0,0.25,0.5,0.75,1.0]
    alpha_list= [0.8, 0.9]
    beta_list= [0.3, 0.4]
    smear="g"
    frac_FCCDbore=1.0
    TL_model="ExLinT"

    #get peak counts - for notl, dlf=1
    #MC_id=detector+"-ba_HS4-top-0r-81z_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+"mm_alpha"+alpha+"_beta"+beta+"_fracFCCDbore"+str(frac_FCCDbore)
    #PeakCounts_MC = dir+"/PeakCounts/"+detector+"/PeakCounts_sim_"+MC_id+".json"
    #with open(PeakCounts_MC) as json_file:
    #    PeakCounts = json.load(json_file)
    #C_356_MC = PeakCounts['C_356']

    for alpha in alpha_list:
        for beta in beta_list:
            MC_id=detector+"-ba_HS4-top-0r-81z_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+"mm__alpha"+str(alpha)+"_beta"+str(beta)+"_fracFCCDbore"+str(frac_FCCDbore)
            sim_path="/lfs/l1/legend/detector_char/enr/hades/simulations/legend-g4simple-simulation/simulations/"+detector+"/ba_HS4/top_0r_81z/hdf5/AV_processed_test/"+MC_id+".hdf5"
            df =  pd.read_hdf(sim_path, key="procdf")
            energy_MC = df['energy']

            PeakCounts_MC = dir+"/PeakCounts/"+detector+"/PeakCounts_sim_"+MC_id+".json"
            with open(PeakCounts_MC) as json_file:
                PeakCounts = json.load(json_file)
            C_356_MC = PeakCounts['C_356']

        #plot MC
            counts_MC, bins, bars = plt.hist(energy_MC, bins = bins, weights=(C_356_data/C_356_MC)*np.ones_like(energy_MC), label = "MC: alpha "+str(alpha)+", beta: "+str(beta)+" (scaled)", histtype = 'step', linewidth = '0.35')


    plt.xlabel("Energy [keV]")
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.legend(loc = "lower left")
    plt.title(detector)
    plt.xlim(0,450)
    plt.tight_layout()
    plt.savefig(dir+"/Spectra/"+detector+"/DataMC_compareDLFs_IanModel.png")

    plt.show()



if __name__ == "__main__":
    main()
