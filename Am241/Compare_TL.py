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


CodePath=dir=os.path.dirname(os.path.realpath(__file__))

def main():

    detector = "V07647A"
    FCCD = 1.41

    #==========DATA==========
    #Parameters
    energy_filter="cuspEmax_ctc"
    cuts="True"
    run=1

    #get data
    df=pd.read_hdf(dir+"/data_calibration/"+detector+"/loaded_energy_"+detector+"_"+energy_filter+"_run"+str(run)+".hdf5", key='energy')
    energy_filter_data=df['energy_filter']

    #calibrate data
    calibration = CodePath+"/data_calibration/"+detector+"/calibration_run1_cuts.json"
    with open(calibration) as json_file:
        calibration_coefs = json.load(json_file)
    m = calibration_coefs[energy_filter]["calibration"][0]
    c = calibration_coefs[energy_filter]["calibration"][1]
    energy_data = energy_filter_data*m + c

    #get peak counts
    PeakCounts_data = dir+"/PeakCounts/"+detector+"/PeakCounts_data_"+detector+"_cuts_"+energy_filter+"_run"+str(run)+".json"
    with open(PeakCounts_data) as json_file:
        PeakCounts = json.load(json_file)
    C_60_data = PeakCounts['C_60']

    #initialise histogram:
    fig = plt.figure()
    binwidth = 0.1 #keV
    bins = np.arange(0,120,binwidth)
    counts_data, bins, bars_data = plt.hist(energy_data, bins=bins,  label = "Data", histtype = 'step', linewidth = '0.35')


    #===========MC============
    #paramaters
    DLF_list =[0.5]#][0.0, 0.1, 0.2 ,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    smear="g"
    frac_FCCDbore=0.5
    TL_model="l"

    #get peak counts - for notl, dlf=1
    MC_id=detector+"-am_HS6-top-0r-198z_"+smear+"_notl_FCCD"+str(FCCD)+"mm_DLF1.0_fracFCCDbore"+str(frac_FCCDbore)
    PeakCounts_MC = dir+"/PeakCounts/"+detector+"/PeakCounts_sim_"+MC_id+".json"
    with open(PeakCounts_MC) as json_file:
        PeakCounts = json.load(json_file)
    C_60_MC = PeakCounts['C_60']

    for DLF in DLF_list:

        #get MC
        if DLF == 1.0:
            MC_id=detector+"-am_HS6-top-0r-198z_"+smear+"_notl_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_fracFCCDbore"+str(frac_FCCDbore)
        else:
            MC_id=detector+"-am_HS6-top-0r-198z_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_fracFCCDbore"+str(frac_FCCDbore)
        sim_path="/lfs/l1/legend/detector_char/enr/hades/simulations/legend-g4simple-simulation/simulations/"+detector+"/am_HS6/top_0r_198z/hdf5/AV_processed/"+MC_id+".hdf5"
        df =  pd.read_hdf(sim_path, key="procdf")
        energy_MC = df['energy']

        #plot MC
        counts_MC, bins, bars = plt.hist(energy_MC, bins = bins, weights=(C_60_data/C_60_MC)*np.ones_like(energy_MC), label = "MC: FCCD "+str(FCCD)+"mm, DLF: "+str(DLF)+" (scaled)", histtype = 'step', linewidth = '0.35')


    plt.xlabel("Energy [keV]")
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.legend(loc = "lower left")
    plt.title(detector)
    plt.xlim(0,120)
    plt.tight_layout()
    plt.savefig(dir+"/Spectra/"+detector+"/DataMC_compareDLFs.png")

    plt.show()



if __name__ == "__main__":
    main()
