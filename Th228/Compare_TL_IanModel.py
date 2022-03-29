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

from GammaLine_Counting_Th import read_all_dsp_lh5

sys.path.insert(1,'/lfs/l1/legend/users/bianca/IC_geometry/analysis/myplot')
from myplot import *

myStyle = True

CodePath=dir=os.path.dirname(os.path.realpath(__file__))



def main():

    detector = "V02160A"
    FCCD = 0.7
    energy_filter="cuspEmax_ctc"
    run=2

    #==========DATA==========
    df=pd.read_hdf(CodePath+"/data_calibration/"+detector+"/calibrated_energy_"+detector+"_"+energy_filter+"_run"+str(run)+".hdf5", key='energy')
    print(df.keys())
    energy_data=df['calib_energy']


    '''
    #Parameters
    data_path="/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/gen/"+detector+"/tier2/th_HS2_top_psa/"
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
    '''
    #initialise histogram:
    #fig = plt.figure()
    p =Plot((12,8),n=1)
    ax0=p.ax

    binwidth = 1 #keV
    bins = np.arange(0,2700,binwidth)
    counts_data, bins, bars_data = ax0.hist(energy_data, bins=bins,  label = "Data", histtype = 'step', linewidth = '0.35')
    C_2100_data = area_between_vals(counts_data, bins, 1000, 2200)

    #===========MC============
    #paramaters
    #DLF_list = [0.0,0.25,0.5,0.75,1.0]
    alpha_list = [0.4] #0.2, 0.3, 0.4, 0.5, 0.6]
    beta_list = [0.2] #, 0.3, 0.4, 0.5, 0.6]
    smear="g"
    frac_FCCDbore=1.5
    TL_model="ExLinT"

    #get peak counts - for notl, dlf=1
    #MC_id=detector+"-ba_HS4-top-0r-81z_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+"mm_alpha"+alpha+"_beta"+beta+"_fracFCCDbore"+str(frac_FCCDbore)
    #PeakCounts_MC = dir+"/PeakCounts/"+detector+"/PeakCounts_sim_"+MC_id+".json"
    #with open(PeakCounts_MC) as json_file:
    #    PeakCounts = json.load(json_file)
    #C_356_MC = PeakCounts['C_356']

    for alpha in alpha_list:
        for beta in beta_list:
            MC_id=detector+"-th_HS2-top-0r-42z_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+"mm__alpha"+str(alpha)+"_beta"+str(beta)+"_fracFCCDbore"+str(frac_FCCDbore)
            sim_path="/lfs/l1/legend/detector_char/enr/hades/simulations/legend-g4simple-simulation/simulations/"+detector+"/th_HS2/top_0r_42z/hdf5/AV_processed_test/"+MC_id+".hdf5"
            try:
                print (sim_path)
                df =  pd.read_hdf(sim_path, key="procdf")
            except:
                print ("no")
                continue

            energy_MC = df['energy']

#            PeakCounts_MC = dir+"/PeakCounts/"+detector+"/PeakCounts_sim_"+MC_id+".json"
#            with open(PeakCounts_MC) as json_file:
#                PeakCounts = json.load(json_file)
#            C_356_MC = PeakCounts['C_356']
        #plot MC
            counts_MC, bins = np.histogram(energy_MC, bins = bins)
            C_2100_MC = area_between_vals(counts_MC, bins, 1000, 2200)
            scaled_counts_MC, bins, bars = ax0.hist(energy_MC, bins = bins,  label = "g4simple simulation", weights=(C_2100_data/C_2100_MC)*np.ones_like(energy_MC), histtype = 'step', linewidth = '0.55')
            for model, observed in zip(scaled_counts_MC, counts_data):
                chi2=(model-observed)**2/observed
            print(alpha, beta, chi2)


    ax0.set_xlabel("Energy [keV]",  family='serif')
    ax0.set_ylabel("Counts", family='serif')
    ax0.set_yscale("log")
    #p.legend(ncol=1, out=False, pos = "lower left")
    p.pretty(large=8, grid=False)

    plt.savefig(dir+"/Spectra/"+detector+"/DataMC_compareDLFs_IanModel.png")

    plt.show()



def find_bin_idx_of_value(bins, value):
    """Finds the bin which the value corresponds to."""
    array = np.asarray(value)
    idx = np.digitize(array,bins)
    return idx-1

def area_between_vals(counts, bins, val1, val2):
    """Calculates the area of the hist between two certain values"""
    left_bin_edge_index = find_bin_idx_of_value(bins, val1)
    right_bin_edge_index = find_bin_idx_of_value(bins, val2)
    bin_width = np.diff(bins)[0]
    area = sum(bin_width * counts[left_bin_edge_index:right_bin_edge_index])
    return area

if __name__ == "__main__":
    main()
