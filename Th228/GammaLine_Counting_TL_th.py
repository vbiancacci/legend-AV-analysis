import pandas as pd
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import argparse
from scipy import optimize
from scipy import stats
import glob
import json
from datetime import datetime
from scipy.integrate import quad
import fnmatch
import sys

# import pygama.io.lh5 as lh5
import pygama.lh5 as lh5
from pygama.analysis import histograms
from pygama.analysis import peak_fitting

import pygama.genpar_tmp.cuts as cut

sys.path.insert(1,'/lfs/l1/legend/users/bianca/IC_geometry/analysis/myplot')
from myplot import *
myStyle = True
#Script to fit the gamma lines in the Ba133 spectra, for data and/or MC


def main():

    par = argparse.ArgumentParser(description="fit and count gamma lines in Ba133 spectrum",
                                  usage="python GammaLine_Counting.py [OPERATION: -d -s] [arguments: ]"
    )
    arg, st, sf = par.add_argument, "store_true", "store_false"
    arg("-d", "--data",  nargs=6, help="fit data, usage: python GammaLine_Counting.py --data <detector> <data_path> <calibration> <energy_filter> <cuts> <run>")
    arg("-s", "--sim", nargs=3, help="fit processed simulations, usage: python GammaLine_Counting.py --sim <detector> <sim_path> <MC_id>")

    args=vars(par.parse_args())

    #get path of this script:
    dir=os.path.dirname(os.path.realpath(__file__))
    print("working directory: ", dir)



    #========DATA MODE=======:
    if args["data"]:
        detector, data_path, calibration, energy_filter, cuts, run = args["data"][0], args["data"][1], args["data"][2], args["data"][3], args["data"][4], int(args["data"][5])
        print("")
        print("MODE: Data")
        print("detector: ", detector)
        print("data path: ", data_path)
        print("calibration path: ", calibration)
        print("energy filter: ", energy_filter)
        print("applying cuts: ", cuts)
        print("run: ", run)
        print("")
        if not os.path.exists(dir+"/PeakCounts/"+detector+"/TL/"):
            os.makedirs(dir+"/PeakCounts/"+detector+"/TL/")

        df=pd.read_hdf("data_calibration/"+detector+"/calibrated_energy_"+detector+"_"+energy_filter+"_run"+str(run)+".hdf5", key='energy')
        print(df.keys())
        energies=df['calib_energy']
        '''
        if cuts == "False":
            cuts = False
        else:
            cuts = True

        #initialise directories for detectors to save
        if not os.path.exists(dir+"/PeakCounts/"+detector+"/plots/data/"):
            os.makedirs(dir+"/PeakCounts/"+detector+"/plots/data/")

        #Get data and concoatonate into df
        if cuts == False:
            df_total_lh5 = read_all_dsp_lh5(data_path,cuts,run=run)
        else:
            sigma_cuts = 4
            print("sigma cuts: ", str(sigma_cuts))
            df_total_lh5 = read_all_dsp_lh5(data_path,cuts,run=run, sigma=sigma_cuts)


        print("df_total_lh5: ", df_total_lh5)
        energy_filter_data = df_total_lh5[energy_filter]

        #Get Calibration
        with open(calibration) as json_file:
            calibration_coefs = json.load(json_file)
        # m = calibration_coefs[energy_filter]["Calibration_pars"][0]
        # c = calibration_coefs[energy_filter]["Calibration_pars"][1]
        m = calibration_coefs[energy_filter]["calibration"][0]
        c = calibration_coefs[energy_filter]["calibration"][1]

        # energies = (energy_filter_data-c)/m
        energies = energy_filter_data*m + c
        '''
    #========SIMULATION MODE:==============
    if args["sim"]:
        detector, sim_path, MC_id = args["sim"][0], args["sim"][1], args["sim"][2]
        print("")
        print("MODE: Simulations")
        print("detector: ", detector)
        print("sim path: ", sim_path)
        print("MC_id: ", MC_id)
        print("")

       #initialise directories for detectors to save
        #if not os.path.exists(dir+"/PeakCounts/"+detector+"/plots/sim/"):
        #    os.makedirs(dir+"/PeakCounts/"+detector+"/plots/sim/")

        #get energies
        # MC_file = hdf5_path+"processed_detector_"+MC_file_id+'_FCCD'+str(FCCD)+'mm_DLF'+str(DLF)+'.hdf5'
        df =  pd.read_hdf(sim_path, key="procdf")
        energies = df['energy']


    #get total pygama histogram
    binwidth = 0.1 #keV
    bins = np.arange(0,2700,binwidth)
    hist_peak, bins_peak, var = histograms.get_hist(energies, bins=bins)
    val1=2610
    val2=2620
    val1_b=2500
    val2_b=2600
    counts_peak, counts_peak_err = area_between_vals(hist_peak, bins_peak, val1, val2)
    print(counts_peak)
    counts_bkg, counts_bkg_err = area_between_vals(hist_peak, bins_peak, val1_b, val2_b)
    print(counts_bkg)
    observable=counts_bkg/counts_peak
    observable_err=observable*np.sqrt((counts_peak_err/counts_peak)**2 + (counts_bkg_err/counts_bkg)**2)

    PeakCounts = {
        "Obs" : observable,
        "Obs_err" : observable_err,
    }

    if args["sim"]:
        with open(dir+"/PeakCounts/"+detector+"/TL/PeakCounts_sim_"+MC_id+".json", "w") as outfile:
            json.dump(PeakCounts, outfile, indent=4)
    if args["data"]:
        with open(dir+"/PeakCounts/"+detector+"/TL/PeakCounts_data.json", "w") as outfile:
            json.dump(PeakCounts, outfile, indent=4)




def read_all_dsp_lh5(t2_folder, cuts, cut_file_path=None, run="all", sigma=4):

    sto = lh5.Store()
    files = os.listdir(t2_folder)
    files = fnmatch.filter(files, "*lh5")
    if run == 1:
        files = fnmatch.filter(files, "*run0001*")
    if run == 2:
        files = fnmatch.filter(files, "*run0002*")

    df_list = []

    if cuts == False:
        for file in files:

            #get data, no cuts
            tb = sto.read_object("raw",t2_folder+file)[0]
            df = lh5.Table.get_dataframe(tb)
            df_list.append(df)

        df_total = pd.concat(df_list, axis=0, ignore_index=True)
        return df_total

    else: #apply cuts
        files = [t2_folder+file for file in files] #get list of full paths
        lh5_group = "raw"
        # df_total_cuts, failed_cuts = cut.load_df_with_cuts(files, lh5_group, cut_file = cut_file_path, cut_parameters= {'bl_mean':sigma,'bl_std':sigma, 'pz_std':sigma}, verbose=True)
        df_total_cuts, failed_cuts = cut.load_df_with_cuts(files, lh5_group, cut_parameters= {'bl_mean':sigma,'bl_std':sigma, 'pz_std':sigma}, verbose=True)

        return df_total_cuts


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
    area = sum(counts[left_bin_edge_index:right_bin_edge_index]/bin_width)
    area_err =  sum (np.sqrt(counts[left_bin_edge_index:right_bin_edge_index]/bin_width))
    return area, area_err


if __name__ =="__main__":
    main()
