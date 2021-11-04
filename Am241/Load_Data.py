import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pygama.analysis.histograms as pgh
import pygama.analysis.calibration as cal
import pygama.analysis.peak_fitting as pgp
import json
import fnmatch
import h5py
import pandas as pd

import pygama.lh5 as lh5
import pygama.genpar_tmp.cuts as cut

#Script to calibrate Ba133 pygama data and obtain resolution fit coefficients

CodePath=os.path.dirname(os.path.realpath(__file__))

def main():

    if(len(sys.argv) != 6):
        print('Example usage: python Load_Data.py <detector> <data_path> <energy_filter> <cuts> <run>')
        sys.exit()

    detector = sys.argv[1]
    data_path = sys.argv[2]
    energy_filter = sys.argv[3]
    cuts = sys.argv[4]
    run = int(sys.argv[5])

    print("")
    print("detector: ", detector)
    print("data path: ", data_path)
    print("energy_filter: ", energy_filter)
    print("applying cuts: ", cuts)
    print("data run: ", run)
    if cuts == "False":
        cuts = False
    else:
        cuts = True

    #initialise directories for detectors to save
    if not os.path.exists(CodePath+"/data_calibration/"+detector+"/plots/"):
        os.makedirs(CodePath+"/data_calibration/"+detector+"/plots/")

    #====Load data======
    print(" ")
    print("Loading data")
    if cuts == False:
        df_total_lh5 = read_all_dsp_lh5(data_path,cuts,run=run)
        failed_cuts = np.zeros(len(df_total_lh5[energy_filter]))
    else:
        sigma_cuts = 4
        print("sigma cuts: ", str(sigma_cuts))
        df_total_lh5, failed_cuts = read_all_dsp_lh5(data_path,cuts,run=run, sigma=sigma_cuts)
        failed_cuts = failed_cuts[energy_filter]

    print("df_total_lh5: ", df_total_lh5)
    energy_filter_data = df_total_lh5[energy_filter]
    energy_filter_data=energy_filter_data.to_frame(name='energy_filter')
    energy_failed=failed_cuts.to_frame(name='failed_cuts')

    output_file=CodePath+"/data_calibration/"+detector+"/loaded_energy_"+detector+"_"+energy_filter+"_run"+str(run)+".hdf5
    energy_filter_data.to_hdf(output_file, key='energy', mode='w')
    energy_failed.to_hdf(output_file, key='failed')


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
        df_total_cuts, failed_cuts = cut.load_df_with_cuts(files, lh5_group, cut_file = cut_file_path, cut_parameters= {'bl_mean':sigma,'bl_std':sigma, 'pz_std':sigma}, verbose=True)

        return df_total_cuts, failed_cuts


if __name__ == "__main__":
    main()
