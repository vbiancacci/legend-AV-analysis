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
from scipy.special import erf, erfc

from pygama.analysis import histograms
from pygama.analysis import peak_fitting

import pygama.genpar_tmp.cuts as cut


dir=os.path.dirname(os.path.realpath(__file__))

def main():

    cuts=False
    source="am_HS6"
    energy_filter="cuspEmax_ctc"

    order_list=[0,1]

    detector_list = dir+"/../detector_list.json"
    with open(detector_list) as json_file:
        detector_list_data = json.load(json_file)

    fig, ax = plt.subplots(figsize=(10,8))
    for order in order_list:
        detectors = detector_list_data["order_"+str(order)]
        for detector in detectors:
            #if detector in ["B00035A","B00032B"]:
            #    continue
            if detector=="B00076C":
                run=2
            else:
                run=1

            if cuts==True:
                df=pd.read_hdf(dir+"/data_calibration/"+detector+"/"+source+"/loaded_energy_"+detector+"_"+energy_filter+"_run"+str(run)+".hdf5", key='energy')
            else:
                df=pd.read_hdf(dir+"/data_calibration/"+detector+"/"+source+"/loaded_energy_"+detector+"_"+energy_filter+"_nocuts_run"+str(run)+".hdf5", key='energy')

            energies=df['energy_filter']
            binwidth = 0.5 #keV
            bins = np.arange(0,140,binwidth)
            hist, bins, var = histograms.get_hist(energies, bins=bins)
            if detector in ["B00061C", "B00035B", "B00035A", "B00076C"]:
                weights=hist*21600/7200
            elif detector=="B00061C":
                weights=hist*21600/14400
            else:
                weights=hist
            histograms.plot_hist(weights, bins, label=detector,  var=None, show_stats=False, stats_hloc=0.75, stats_vloc=0.85, linewidth=0.5)

    plt.yscale("log")
    plt.xlabel("Energy [keV]")
    plt.ylabel("Counts / 0.5keV")
    plt.legend(loc="lower left", prop={'size': 8.5})
    #plt.show()
    plt.savefig(dir+"/data_calibration"+"/allBEGespectra_nocuts.png")



if __name__ =="__main__":
    main()
