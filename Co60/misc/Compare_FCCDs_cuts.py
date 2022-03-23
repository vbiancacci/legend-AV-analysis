import sys
import os
import json
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys

#Script to compare FCCDs for each detector

CodePath=os.path.dirname(os.path.realpath(__file__))

def lighten_color(color, amount = 0.5):
    try:
        c = mc.cnames[color]
    except:
       c = color
    c = colorsys.rgb_to_hls( * mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def main():

    #Get detector list
    detector_list = CodePath+"/../../detector_list.json"
    with open(detector_list) as json_file:
        detector_list_data = json.load(json_file)

    
    source = "co_HS5"

    fig, ax = plt.subplots(figsize=(12,8))
    linestyle_cuts = {"QC": "-", "No QC":"-."}

    order = "BEGe"
    detectors = detector_list_data["order_"+str(order)]

    FCCDs = []
    FCCD_err_ups = []
    FCCD_err_lows = []

    FCCDs_nocuts = []
    FCCD_err_ups_nocuts = []
    FCCD_err_lows_nocuts = []

    smear = "g"
    TL_model = "notl"
    frac_FCCDbore = 0.5
    energy_filter = "cuspEmax_ctc"
    run="1"
    source_z = "198z"

    for detector in detectors:

        
        #=====Cuts = True=========
        # cuts = True

        FCCD_file = CodePath+"/../FCCD/FCCD_data_"+detector+"-"+source+"-top-0r-"+source_z+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_cuts.json"
    
        try:
            with open(FCCD_file) as json_file:
                FCCD_data = json.load(json_file)
            FCCD, FCCD_err_up, FCCD_err_low = FCCD_data["FCCD_av"], FCCD_data["FCCD_av_err_up"], FCCD_data["FCCD_av_err_low"]
            FCCDs.append(FCCD)
            FCCD_err_ups.append(FCCD_err_up)
            FCCD_err_lows.append(FCCD_err_low)
        except:

            print("no CUTS analysis for ", detector)
        
        #=====Cuts = False=========
        # cuts = False

        FCCD_file = CodePath+"/../FCCD/FCCD_data_"+detector+"-"+source+"-top-0r-"+source_z+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+".json"
    
        try:
            with open(FCCD_file) as json_file:
                FCCD_data = json.load(json_file)
            FCCD, FCCD_err_up, FCCD_err_low = FCCD_data["FCCD_av"], FCCD_data["FCCD_av_err_up"], FCCD_data["FCCD_av_err_low"]
            FCCDs_nocuts.append(FCCD)
            FCCD_err_ups_nocuts.append(FCCD_err_up)
            FCCD_err_lows_nocuts.append(FCCD_err_low)
        except:
            print("no analysis for ", detector)

    # print(detectors)
    # print(FCCDs)
    # print(FCCDs_nocuts) 
    # print(FCCD_err_ups)  
    # print(FCCD_err_ups_nocuts)
    # print(FCCD_err_lows)

    ax.errorbar(detectors,FCCDs, yerr = [FCCD_err_lows, FCCD_err_ups],linestyle = linestyle_cuts["QC"])
    ax.errorbar(detectors,FCCDs_nocuts, yerr = [FCCD_err_lows_nocuts, FCCD_err_ups_nocuts], linestyle = linestyle_cuts["No QC"])

 
    ax3 = ax.twinx()
    for cuts in linestyle_cuts:
        linestyle = linestyle_cuts[cuts]
        ax3.plot(np.NaN, np.NaN, linestyle=linestyle,c='grey',label=cuts)
    ax3.get_yaxis().set_visible(False)


    ax.legend(loc='upper left', bbox_to_anchor=(0.09, 1), prop={"size":9})
    ax3.legend(loc='upper right', prop={"size":11})


    ax.tick_params(axis='x', labelrotation=45)
    ax.set_xlabel('Detector')
    ax.set_ylabel('FCCD (mm)')
    ax.grid(linestyle='dashed', linewidth=0.5)
    plt.tight_layout()
    # ax.set_title("FCC")
    plt.show()

if __name__ == "__main__":
    main()
