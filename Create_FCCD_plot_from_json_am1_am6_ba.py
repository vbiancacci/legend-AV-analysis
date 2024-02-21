import sys
import os
import json

import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys

#Script to compare FCCDs for each detector, for Ba_hs4, Am_HS1 with ICPC and BEGe corrections
#Creates a json file of results

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
    detector_list = CodePath+"/detector_list.json"
    with open(detector_list) as json_file:
        detector_list_data = json.load(json_file)
    order_list = [4,7,8,9]

    #Get FCCD results
    FCCD_results = CodePath+"/FCCD_parameters_covmatrix_am1_am6_ba.json"
    with open(FCCD_results) as json_file:
        FCCD_results_data = json.load(json_file)


    fig, ax = plt.subplots(figsize=(12,8))
    colors_orders = {4:'deepskyblue', 7:'green', 8:'gold', 9:'magenta'}
    markers_sources = {"ba_HS4": "o", "am_HS1":"^", "am_HS6":"s"} #abi plot

    detectors_all = []
    orders_all = []

    for order in order_list:

        detectors = detector_list_data["order_"+str(order)]

        FCCDs_ba = []
        FCCD_err_ups_ba = []
        FCCD_err_lows_ba = []

        FCCDs_am1_BEGe = []
        FCCD_err_ups_am1_BEGe = []
        FCCD_err_lows_am1_BEGe = []

        FCCDs_am6_BEGe = []
        FCCD_err_ups_am6_BEGe = []
        FCCD_err_lows_am6_BEGe = []


        detectors_ba = []
        detectors_am1_BEGe = []
        detectors_am6_BEGe = []

        for detector in detectors:

            detectors_all.append(detector)
            orders_all.append(order)

            #============Ba133=============
            try:
                FCCD_ba, FCCD_err_up_ba, FCCD_err_low_ba = FCCD_results_data[detector]["FCCD_ba"], FCCD_results_data[detector]["FCCD_ba_err_up"], FCCD_results_data[detector]["FCCD_ba_err_low"] 
                FCCDs_ba.append(FCCD_ba)
                FCCD_err_ups_ba.append(FCCD_err_up_ba)
                FCCD_err_lows_ba.append(FCCD_err_low_ba)
                detectors_ba.append(detector)
            except:
                print("no Ba133 analysis for ", detector)


            #===============Am241 HS1===============

            try:
                
                FCCD_am6_BEGe, FCCD_err_up_am6_BEGe, FCCD_err_low_am6_BEGe = FCCD_results_data[detector]["FCCD_am6"], FCCD_results_data[detector]["FCCD_am6_err_up"], FCCD_results_data[detector]["FCCD_am6_err_low"]
                FCCDs_am6_BEGe.append(FCCD_am6_BEGe)
                FCCD_err_ups_am6_BEGe.append(FCCD_err_up_am6_BEGe)
                FCCD_err_lows_am6_BEGe.append(FCCD_err_low_am6_BEGe)
                detectors_am6_BEGe.append(detector)
                print("Am241_HS6 BEGe analysis for ", detector)
            except:
                print("no Am241_HS6 analysis for ", detector)

            try:
                
                FCCD_am1_BEGe, FCCD_err_up_am1_BEGe, FCCD_err_low_am1_BEGe = FCCD_results_data[detector]["FCCD_am1"], FCCD_results_data[detector]["FCCD_am1_err_up"], FCCD_results_data[detector]["FCCD_am1_err_low"]
                FCCDs_am1_BEGe.append(FCCD_am1_BEGe)
                FCCD_err_ups_am1_BEGe.append(FCCD_err_up_am1_BEGe)
                FCCD_err_lows_am1_BEGe.append(FCCD_err_low_am1_BEGe)
                detectors_am1_BEGe.append(detector)
                print("Am241_HS1 BEGe analysis for ", detector)
            except:
                print("no Am241_HS1 analysis for ", detector)

            

        cc = colors_orders[order]

        #abi plot: _BEGe_corrections.png
        ax.errorbar(detectors_ba,FCCDs_ba, yerr = [FCCD_err_lows_ba, FCCD_err_ups_ba], marker = markers_sources["ba_HS4"], color=lighten_color(cc,1.05), linestyle = '-')
        ax.errorbar(detectors_am1_BEGe,FCCDs_am1_BEGe, yerr = [FCCD_err_lows_am1_BEGe, FCCD_err_ups_am1_BEGe], marker = markers_sources["am_HS1"], color=lighten_color(cc,1.2) ,linestyle = '-')
        ax.errorbar(detectors_am6_BEGe,FCCDs_am6_BEGe, yerr = [FCCD_err_lows_am6_BEGe, FCCD_err_ups_am6_BEGe], marker = markers_sources["am_HS6"], color=lighten_color(cc,0.6) ,linestyle = '-')


    for order in colors_orders:
        color = colors_orders[order]
        ax.plot(np.NaN, np.NaN, c=color, label=f'Order #'+str(order))

    ax2 = ax.twinx()
    for source in markers_sources:
        marker = markers_sources[source]
        ax2.plot(np.NaN, np.NaN, marker=marker,c='grey',label=source)
    ax2.get_yaxis().set_visible(False)

    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=13)
    ax2.legend(loc='upper left', bbox_to_anchor=(0.16, 1), fontsize=13)

    ax.tick_params(axis='x', labelrotation=45)
    #ax.set_xlabel('Detector', fontsize=13)
    ax.set_ylabel('FCCD [mm]', fontsize=20)
    ax.grid(linestyle='dashed', linewidth=0.5)
    ax.tick_params(axis="both", labelsize=17)
    plt.tight_layout()
    #ax.set_title("FCCDs from Ba-133 HS4 and Am-241 HS1", fontsize=14) #, Am-241 HS6")
    plt.savefig(CodePath+"/FCCDs_Ba133_Am241_ICPC_corr_am1_am6_ba.pdf", bbox_inches='tight') #abi plot
    #plt.show()



if __name__ == "__main__":
    main()
