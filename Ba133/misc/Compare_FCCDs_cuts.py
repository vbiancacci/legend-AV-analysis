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
    order_list = [9]

    fig, ax = plt.subplots(figsize=(12,8))
    colors_orders = {2:'darkviolet', 4:'deepskyblue', 5:'orangered', 7:'green', 8:'gold', 9:'magenta'}
    markers_sources = {"ba_HS4": "o", "am_HS1":"s", "am_HS6":"^"}
    linestyle_cuts = {"QC": "-", "No QC":"-."}

    detectors_all = []
    orders_all = []

    for order in order_list:

        detectors = detector_list_data["order_"+str(order)]

        FCCDs_ba = []
        FCCD_err_ups_ba = []
        FCCD_err_lows_ba = []

        FCCDs_ba_nocuts = []
        FCCD_err_ups_ba_nocuts = []
        FCCD_err_lows_ba_nocuts = []

        smear = "g"
        TL_model = "notl"
        frac_FCCDbore = 0.5
        energy_filter = "cuspEmax_ctc"

        for detector in detectors:

            if order == 7 or order == 8:
                run = 2
            else:
                run = 1
            
            detectors_all.append(detector)
            orders_all.append(order)

            #=====Cuts = True=========
            cuts = True

            if order == 8:
                source_z = "88z"
            elif order == 9:
                source_z = "74z"
            else:
                source_z = "78z"
            Ba133_FCCD_file = CodePath+"/../FCCD/FCCD_data_"+detector+"-ba_HS4-top-0r-"+source_z+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_cuts.json"
        
            try:
                with open(Ba133_FCCD_file) as json_file_ba:
                    FCCD_data_ba = json.load(json_file_ba)
                FCCD_ba, FCCD_err_up_ba, FCCD_err_low_ba = FCCD_data_ba["FCCD"], FCCD_data_ba["FCCD_err_up"], FCCD_data_ba["FCCD_err_low"]
                FCCDs_ba.append(FCCD_ba)
                FCCD_err_ups_ba.append(FCCD_err_up_ba)
                FCCD_err_lows_ba.append(FCCD_err_low_ba)
            except:
                print("no Ba133 analysis for ", detector)
            
            #=====Cuts = False=========
            cuts = False

            if order == 8:
                source_z = "88z"
            elif order == 9:
                source_z = "74z"
            else:
                source_z = "78z"
            Ba133_FCCD_file = CodePath+"/../FCCD/FCCD_data_"+detector+"-ba_HS4-top-0r-"+source_z+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+".json"
        
            try:
                with open(Ba133_FCCD_file) as json_file_ba:
                    FCCD_data_ba = json.load(json_file_ba)
                FCCD_ba, FCCD_err_up_ba, FCCD_err_low_ba = FCCD_data_ba["FCCD"], FCCD_data_ba["FCCD_err_up"], FCCD_data_ba["FCCD_err_low"]
                FCCDs_ba_nocuts.append(FCCD_ba)
                FCCD_err_ups_ba_nocuts.append(FCCD_err_up_ba)
                FCCD_err_lows_ba_nocuts.append(FCCD_err_low_ba)
            except:
                print("no Ba133 analysis for ", detector)

        print(detectors_all)
        print(FCCDs_ba)
        print(FCCDs_ba_nocuts)   

        cc = colors_orders[order]
        ax.errorbar(detectors_all,FCCDs_ba, yerr = [FCCD_err_lows_ba, FCCD_err_ups_ba], marker = markers_sources["ba_HS4"], color=cc, linestyle = linestyle_cuts["QC"])
        ax.errorbar(detectors_all,FCCDs_ba_nocuts, yerr = [FCCD_err_lows_ba_nocuts, FCCD_err_ups_ba_nocuts], marker = markers_sources["ba_HS4"], color=cc, linestyle = linestyle_cuts["No QC"])

    for order in colors_orders:
        color = colors_orders[order]
        ax.plot(np.NaN, np.NaN, c=color, label=f'Order #'+str(order))

    ax2 = ax.twinx()
    for source in markers_sources:
        marker = markers_sources[source]
        ax2.plot(np.NaN, np.NaN, marker=marker,c='grey',label=source)
    ax2.get_yaxis().set_visible(False)

    ax3 = ax.twinx()
    for cuts in linestyle_cuts:
        linestyle = linestyle_cuts[cuts]
        ax3.plot(np.NaN, np.NaN, linestyle=linestyle,c='grey',label=cuts)
    ax3.get_yaxis().set_visible(False)


    ax.legend(loc='upper left', bbox_to_anchor=(0.09, 1), prop={"size":9})
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 1), prop={"size":9})
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
