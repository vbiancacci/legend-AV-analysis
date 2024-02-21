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
    detector_list = CodePath+"/detector_list.json"
    with open(detector_list) as json_file:
        detector_list_data = json.load(json_file)
    order_list = [2,4,5,7,8,9]

    positions_list=CodePath+"/positions_am1_list.json"
    with open(positions_list) as json_file:
        positions_list_data=json.load(json_file)

    fig, ax = plt.subplots(figsize=(12,8))
    colors_orders = {2:'darkviolet', 4:'deepskyblue', 5:'orangered', 7:'green', 8:'gold', 9:'magenta'}
    markers_sources = { "ICPC correction":"s","BEGe correction":"p", "am_HS6":"^"} #"ba_HS4": "o",

    detectors_all = []
    orders_all = []

    smear = "g"
    TL_model = "notl"
    frac_FCCDbore = 0.5
    energy_filter = "cuspEmax_ctc"
    cuts = True

    for order in order_list:

        detectors = detector_list_data["order_"+str(order)]

        FCCDs_ba = []
        FCCD_err_ups_ba = []
        FCCD_err_lows_ba = []

        FCCDs_am1 = []
        FCCD_err_ups_am1 = []
        FCCD_err_lows_am1 = []

        FCCDs_am1_2 = []
        FCCD_err_ups_am1_2 = []
        FCCD_err_lows_am1_2 = []

        FCCDs_am6 = []
        FCCD_err_ups_am6 = []
        FCCD_err_lows_am6 = []

        detectors_ba = []
        detectors_am1 = []
        detectors_am6 = []

        for detector in detectors:

            detectors_all.append(detector)
            orders_all.append(order)

            #============Ba133=================
            if order == 7 or order == 8:
                run = 2
            else:
                run = 1

            if order == 8:
                source_z = "88z"
            elif order == 9:
                source_z = "74z"
            else:
                source_z = "78z"
            Ba133_FCCD_file = CodePath+"/Ba133/FCCD/FCCD_data_"+detector+"-ba_HS4-top-0r-"+source_z+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_cuts.json"

            try:
                with open(Ba133_FCCD_file) as json_file_ba:
                    FCCD_data_ba = json.load(json_file_ba)
                FCCD_ba, FCCD_err_up_ba, FCCD_err_low_ba = FCCD_data_ba["FCCD"], FCCD_data_ba["FCCD_err_up"], FCCD_data_ba["FCCD_err_low"]
                FCCDs_ba.append(FCCD_ba)
                FCCD_err_ups_ba.append(FCCD_err_up_ba)
                FCCD_err_lows_ba.append(FCCD_err_low_ba)
                detectors_ba.append(detector)
            except:
                print("no Ba133 analysis for ", detector)


            #===========Am241 HS1==============
            if detector=='V08682A':
                run=3
            elif detector=='V09372A':
                run=4
            elif detector=='V02166B' or detector =='V04545A'or detector=='V09374A':
                run=2
            elif detector=='V02160A' or detector=='V02162B':
                continue
            else:
                run=1

            position=positions_list_data[detector]
            am1_FCCD_file = CodePath+"/Am241/FCCD/am_HS1/ICPC_correction/FCCD_data_"+detector+"-am_HS1-"+position+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_cuts.json"
            try:
                with open(am1_FCCD_file) as json_file_am1:
                    FCCD_data_am1 = json.load(json_file_am1)
                FCCD_am1, FCCD_err_up_am1, FCCD_err_low_am1 = FCCD_data_am1["FCCD"], FCCD_data_am1["FCCD_err_up"], FCCD_data_am1["FCCD_err_low"]
                FCCDs_am1.append(FCCD_am1)
                FCCD_err_ups_am1.append(FCCD_err_up_am1)
                FCCD_err_lows_am1.append(FCCD_err_low_am1)
                detectors_am1.append(detector)
            except:
                print("no Am241_HS1 analysis for ", detector)

            am1_FCCD_file_2 = CodePath+"/Am241/FCCD/am_HS1/BEGe_correction/FCCD_data_"+detector+"-am_HS1-"+position+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_cuts.json"
            try:
                with open(am1_FCCD_file_2) as json_file_am1_2:
                    FCCD_data_am1_2 = json.load(json_file_am1_2)
                FCCD_am1_2, FCCD_err_up_am1_2, FCCD_err_low_am1_2 = FCCD_data_am1_2["FCCD"], FCCD_data_am1_2["FCCD_err_up"], FCCD_data_am1_2["FCCD_err_low"]
                FCCDs_am1_2.append(FCCD_am1_2)
                FCCD_err_ups_am1_2.append(FCCD_err_up_am1_2)
                FCCD_err_lows_am1_2.append(FCCD_err_low_am1_2)
            except:
                print("no Am241_HS1 analysis for ", detector)

            #================Am241 HS6==============
            run = 1
            am6_FCCD_file = CodePath+"/Am241/FCCD/am_HS6/FCCD_data_"+detector+"-am_HS6-top-0r-198z_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_cuts.json"

            try:
                with open(am6_FCCD_file) as json_file_am6:
                    FCCD_data_am6 = json.load(json_file_am6)
                FCCD_am6, FCCD_err_up_am6, FCCD_err_low_am6 = FCCD_data_am6["FCCD"], FCCD_data_am6["FCCD_err_up"], FCCD_data_am6["FCCD_err_low"]
                FCCDs_am6.append(FCCD_am6)
                FCCD_err_ups_am6.append(FCCD_err_up_am6)
                FCCD_err_lows_am6.append(FCCD_err_low_am6)
                detectors_am6.append(detector)
            except:
                print("no Am241_HS6 analysis for ", detector)

        cc = colors_orders[order]
        #ax.errorbar(detectors_ba,FCCDs_ba, yerr = [FCCD_err_lows_ba, FCCD_err_ups_ba], marker = markers_sources["ba_HS4"], color=cc, linestyle = '-')
        ax.errorbar(detectors_am1,FCCDs_am1_2, yerr = [FCCD_err_lows_am1_2, FCCD_err_ups_am1_2], marker = markers_sources["ICPC correction"], color=lighten_color(cc,0.8) ,linestyle = '-')
        ax.errorbar(detectors_am1,FCCDs_am1, yerr = [FCCD_err_lows_am1, FCCD_err_ups_am1], marker = markers_sources["BEGe correction"], color=lighten_color(cc,1.2) ,linestyle = '-')
        if order==7:
            ax.errorbar(detectors_am6,FCCDs_am6, yerr = [FCCD_err_lows_am6, FCCD_err_ups_am6], marker = markers_sources["am_HS6"], color=lighten_color(cc,0.3), linestyle = '-')

    for order in colors_orders:
        color = colors_orders[order]
        ax.plot(np.NaN, np.NaN, c=color, label=f'Order #'+str(order))

    ax2 = ax.twinx()
    for source in markers_sources:
        marker = markers_sources[source]
        ax2.plot(np.NaN, np.NaN, marker=marker,c='grey',label=source)
    ax2.get_yaxis().set_visible(False)

    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 15})
    ax2.legend(loc='upper left', bbox_to_anchor=(0.2, 1), prop={'size': 15})

    ax.tick_params(axis='x', labelsize= 15, labelrotation=45)
    #ax.set_xlabel('Detector', fontsize=10)
    ax.set_ylabel('FCCD [mm]',fontsize=20)
    ax.grid(linestyle='dashed', linewidth=0.5)
    plt.tight_layout()
    #ax.set_title("FCCDs from Ba-133, Am-241 HS1, Am-241 HS6")
    #plt.savefig(CodePath+"/FCCDs_Ba133_Am241_ICPC_correction_both.png", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
