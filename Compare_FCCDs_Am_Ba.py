import sys
import os
import json
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys

#Script to compare FCCDs for each detector - currently just for Ba133

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
    order_list = [2,4,5,7,8]

    positions_list=CodePath+"/positions_am1_list.json"
    with open(positions_list) as json_file:
        positions_list_data=json.load(json_file)

    plt.rcParams['figure.figsize'] = (12, 8)
    plt.figure()

    detectors_all = []
    orders_all = []
    FCCDs_all =[]
    FCCD_err_ups_all = []
    FCCD_err_lows_all = []

    for order in order_list:

        detectors = detector_list_data["order_"+str(order)]

        FCCDs_ba = []
        FCCD_err_ups_ba = []
        FCCD_err_lows_ba = []

        FCCDs_am1 = []
        FCCD_err_ups_am1 = []
        FCCD_err_lows_am1 = []

        FCCDs_am6 = []
        FCCD_err_ups_am6 = []
        FCCD_err_lows_am6 = []

        detectors_ba = []
        detectors_am1 = []
        detectors_am6 = []

        smear = "g"
        TL_model = "notl"
        frac_FCCDbore = 0.5
        energy_filter = "cuspEmax_ctc"

        for detector in detectors:

            if order == 7 or order == 8:
                run = 2
            else:
                run = 1
            cuts = True
            detectors_all.append(detector)
            orders_all.append(order)

            #Ba133
            Ba133_FCCD_file = CodePath+"/Ba133/FCCD/FCCD_data_"+detector+"-ba_HS4-top-0r-78z_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_cuts.json"
            #print(Ba133_FCCD_file)
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


            #Am241 HS1
            if detector=='V02160A':
                run=3
            if detector=='V08682A':
                run=3
            elif detector=='V02166B' or detector =='V04545A':
                run=2
            elif detector=='V02162B':
                continue
            else:
                run=1
            position=positions_list_data[detector]
            am1_FCCD_file = CodePath+"/Am241/FCCD/am_HS1/weighted_mean/FCCD_data_"+detector+"-am_HS1-"+position+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_cuts.json"
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


            #Am241 HS6
            run = 1
            am6_FCCD_file = CodePath+"/Am241/FCCD/am_HS6/FCCD_data_"+detector+"-am_HS6-top-0r-198z_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_cuts.json"
            #print(am6_FCCD_file)
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


            #FCCDs_all.append(FCCD)
            #FCCD_err_ups_all.append(FCCD_err_up)
            #FCCD_err_lows_all.append(FCCD_err_low)
        cc=''
        if order==7:
            cc='green'
        elif order==8:
            cc='gold'
        elif order==5:
            cc='orangered'
        elif order==4:
            cc='deepskyblue'
        else:
            cc='darkviolet'
        plt.errorbar(detectors_ba,FCCDs_ba, yerr = [FCCD_err_lows_ba, FCCD_err_ups_ba], marker = 'o', color=cc, linestyle = '-', label=f'Order #'+str(order)+' Ba')
        plt.errorbar(detectors_am1,FCCDs_am1, yerr = [FCCD_err_lows_am1, FCCD_err_ups_am1], marker = 's', color=lighten_color(cc,1.3) ,linestyle = '-', label=f'Order #'+str(order)+' Am HS1')
        if order==7:
            plt.errorbar(detectors_am6,FCCDs_am6, yerr = [FCCD_err_lows_am6, FCCD_err_ups_am6], marker = '^', color=lighten_color(cc,0.3), linestyle = '-', label=f'Order #'+str(order)+' Am HS6')

    plt.xticks(rotation = 45)
    plt.xlabel('Detector')
    plt.ylabel('FCCD (mm)')
    plt.legend()
    plt.grid(linestyle='dashed', linewidth=0.5)
    plt.tight_layout()
    plt.title("FCCDs from Ba-133, Am-241 HS1, Am-241 HS6")
    plt.savefig(CodePath+"/FCCDs_Ba133_Am241_new_correction.png", bbox_inches='tight')
    plt.show()


    #Save all values to csv file
    #dict = {"detector": detectors_all, "detector_order": orders_all, "FCCD": FCCDs_all, "FCCD_err_up": FCCD_err_ups_all, "FCCD_err_low": FCCD_err_lows_all}
    #df = pd.DataFrame(dict)
    #print(df)
    #df.to_csv(CodePath+"/FCCDs_Ba133.csv")



if __name__ == "__main__":
    main()
