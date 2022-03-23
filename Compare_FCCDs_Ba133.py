import sys
import os
import json
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


#old
#Script to compare FCCDs for each detector - currently just for Ba133

CodePath=os.path.dirname(os.path.realpath(__file__))

def main():

    #Get detector list
    detector_list = CodePath+"/detector_list.json" 
    with open(detector_list) as json_file: 
        detector_list_data = json.load(json_file)
    order_list = [2,3,4,5,7]

    plt.rcParams['figure.figsize'] = (12, 8)
    plt.figure()

    detectors_all = []
    orders_all = []
    FCCDs_all =[]
    FCCD_err_ups_all = []
    FCCD_err_lows_all = []

    for order in order_list:

        detectors = detector_list_data["order_"+str(order)]

        FCCDs = []
        FCCD_err_ups = []
        FCCD_err_lows = []

        for detector in detectors:

            # if detector == "V04549B": #doesnt yet have an FCCD - calibration problems 
            #     FCCD, FCCD_err_up, FCCD_err_low = 0,0,0
            #     FCCDs.append(FCCD)
            #     FCCD_err_ups.append(FCCD_err_up)
            #     FCCD_err_lows.append(FCCD_err_low)
            #     continue

            #get best fit FCCD:
            smear = "g"
            TL_model = "notl"
            frac_FCCDbore = 0.5
            energy_filter = "cuspEmax_ctc"
            if order == 7 or order == 8:
                run = 2
            else:
                run = 1
            cuts = True

            #Just Ba133 for now
            Ba133_FCCD_file = CodePath+"/Ba133/FCCD/FCCD_data_"+detector+"-ba_HS4-top-0r-78z_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_cuts.json"
            with open(Ba133_FCCD_file) as json_file: 
                FCCD_data = json.load(json_file)
            
            FCCD, FCCD_err_up, FCCD_err_low = FCCD_data["FCCD"], FCCD_data["FCCD_err_up"], FCCD_data["FCCD_err_low"]
            FCCDs.append(FCCD)
            FCCD_err_ups.append(FCCD_err_up)
            FCCD_err_lows.append(FCCD_err_low)


            detectors_all.append(detector)
            orders_all.append(order)
            FCCDs_all.append(FCCD)
            FCCD_err_ups_all.append(FCCD_err_up)
            FCCD_err_lows_all.append(FCCD_err_low)

        
        plt.errorbar(detectors,FCCDs, yerr = [FCCD_err_lows, FCCD_err_ups], marker = 'o', linestyle = '-', label=f'Order #'+str(order))

    plt.xticks(rotation = 45)
    plt.xlabel('Detector')
    plt.ylabel('FCCD (mm)')
    plt.legend()
    plt.grid(linestyle='dashed', linewidth=0.5)
    plt.tight_layout()
    plt.title("FCCDs from Ba-133")
    plt.savefig(CodePath+"/FCCDs_Ba133.png", bbox_inches='tight')
    plt.show()


    #Save all values to csv file
    dict = {"detector": detectors_all, "detector_order": orders_all, "FCCD": FCCDs_all, "FCCD_err_up": FCCD_err_ups_all, "FCCD_err_low": FCCD_err_lows_all}
    df = pd.DataFrame(dict)
    print(df)
    df.to_csv(CodePath+"/FCCDs_Ba133.csv")
    


if __name__ == "__main__":
    main()