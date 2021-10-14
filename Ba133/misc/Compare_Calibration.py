import sys
import os
import json
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

CodePath=os.path.dirname(os.path.realpath(__file__))

def main():

    order_list = [2,3,4,5]
    energy_filter="cuspEmax_ctc"

    #Get detector list
    detector_list = CodePath+"/../../detector_list.json" 
    with open(detector_list) as json_file: 
        detector_list_data = json.load(json_file)

    m_Th_list =[]
    m_Ba_list =[]
    c_Th_list =[]
    c_Ba_list =[]
    a_Th_list =[]
    a_Ba_list =[]
    b_Th_list =[]
    b_Ba_list =[]

    detectors_all = []

    for order in order_list:
        detectors = detector_list_data["order_"+str(order)]
        for detector in detectors:

            detectors_all.append(detector)

            if order == 7 or order==8:
                run=2
            else:
                run=1

            if order == 2:
                detector_oldname = "I"+detector[1:]
                calibration_Th="/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/genpar/dsp_ecal/"+detector_oldname+".json"
            else:
                calibration_Th="/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/genpar/dsp_ecal/"+detector+".json"
         
            calibration_Ba = CodePath+"/../data_calibration/"+detector+"/calibration_run"+str(run)+"_cuts.json"

            with open(calibration_Ba) as json_file:
                calibration_coefs = json.load(json_file)
            m_Ba = calibration_coefs[energy_filter]["calibration"][0]
            c_Ba = calibration_coefs[energy_filter]["calibration"][1]
            a_Ba = calibration_coefs[energy_filter]["resolution"][0]
            b_Ba = calibration_coefs[energy_filter]["resolution"][1]

            with open(calibration_Th) as json_file:
                calibration_coefs = json.load(json_file)
            # print(calibration_coefs)
            m_Th = calibration_coefs[energy_filter]["Calibration_pars"][0]
            c_Th = calibration_coefs[energy_filter]["Calibration_pars"][1]
            a_Th = calibration_coefs[energy_filter]["m0"]
            b_Th = calibration_coefs[energy_filter]["m1"]
            
            m_Ba_list.append(m_Ba)
            m_Th_list.append(m_Th)
            c_Ba_list.append(c_Ba)
            c_Th_list.append(c_Th)
            a_Ba_list.append(a_Ba)
            a_Th_list.append(a_Th)
            b_Ba_list.append(b_Ba)
            b_Th_list.append(b_Th)
    

    plt.rcParams['figure.figsize'] = (12, 8)

    fig, axs = plt.subplots(2, 2, sharex=True)
    axs[0, 0].scatter(detectors_all,m_Ba_list, label=f'Ba-133')
    axs[0, 0].scatter(detectors_all,m_Th_list, label=f'Th-133')
    axs[0, 0].set_ylabel("m")
    axs[0, 0].legend()
    axs[1, 0].scatter(detectors_all,c_Ba_list, label=f'Ba-133')
    axs[1, 0].scatter(detectors_all,c_Th_list, label=f'Th-133')
    axs[1, 0].hlines(0, detectors_all[0], detectors_all[-1], colors = "grey", linestyles='dashed')
    axs[1, 0].set_ylabel("c")
    axs[1, 0].legend()
    axs[0, 1].scatter(detectors_all,a_Ba_list, label=f'Ba-133')
    axs[0, 1].scatter(detectors_all,a_Th_list, label=f'Th-133')
    axs[0, 1].set_ylabel("a")
    axs[0, 1].legend()
    axs[1, 1].scatter(detectors_all,b_Ba_list, label=f'Ba-133')
    axs[1, 1].scatter(detectors_all,b_Th_list, label=f'Th-133')
    axs[1, 1].set_ylabel("b")
    axs[1, 1].legend()

    fig.tight_layout()
    axs[1,0].set_xticks(range(-1,len(detectors_all)-1,1))
    axs[1,0].set_xticklabels(detectors_all,rotation=45)
    axs[1,0].set_xlabel('Detector')
    axs[1,1].set_xticks(range(-1,len(detectors_all)-1,1))
    axs[1,1].set_xticklabels(detectors_all,rotation=45)
    axs[1,1].set_xlabel('Detector')
    axs[0, 0].set_title("Calibration Coefficients")
    axs[0, 1].set_title("Resolution Coefficients")
    fig.tight_layout()
    plt.savefig("compare_calibration.png")
    # fig.suptitle("Calibration Coefficients")


    #compute percentage difference for resolution coefficients
    a_dif_pct = ((np.array(a_Ba_list)-np.array(a_Th_list))/np.array(a_Ba_list))*100
    b_dif_pct = ((np.array(b_Ba_list)-np.array(b_Th_list))/np.array(b_Ba_list))*100
    plt.figure()
    plt.scatter(detectors_all,a_dif_pct, label=f'x = a')
    plt.scatter(detectors_all,b_dif_pct, label=f'x = b')
    plt.hlines(0, detectors_all[0], detectors_all[-1], colors = "grey", linestyles='dashed')
    plt.xticks(rotation = 45)
    plt.xlabel('Detector')
    plt.ylabel(r'$(x_{Ba}-x_{Th})/x_{Ba}  \%$')
    plt.legend()
    plt.tight_layout()
    plt.title(r"$\%$ difference of resolution coefficients")
    plt.savefig("compare_resolution_coef_pct.png")

    # plt.figure()
    # plt.scatter(detectors_all,b_Ba_list, label=f'Ba-133')
    # plt.scatter(detectors_all,b_Th_list, label=f'Th-228')
    # plt.xticks(rotation = 45)
    # plt.xlabel('Detector')
    # plt.ylabel('b')
    # plt.legend()
    # plt.tight_layout()
    # plt.title("")






    # plt.figure()
    # plt.scatter(detectors_all,m_Ba_list, label=f'Ba-133')
    # plt.scatter(detectors_all,m_Th_list, label=f'Th-228')
    # plt.xticks(rotation = 45)
    # plt.xlabel('Detector')
    # plt.ylabel('m')
    # plt.legend()
    # plt.tight_layout()
    # plt.title("")

    # plt.figure()
    # plt.scatter(detectors_all,c_Ba_list, label=f'Ba-133')
    # plt.scatter(detectors_all,c_Th_list, label=f'Th-228')
    # plt.xticks(rotation = 45)
    # plt.xlabel('Detector')
    # plt.ylabel('c')
    # plt.hlines(0, detectors_all[0], detectors_all[-1])
    # plt.legend()
    # plt.tight_layout()
    # plt.title("")

    # plt.figure()
    # plt.scatter(detectors_all,a_Ba_list, label=f'Ba-133')
    # plt.scatter(detectors_all,a_Th_list, label=f'Th-228')
    # plt.xticks(rotation = 45)
    # plt.xlabel('Detector')
    # plt.ylabel('a')
    # plt.legend()
    # plt.tight_layout()
    # plt.title("")

    # plt.figure()
    # plt.scatter(detectors_all,b_Ba_list, label=f'Ba-133')
    # plt.scatter(detectors_all,b_Th_list, label=f'Th-228')
    # plt.xticks(rotation = 45)
    # plt.xlabel('Detector')
    # plt.ylabel('b')
    # plt.legend()
    # plt.tight_layout()
    # plt.title("")






    plt.show()



if __name__ == "__main__":
    main()