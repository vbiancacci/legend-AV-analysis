import numpy as np
import pandas as pd
import json

def main():

#    detector= "V07647A"
#    MC_id_am6 = "am_HS6-top-0r-198z"
#    MC_id_am1 = "am_HS1-top-46r-2z"
#    smear = "g"
#    TL_model ="notl"
#    frac_FCCDbore = "0.5"
#    energy_filter = "cuspEmax_ctc"
#    run = 1

#    O_am_list = []
#    O_am_err_list =[]
#    a_am_list = []
#    a_am_err_list = []

    ratio_new_list=[]
    ratio_new_err_list=[]

    detectors_list = ["V07647A", "V07647B", "V07298B", "V07302A", "V07302B"]

    #ParametersCal_am1 = "FCCD/FCCD_data_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+"_cuts.json"

    ParametersCal = "par_calibration.json"

    with open(ParametersCal) as json_file:
        Parameters = json.load(json_file)
        for detector in detectors_list:
            O_am1 = Parameters[detector]["am_HS1"]['O_Am241_data']
            O_am1_err = Parameters[detector]["am_HS1"]['O_Am241_data_err']
            a_am1 = Parameters[detector]["am_HS1"]['a']
            a_am1_err = Parameters[detector]["am_HS1"]['a_err']

            O_am6 = Parameters[detector]["am_HS6"]['O_Am241_data']
            O_am6_err = Parameters[detector]["am_HS6"]['O_Am241_data_err']
            a_am6 = Parameters[detector]["am_HS6"]['a']
            a_am6_err = Parameters[detector]["am_HS6"]['a_err']

            ratio = O_am6/O_am1
            ratio_err = StatisticalError(O_am6, O_am1, O_am6_err, O_am1_err)*ratio
            num_err = np.sqrt((ratio*a_am1_err)**2 + (a_am1*ratio_err)**2)
            ratio_new = a_am1 /(a_am6 /ratio)
            ratio_new_err = StatisticalError(a_am1*ratio, a_am6, num_err, a_am6_err)

            print("calibration_factor ", detector, " " , ratio_new, " +- ", ratio_new_err)
            ratio_new_list.append(ratio_new)
            ratio_new_err_list.append(ratio_new_err)

    mean_ratio=sum(ratio_new_list)/len(ratio_new_list)
    mean_ratio_err=np.sqrt(sum(r*r for r in ratio_new_err_list))/len(ratio_new_list)

    print ("mean ratio ", mean_ratio, " +- ", mean_ratio_err )

        #    O_am_list.append(O_am)
        #    O_am_err_list.append(O_am_err)
        #    a_am_list.append(a_am)
        #    a_am_err_list.append(a_am_err)

#    ParametersCal_am6 = "FCCD/FCCD_data_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+"_cuts.json"

#    with open(ParametersCal_am6) as json_file_am6:
#        Parameters_am6 = json.load(json_file_am6)
#        O_am6 = Parameters_am6['O_Am241_data']
#        O_am6_err = Parameters_am6['O_Am241_data_err']
#        a_am6 = Parameters_am6['a']
#        a_am6_err = Parameters_am6['a_err']

#    ratio = O_am6/O_am1
#    ratio_err = StatisticalError(O_am6, O_am1, O_am6_err, O_am1_err)*ratio
#    num_err = np.sqrt((ratio*a_am1_err)**2 + (a_am1*ratio_err)**2)
#    ratio_new = a_am1 /(a_am6 /ratio)
#    ratio_new_err = StatisticalError(a_am1*ratio, a_am6, num_err, a_am6_err)

#    print("calibration_factor ", ratio_new, " +- ", ratio_new_err)




def StatisticalError (A, B, errA, errB):
   se=(errA/A)**2+(errB/B)**2
   return np.sqrt(se)

if __name__ == "__main__":
    main()
