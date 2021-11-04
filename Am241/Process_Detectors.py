import sys
import os
import json
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Master script to automatically launch other scripts in order to automatically process multiple detectors at once

CodePath=os.path.dirname(os.path.realpath(__file__))

def main():

    #Processing instructions
    order_list = [7] #List of orders to process
    Load_Data = False #Pre-reqs: needs dsp pygama data
    Calibrate_Data = False #Pre-reqs: needs load data
    Gamma_line_count_data = False #Pre-reqs: needs calibration
    Gamma_line_count_MC = False #Pre-reqs: needs AV post processed MC for range of FCCDs
    Calculate_FCCD = False #Pre-reqs: needs gammaline counts for data and MC
    Gamma_line_count_MC_bestfitFCCD = True #Pre-reqs: needs AV postprocessed MC for best fit FCCD
    PlotSpectra = True #Pre-reqs: needs all above stages

    #Get detector list
    detector_list = CodePath+"/../../detector_list.json"
    with open(detector_list) as json_file:
        detector_list_data = json.load(json_file)

    for order in order_list:
        detectors = detector_list_data["order_"+str(order)]
        for detector in detectors:

            if detector != "V07647A":
                continue

            energy_filter="cuspEmax_ctc"
            cuts="True"
            run=1

            smear="g"
            TL_model="notl"
            frac_FCCDbore=0.5

            #========Load - DATA==========
            if Load_Data == True:

                if order == 2:
                    detector_oldname = "I"+detector[1:]
                    data_path = "/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/gen/"+detector_oldname+"/tier2/am_HS6_top_dlt/"
                else:
                    data_path="/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/gen/"+detector+"/tier2/am_HS6_top_dlt/"


                os.system("python3 "+CodePath+"/Load_Data.py "+detector+" "+data_path+" "+energy_filter+" "+cuts+" "+str(run))


            #========Calibration - DATA==========
            if Calibrate_Data == True:


                os.system("python3 "+CodePath+"/Calibration_Am241.py "+detector+" "+energy_filter+" "+cuts+" "+str(run))


            #========GAMMA LINE COUNTING - DATA==========
            if Gamma_line_count_data == True:

                if cuts == "False":
                    calibration = CodePath+"/../../Ba133/data_calibration/"+detector+"/calibration_run"+str(run)+".json"
                else:
                    calibration = CodePath+"/../../Ba133/data_calibration/"+detector+"/calibration_run2_cuts.json"


                os.system("python3 "+CodePath+"/GammaLine_Counting_Am241.py --data "+detector+" "+calibration+" "+energy_filter+" "+cuts+" "+str(run))

            #=========GAMMA LINE COUNTING - MC=============
            if Gamma_line_count_MC == True:

                DLF_list=[1.0]
                FCCD_list=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

                for FCCD in FCCD_list:

                    for DLF in DLF_list:
                        MC_id=detector+"-am_HS6-top-0r-198z_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_fracFCCDbore"+str(frac_FCCDbore)
                        sim_path="/lfs/l1/legend/detector_char/enr/hades/simulations/legend-g4simple-simulation/simulations/"+detector+"/am_HS6/top_0r_198z/hdf5/AV_processed/"+MC_id+".hdf5"
                        os.system("python3 "+CodePath+"/GammaLine_Counting_Am241.py --sim "+detector+" "+sim_path+" "+MC_id)

            #=============Calculate FCCD===============
            if Calculate_FCCD == True:

                MC_id=detector+"-am_HS6-top-0r-198z"

                os.system("python3 "+CodePath+"/Calculate_FCCD.py "+detector+" "+MC_id+" "+smear+" "+TL_model+" "+str(frac_FCCDbore)+" "+energy_filter+" "+cuts+" "+str(run))

            #=========GAMMA LINE COUNTING - MC, best fit FCCD=============
            if Gamma_line_count_MC_bestfitFCCD == True:

                DLF=1.0

                if cuts == "False":
                    with open(CodePath+"/FCCD/FCCD_data_"+detector+"-am_HS6-top-0r-198z_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+".json") as json_file:
                        FCCD_data = json.load(json_file)
                else:
                    with open(CodePath+"/FCCD/FCCD_data_"+detector+"-am_HS6-top-0r-198z_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_cuts.json") as json_file:
                        FCCD_data = json.load(json_file)
                FCCD = round(FCCD_data["FCCD"],2)

                MC_id=detector+"-am_HS6-top-0r-198z_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_fracFCCDbore"+str(frac_FCCDbore)
                sim_path="/lfs/l1/legend/detector_char/enr/hades/simulations/legend-g4simple-simulation/simulations/"+detector+"/am_HS6/top_0r_198z/hdf5/AV_processed/"+MC_id+".hdf5"
                os.system("python3 "+CodePath+"/GammaLine_Counting_Am241.py --sim "+detector+" "+sim_path+" "+MC_id)


            #=============Plot Spectra===============
            if PlotSpectra == True:

                if cuts == "False":
                    calibration = CodePath+"/../../Ba133/data_calibration/"+detector+"/calibration_run"+str(run)+".json"
                else:
                    calibration = CodePath+"/../../Ba133/data_calibration/"+detector+"/calibration_run2_cuts.json"

                DLF=1.0

                if cuts == "False":
                    with open(CodePath+"/FCCD/FCCD_data_"+detector+"-am_HS6-top-0r-198z_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+".json") as json_file:
                        FCCD_data = json.load(json_file)
                else:
                    with open(CodePath+"/FCCD/FCCD_data_"+detector+"-am_HS6-top-0r-198z_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_cuts.json") as json_file:
                        FCCD_data = json.load(json_file)
                FCCD = round(FCCD_data["FCCD"],2)

                MC_id=detector+"-am_HS6-top-0r-198z_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_fracFCCDbore"+str(frac_FCCDbore)
                sim_path="/lfs/l1/legend/detector_char/enr/hades/simulations/legend-g4simple-simulation/simulations/"+detector+"/am_HS6/top_0r_198z/hdf5/AV_processed/"+MC_id+".hdf5"

                os.system("python3 "+CodePath+"/PlotSpectra.py "+detector+" "+MC_id+" "+sim_path+" "+str(FCCD)+" "+str(DLF)+" "+calibration+" "+energy_filter+" "+cuts+" "+str(run))


if __name__ == "__main__":
    main()
