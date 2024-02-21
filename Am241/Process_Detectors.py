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
    order_list = [1] #[0, 4, 5, 7, 8, 9] #List of orders to process
    Calibrate_Data = False #Pre-reqs: needs load data
    Gamma_line_count_data = False  #Pre-reqs: needs calibration
    Gamma_line_count_MC = False #Pre-reqs: needs AV post processed MC for range of FCCDs
    Calculate_FCCD = True #Pre-reqs: needs gammaline counts for data and MC
    Gamma_line_count_MC_bestfitFCCD = False #Pre-reqs: needs AV postprocessed MC for best fit FCCD
    PlotSpectra = False #Pre-reqs: needs all above stages

    #Get detector list
    detector_list = CodePath+"/../detector_list.json"
    with open(detector_list) as json_file:
        detector_list_data = json.load(json_file)

    positions_list=CodePath+"/../positions_am1_list.json"
    with open(positions_list) as json_file:
        positions_list_data=json.load(json_file)

    for order in order_list:
        detectors = detector_list_data["order_"+str(order)]
        for detector in detectors:
            if detector != "B00035A":
                 continue

            energy_filter="cuspEmax_ctc"
            cuts="False"

            #if detector=='V09372A':
            #    run=4
            #elif detector=='V02160A'or detector=='V08682A': # or detector=="B00000B":# or detector =='V04545A': #do them again! bad run 3
            #    run=3
            #elif detector=='V02166B' or detector=='V04545A'or detector=='V02162B' or detector=='V09374A' or detector=='B00076C':
            #    run=2
           # else:
            run=1


            smear="g"
            TL_model="notl"
            frac_FCCDbore=0.5
            source = "am_HS6"
            position = "top-0r-198z"
            #position=positions_list_data[detector]
            #========Calibration - DATA==========
            if Calibrate_Data == True:

                if order == 2:
                    detector_oldname = "I"+detector[1:]
                    data_path = "/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/gen/"+detector_oldname+"/tier2/"+source+"_top_dlt/"
                else:
                    data_path="/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/gen/"+detector+"/tier2/"+source+"_top_dlt/"
                    #data_path = "/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-full_dl-v01/gen/"+detector+"/tier2/"+source+"_top_dlt/"


                os.system("python3 "+CodePath+"/Calibration_Am241.py "+detector+" "+data_path+" "+energy_filter+" "+cuts+" "+str(run)+" "+source)


            #========GAMMA LINE COUNTING - DATA==========
            if Gamma_line_count_data == True:

                os.system("python3 "+CodePath+"/GammaLine_Counting_Am241_HS1_data.py --data "+detector+" "+energy_filter+" "+cuts+" "+str(run)+" "+source)

            #=========GAMMA LINE COUNTING - MC=============
            if Gamma_line_count_MC == True:

                DLF_list=[1.0]
                FCCD_list=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
                #position="top-0r-198z"
                #position="top-45r-4z"
                for FCCD in FCCD_list:

                    for DLF in DLF_list:
                        MC_id=detector+"-"+source+"-"+position+"_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_fracFCCDbore"+str(frac_FCCDbore)
                        sim_path="/lfs/l1/legend/detector_char/enr/hades/simulations/legend-g4simple-simulation/simulations/"+detector+"/"+source+"/"+position.replace("-","_")+"/hdf5/AV_processed/"+MC_id+".hdf5"
                        os.system("python3 "+CodePath+"/GammaLine_Counting_Am241_HS6.py --sim "+detector+" "+sim_path+" "+MC_id+" "+source)

            #=============Calculate FCCD===============
            if Calculate_FCCD == True:
                #position="top-0r-198z"
                MC_id=detector+"-"+source+"-"+position

                os.system("python3 "+CodePath+"/Calculate_FCCD.py "+detector+" "+MC_id+" "+smear+" "+TL_model+" "+str(frac_FCCDbore)+" "+energy_filter+" "+cuts+" "+str(run)+" "+source)

            #=========GAMMA LINE COUNTING - MC, best fit FCCD=============
            if Gamma_line_count_MC_bestfitFCCD == True:

                DLF=1.0

                if cuts == "False":
                    with open(CodePath+"/FCCD/"+source+"/FCCD_data_"+detector+"-"+source+"-"+position+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+".json") as json_file:
                        FCCD_data = json.load(json_file)
                else:
                    with open(CodePath+"/FCCD/"+source+"/FCCD_data_"+detector+"-"+source+"-"+position+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_cuts.json") as json_file:
                        FCCD_data = json.load(json_file)
                FCCD = round(FCCD_data["FCCD"],2)
                FCCD = 1.55
                MC_id=detector+"-"+source+"-"+position.replace("_","-")+"_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_fracFCCDbore"+str(frac_FCCDbore)
                sim_path="/lfs/l1/legend/detector_char/enr/hades/simulations/legend-g4simple-simulation/simulations/"+detector+"/"+source+"/"+position.replace("-","_")+"/hdf5/AV_processed/"+MC_id+".hdf5"
                os.system("python3 "+CodePath+"/GammaLine_Counting_Am241_HS6.py --sim "+detector+" "+sim_path+" "+MC_id+" "+source)


            #=============Plot Spectra===============
            if PlotSpectra == True:

                DLF=1.0

                #if cuts == "False":
                #    with open(CodePath+"/FCCD/FCCD_data_"+detector+"/"+source+"-"+position+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+".json") as json_file:
                #        FCCD_data = json.load(json_file)
                #else:
                #    with open(CodePath+"/FCCD/FCCD_data_"+detector+"/"+source+"-"+position+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_cuts.json") as json_file:
                #        FCCD_data = json.load(json_file)
                #FCCD = round(FCCD_data["FCCD"],2)
                FCCD=1.55
                #position="top_0r_198z"
                MC_id=detector+"-"+source+"-"+position.replace("_","-")+"_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_fracFCCDbore"+str(frac_FCCDbore)
                sim_path="/lfs/l1/legend/detector_char/enr/hades/simulations/legend-g4simple-simulation/simulations/"+detector+"/"+source+"/"+position.replace("-","_")+"/hdf5/AV_processed/"+MC_id+".hdf5"

                os.system("python3 "+CodePath+"/PlotSpectra.py "+detector+" "+MC_id+" "+sim_path+" "+str(FCCD)+" "+str(DLF)+" "+energy_filter+" "+cuts+" "+str(run)+" "+source)


if __name__ == "__main__":
    main()
