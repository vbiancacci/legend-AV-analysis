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
    order_list = [0] #List of orders to process
    Calibrate_Data = False #Pre-reqs: needs dsp pygama data
    Gamma_line_count_data = False #Pre-reqs: needs calibration
    Gamma_line_count_MC = False #Pre-reqs: needs AV post processed MC for range of FCCDs
    Calculate_FCCD =  True #Pre-reqs: needs gammaline counts for data and MC
    Gamma_line_count_MC_bestfitFCCD = False #Pre-reqs: needs AV postprocessed MC for best fit FCCD
    PlotSpectra = False #Pre-reqs: needs all above stages

    #Get detector list
    detector_list = CodePath+"/../detector_list.json"
    with open(detector_list) as json_file:
        detector_list_data = json.load(json_file)

    for order in order_list:
        detectors = detector_list_data["order_"+str(order)]
        for detector in detectors:

            if detector != "B00091B":
                continue

            if order == 7 or order==8:
                run=2
            elif detector=="B00032B": # detector=="B00035A" or
                run=3
            elif detector=="B00035A":
                run=3
            else:
                run=2

            energy_filter="cuspEmax_ctc" #"trapEmax" #"cuspEmax_ctc"
            cuts="False"

            #========Calibration - DATA==========
            if Calibrate_Data == True:

                if order == 2:
                    detector_oldname = "I"+detector[1:]
                    data_path = "/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/gen/"+detector_oldname+"/tier2/ba_HS4_top_dlt/"
                else:
                    data_path="/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/gen/"+detector+"/tier2/ba_HS4_top_dlt/"
                    #data_path = "/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-full_dl-v01/gen/"+detector+"/tier2/ba_HS4_top_dlt/"

                os.system("python "+CodePath+"/Calibration_Ba133.py "+detector+" "+data_path+" "+energy_filter+" "+cuts+" "+str(run))


            #========GAMMA LINE COUNTING - DATA==========
            if Gamma_line_count_data == True:
                if order == 2:
                    detector_oldname = "I"+detector[1:]
                    data_path = "/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/gen/"+detector_oldname+"/tier2/ba_HS4_top_dlt/"
                    # calibration="/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/genpar/dsp_ecal/"+detector_oldname+".json"

                else:
                    #data_path="/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/gen/"+detector+"/tier2/ba_HS4_top_dlt/"
                    data_path = "/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-full_dl-v01/gen/"+detector+"/tier2/ba_HS4_top_dlt/"
                    # calibration="/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/genpar/dsp_ecal/"+detector+".json"

                if cuts == "False":
                    calibration = CodePath+"/data_calibration/"+detector+"/calibration_run"+str(run)+".json"
                else:
                    calibration = CodePath+"/data_calibration/"+detector+"/calibration_run"+str(run)+"_cuts.json"


                os.system("python "+CodePath+"/GammaLine_Counting_Ba133.py --data "+detector+" "+data_path+" "+calibration+" "+energy_filter+" "+cuts+" "+str(run))

            #=========GAMMA LINE COUNTING - MC=============
            if Gamma_line_count_MC == True:

                DLF_list=[1.0]
                smear="g"
                frac_FCCDbore=0.5
                TL_model="notl"
                FCCD_list= [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]#, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.] #[0.0,0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 3.0]
                alpha_list= [0.8, 0.9]
                beta_list= [0.3, 0.4]

                if order == 8:
                    source_z = "88z" #top-0r-78z
                elif order ==9:
                    source_z = "74z"
                else:
                    source_z = "78z"

                #source_z="81z"
                for FCCD in FCCD_list:
                    #for alpha in alpha_list:
                        #for beta in beta_list:
                        for DLF in DLF_list:
                            #MC_id=detector+"-ba_HS4-top-0r-81z_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+"mm__alpha"+str(alpha)+"_beta"+str(beta)+"_fracFCCDbore"+str(frac_FCCDbore)
                            #sim_path="/lfs/l1/legend/detector_char/enr/hades/simulations/legend-g4simple-simulation/simulations/"+detector+"/ba_HS4/top_0r_81z/hdf5/AV_processed_test/"+MC_id+".hdf5"

                            MC_id=detector+"-ba_HS4-top-0r-"+source_z+"_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_fracFCCDbore"+str(frac_FCCDbore)
                            #sim_path="/lfs/l1/legend/users/aalexander/legend-g4simple-simulation/simulations/"+detector+"/ba_HS4/top_0r_"+source_z+"/hdf5/AV_processed/"+MC_id+".hdf5"
                            sim_path="/lfs/l1/legend/detector_char/enr/hades/simulations/legend-g4simple-simulation/simulations/"+detector+"/ba_HS4/top_0r_"+source_z+"/hdf5/AV_processed/"+MC_id+".hdf5"
                            os.system("python "+CodePath+"/GammaLine_Counting_Ba133.py --sim "+detector+" "+sim_path+" "+MC_id)

            #=============Calculate FCCD===============
            if Calculate_FCCD == True:

                if order == 8:
                    source_z = "88z" #top-0r-78z
                elif order ==9:
                    source_z = "74z"
                else:
                    source_z = "78z"

                # MC_id=detector+"-ba_HS4-top-0r-78z"
                MC_id=detector+"-ba_HS4-top-0r-"+source_z
                smear="g"
                TL_model="notl"
                frac_FCCDbore=0.5


                os.system("python "+CodePath+"/Calculate_FCCD_abi.py "+detector+" "+MC_id+" "+smear+" "+TL_model+" "+str(frac_FCCDbore)+" "+energy_filter+" "+cuts+" "+str(run))

            #=========GAMMA LINE COUNTING - MC, best fit FCCD=============
            if Gamma_line_count_MC_bestfitFCCD == True:

                DLF=1.0
                smear="g"
                frac_FCCDbore=0.5
                TL_model="notl"

                if order == 8:
                    source_z = "88z" #top-0r-78z
                elif order ==9:
                    source_z = "74z"
                else:
                    source_z = "198z"


                if cuts == "False":
                    with open(CodePath+"/FCCD/new/FCCD_data_"+detector+"-ba_HS4-top-0r-"+source_z+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+".json") as json_file:
                        FCCD_data = json.load(json_file)
                else:
                    with open(CodePath+"/FCCD/FCCD_data_"+detector+"-ba_HS4-top-0r-"+source_z+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_cuts.json") as json_file:
                        FCCD_data = json.load(json_file)

                FCCD = round(FCCD_data["FCCD"],2)
                TL_model="notl"

                MC_id=detector+"-ba_HS4-top-0r-"+source_z+"_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_fracFCCDbore"+str(frac_FCCDbore)
                #sim_path="/lfs/l1/legend/users/aalexander/legend-g4simple-simulation/simulations/"+detector+"/ba_HS4/top_0r_"+source_z+"/hdf5/AV_processed/"+MC_id+".hdf5"
                sim_path="/lfs/l1/legend/detector_char/enr/hades/simulations/legend-g4simple-simulation/simulations/"+detector+"/ba_HS4/top_0r_"+source_z+"/hdf5/AV_processed/"+MC_id+".hdf5"
                os.system("python "+CodePath+"/GammaLine_Counting_Ba133.py --sim "+detector+" "+sim_path+" "+MC_id)


            #=============Plot Spectra===============
            if PlotSpectra == True:

                if order == 2:
                    detector_oldname = "I"+detector[1:]
                    data_path = "/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/gen/"+detector_oldname+"/tier2/ba_HS4_top_dlt/"
                    # calibration="/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/genpar/dsp_ecal/"+detector_oldname+".json"

                else:
                    data_path="/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/gen/"+detector+"/tier2/ba_HS4_top_dlt/"
                    # calibration="/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/genpar/dsp_ecal/"+detector+".json"
                    #data_path = "/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-full_dl-v01/gen/"+detector+"/tier2/ba_HS4_top_dlt/"

                if cuts == "False":
                    calibration = CodePath+"/data_calibration/"+detector+"/calibration_run"+str(run)+".json"
                else:
                    calibration = CodePath+"/data_calibration/"+detector+"/calibration_run"+str(run)+"_cuts.json"

                DLF=1.0
                smear="g"
                frac_FCCDbore=0.5
                TL_model="notl"
                if order == 8:
                    source_z = "88z" #top-0r-78z
                elif order ==9:
                    source_z = "74z"
                else:
                    source_z = "78z"

                if cuts == "False":
                    with open(CodePath+"/FCCD/new/FCCD_data_"+detector+"-ba_HS4-top-0r-"+source_z+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+".json") as json_file:
                        FCCD_data = json.load(json_file)
                else:
                    with open(CodePath+"/FCCD/FCCD_data_"+detector+"-ba_HS4-top-0r-"+source_z+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_cuts.json") as json_file:
                        FCCD_data = json.load(json_file)

                FCCD = round(FCCD_data["FCCD"],2)
                #TL_model="l"

                MC_id=detector+"-ba_HS4-top-0r-"+source_z+"_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_fracFCCDbore"+str(frac_FCCDbore)
                #sim_path="/lfs/l1/legend/users/aalexander/legend-g4simple-simulation/simulations/"+detector+"/ba_HS4/top_0r_"+source_z+"/hdf5/AV_processed/"+MC_id+".hdf5"
                sim_path="/lfs/l1/legend/detector_char/enr/hades/simulations/legend-g4simple-simulation/simulations/"+detector+"/ba_HS4/top_0r_"+source_z+"/hdf5/AV_processed/"+MC_id+".hdf5"
                os.system("python "+CodePath+"/PlotSpectra.py "+detector+" "+MC_id+" "+sim_path+" "+str(FCCD)+" "+str(DLF)+" "+data_path+" "+calibration+" "+energy_filter+" "+cuts+" "+str(run))



if __name__ == "__main__":
    main()
