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
    order_list = [8] #[2, 4, 5, 7] #List of orders to process
    Calibrate_Data = False #Pre-reqs: needs load data
    Gamma_line_count_data = False #Pre-reqs: needs calibration
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
        #detectors=["V04549B","V04545A"]#["V02162B","V02166B",],"V04545A"]#["V05261B","V05266A","V05266B","V05267B","V05268A","V05612A","V05612B"]#["V05261B","V05266A","V05266B"]#["V02160A","V02160B","V02162B","V02166B"]#["V07298B", "V07302A", "V07302B", "V07647A","V07647B"]#,"V05612A","V05612B"]#["V05261B","V05266A","V05266B","V04545A","V04549A","V04549B"]#"V05267B","V05268A","V05612A","V05612B"]#["V02160A","V02160B","V02162B","V02166B"]
        #positions=["top_46r_4z","top_82r_5z"]#"top_50r_3z","top_35r_7z",["top_46r_3z","top_46r_3z","top_46r_3z","top_50r_4z","top_45r_4z","top_46r_4z","top_46r_2z"]#["top_46r_3z","top_46r_3z","top_50r_4z"]#["top_68r_4z","top_46r_15z","top_70r_3z","top_65r_7z"]#["top_46r_3z","top_46r_3z","top_46r_3z","top_62r_5z","top_46r_4z","top_46r_4z"]#["top_47r_4z","top_46r_3z","top_48r_4z","top_46r_2z","top_46r_3z"]#["top_47r_4z","top_46r_3z","top_48r_4z","top_46r_2z","top_46r_3z"]#,"top_46r_4z"]
        #for detector, position in zip(detectors,positions):
        for detector in detectors:
            if detector != "V08682A":
                 continue

            energy_filter="cuspEmax_ctc"
            cuts="True"
            if detector=='V02160A'or detector=='V08682A':# or detector =='V04545A':
                run=3
            elif detector=='V02166B' or detector=='V04545A'or detector=='V02162B':
                run=2
            else:
                run=1

            smear="g"
            TL_model="notl"
            frac_FCCDbore=0.5
            source = "am_HS1"
            position=positions_list_data[detector]
            #========Calibration - DATA==========
            if Calibrate_Data == True:

                if order == 2:
                    detector_oldname = "I"+detector[1:]
                    data_path = "/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/gen/"+detector_oldname+"/tier2/"+source+"_top_dlt/"
                else:
                    data_path="/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/gen/"+detector+"/tier2/"+source+"_top_dlt/"



                os.system("python3 "+CodePath+"/Calibration_Am241.py "+detector+" "+data_path+" "+energy_filter+" "+cuts+" "+str(run)+" "+source)


            #========GAMMA LINE COUNTING - DATA==========
            if Gamma_line_count_data == True:

                os.system("python3 "+CodePath+"/GammaLine_Counting_Am241.py --data "+detector+" "+energy_filter+" "+cuts+" "+str(run)+" "+source)

            #=========GAMMA LINE COUNTING - MC=============
            if Gamma_line_count_MC == True:

                DLF_list=[1.0]
                FCCD_list=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4,1.41, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]

                for FCCD in FCCD_list:

                    for DLF in DLF_list:
                        MC_id=detector+"-"+source+"-"+position+"_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_fracFCCDbore"+str(frac_FCCDbore)
                        sim_path="/lfs/l1/legend/detector_char/enr/hades/simulations/legend-g4simple-simulation/simulations/"+detector+"/"+source+"/"+position.replace("-","_")+"/hdf5/AV_processed/"+MC_id+".hdf5"
                        os.system("python3 "+CodePath+"/GammaLine_Counting_Am241.py --sim "+detector+" "+sim_path+" "+MC_id+" "+source)

            #=============Calculate FCCD===============
            if Calculate_FCCD == True:

                MC_id=detector+"-"+source+"-"+position

                os.system("python3 "+CodePath+"/Calculate_FCCD_am1.py "+detector+" "+MC_id+" "+smear+" "+TL_model+" "+str(frac_FCCDbore)+" "+energy_filter+" "+cuts+" "+str(run)+" "+source)

            #=========GAMMA LINE COUNTING - MC, best fit FCCD=============
            if Gamma_line_count_MC_bestfitFCCD == True:

                DLF=1.0

                if cuts == "False":
                    with open(CodePath+"/FCCD/"+source+"/FCCD_data_"+detector+"/"+source+"-"+position+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+".json") as json_file:
                        FCCD_data = json.load(json_file)
                else:
                    with open(CodePath+"/FCCD/"+source+"/FCCD_data_"+detector+"/"+source+"-"+position+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_cuts.json") as json_file:
                        FCCD_data = json.load(json_file)
                FCCD = round(FCCD_data["FCCD"],2)

                MC_id=detector+"-"+source+"-"+position.replace("_","-")+"_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_fracFCCDbore"+str(frac_FCCDbore)
                sim_path="/lfs/l1/legend/detector_char/enr/hades/simulations/legend-g4simple-simulation/simulations/"+detector+"/"+source+"/"+position.replace("-","_")+"/hdf5/AV_processed/"+MC_id+".hdf5"
                os.system("python3 "+CodePath+"/GammaLine_Counting_Am241.py --sim "+detector+" "+sim_path+" "+MC_id+" "+source)


            #=============Plot Spectra===============
            if PlotSpectra == True:

                DLF=1.0

                if cuts == "False":
                    with open(CodePath+"/FCCD/FCCD_data_"+detector+"/"+source+"-"+position+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+".json") as json_file:
                        FCCD_data = json.load(json_file)
                else:
                    with open(CodePath+"/FCCD/FCCD_data_"+detector+"/"+source+"-"+position+"_"+smear+"_"+TL_model+"_fracFCCDbore"+str(frac_FCCDbore)+"_"+energy_filter+"_run"+str(run)+"_cuts.json") as json_file:
                        FCCD_data = json.load(json_file)
                FCCD = round(FCCD_data["FCCD"],2)

                MC_id=detector+"/"+source+"-"+position.replace("_","-")+"_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_fracFCCDbore"+str(frac_FCCDbore)
                sim_path="/lfs/l1/legend/detector_char/enr/hades/simulations/legend-g4simple-simulation/simulations/"+detector+"/"+source+"/"+position.replace("-","_")+"/hdf5/AV_processed/"+MC_id+".hdf5"

                os.system("python3 "+CodePath+"/PlotSpectra.py "+detector+" "+MC_id+" "+sim_path+" "+str(FCCD)+" "+str(DLF)+" "+energy_filter+" "+cuts+" "+str(run)+" "+source)


if __name__ == "__main__":
    main()
