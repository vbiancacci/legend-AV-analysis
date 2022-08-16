from cmath import nan
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

    batch_1 = ["B00000B", "B00032B", "B00091B"]
    batch_2 = ["B00000D", "B00035A", "B00035B", "B00002C", "B00061C", "B00076C"]
    detectors = batch_1 + batch_2

    FCCD_Ba = []
    FCCD_Ba_err = []

    data = {"B00000B": {"batch":1,"QC":{"FCCD_Ba":1.32,"err_Ba":0.06, "FCCD_Am":1.52,"err_Am":0.05}, "noQC": {"FCCD_Ba":1.36,"err_Ba":0.06, "FCCD_Am":1.55,"err_Am":0.05} },
           
            "B00091B": {"batch":1,"QC":{"FCCD_Ba":1.05,"err_Ba":0.06, "FCCD_Am":1.32,"err_Am":0.05}, "noQC": {"FCCD_Ba":1.15,"err_Ba":0.06, "FCCD_Am":1.32,"err_Am":0.05} },
           
            "B00032B": {"batch":1,"QC":{"FCCD_Ba":1.15,"err_Ba":0.06, "FCCD_Am":1.34,"err_Am":0.05}, "noQC": {"FCCD_Ba":1.14,"err_Ba":0.06, "FCCD_Am":1.38,"err_Am":0.05} },
           
            "B00000D": {"batch":2,"QC":{"FCCD_Ba":0.83,"err_Ba":0.07, "FCCD_Am":0.98,"err_Am":0.05}, "noQC": {"FCCD_Ba":0.87,"err_Ba":0.07, "FCCD_Am":1.05,"err_Am":0.05} },

            "B00035A": {"batch":2,"QC":{"FCCD_Ba":1.59,"err_Ba":0.06, "FCCD_Am":1.69,"err_Am":0.08}, "noQC": {"FCCD_Ba":1.59,"err_Ba":0.06, "FCCD_Am":1.69,"err_Am":0.08} },

            "B00035B": {"batch":2,"QC":{"FCCD_Ba":1.24,"err_Ba":0.07, "FCCD_Am":1.37,"err_Am":0.07}, "noQC": {"FCCD_Ba":1.25,"err_Ba":0.07, "FCCD_Am":1.38,"err_Am":0.07} },

            "B00002C": {"batch":2,"QC":{"FCCD_Ba":0.98,"err_Ba":0.07, "FCCD_Am":1.13,"err_Am":0.06}, "noQC": {"FCCD_Ba":0.99,"err_Ba":0.07, "FCCD_Am":1.13,"err_Am":0.06} },

            "B00061C": {"batch":2,"QC":{"FCCD_Ba":0.79,"err_Ba":0.07, "FCCD_Am":0.99,"err_Am":0.06}, "noQC": {"FCCD_Ba":0.81,"err_Ba":0.07, "FCCD_Am":0.99,"err_Am":0.06} },

            "B00076C": {"batch":2,"QC":{"FCCD_Ba":0.93,"err_Ba":0.07, "FCCD_Am":1.08,"err_Am":0.06}, "noQC": {"FCCD_Ba":0.95,"err_Ba":0.07, "FCCD_Am":1.09,"err_Am":0.06} },

            }

    fig, ax = plt.subplots(figsize=(12,8))
    # marker_Ba, marker_Am = "o", "s"
    # # colors = {}
    # color_QC = "red"
    # color_noQC = "blue"

    color_Ba, color_Am = "blue", "orange"
    color_Av, color_weighted_Av = "red", "red"
    marker_QC, marker_noQC = "x", "."


    for det in data:

        ax.errorbar([det], [data[det]["QC"]["FCCD_Ba"]], yerr = [data[det]["QC"]["err_Ba"]], marker = marker_QC, color=color_Ba)
        ax.errorbar([det], [data[det]["QC"]["FCCD_Am"]], yerr = [data[det]["QC"]["err_Am"]], marker = marker_QC, color=color_Am)
        
        av_QC = (data[det]["QC"]["FCCD_Ba"] + data[det]["QC"]["FCCD_Am"])/2
        weighted_av_QC = ( (data[det]["QC"]["FCCD_Ba"])/(data[det]["QC"]["err_Ba"])**2 +  (data[det]["QC"]["FCCD_Am"])/(data[det]["QC"]["err_Am"])**2)/(1/(data[det]["QC"]["err_Ba"])**2 + 1/(data[det]["QC"]["err_Am"])**2)
        ax.errorbar([det], [weighted_av_QC], marker = marker_QC, color=color_weighted_Av)

        ax.errorbar([det], [data[det]["noQC"]["FCCD_Ba"]], yerr = [data[det]["noQC"]["err_Ba"]], marker = marker_noQC, color=color_Ba)
        ax.errorbar([det], [data[det]["noQC"]["FCCD_Am"]], yerr = [data[det]["noQC"]["err_Am"]], marker = marker_noQC, color=color_Am)
        av_noQC = (data[det]["noQC"]["FCCD_Ba"] + data[det]["noQC"]["FCCD_Am"])/2
        weighted_av_noQC = ( (data[det]["noQC"]["FCCD_Ba"])/(data[det]["noQC"]["err_Ba"])**2 +  (data[det]["noQC"]["FCCD_Am"])/(data[det]["noQC"]["err_Am"])**2)/(1/(data[det]["noQC"]["err_Ba"])**2 + 1/(data[det]["noQC"]["err_Am"])**2)

        # ax.errorbar([det], [av], marker = marker_noQC, color=color_Av)
        ax.errorbar([det], [weighted_av_noQC], marker = marker_noQC, color=color_weighted_Av)
        
        print(det)
        print(weighted_av_noQC)

    ax2 = ax.twinx()
    ax2.plot(np.NaN, np.NaN, marker=marker_QC,c='grey',label="QC")
    ax2.plot(np.NaN, np.NaN, marker=marker_noQC,c='grey',label="no QC")
    ax2.get_yaxis().set_visible(False)
    ax2.legend()#loc='upper left', bbox_to_anchor=(0, 0.8))

    ax3 = ax.twinx()
    ax3.plot(np.NaN, np.NaN,c=color_Ba,label="Ba-133")
    ax3.plot(np.NaN, np.NaN,c=color_Am,label="Am-241")
    # ax3.plot(np.NaN, np.NaN,c=color_Av,label="Average")
    ax3.plot(np.NaN, np.NaN,c=color_weighted_Av,label="Weighted mean")
    ax3.get_yaxis().set_visible(False)
    ax3.legend(loc='upper left')#, bbox_to_anchor=(0.8, 0.8))


    ax.tick_params(axis='x', labelrotation=45)
    ax.set_xlabel('Detector')
    ax.set_ylabel('FCCD (mm)')
    # ax.grid(linestyle='dashed', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("FCCD_Gerda_BEGe_comparison.png")
    plt.show()


    #plot comparison plot
    old_HADES = {"B00035B":0.55, "B00000D":0.74, "B00002C":0.75, "B00035A":0.61,"B00061C":0.68,"B00076C":0.86, "B00000B":0.76,"B00032B":0.8,"B00091B":0.68}
    official = {"B00035B":0.78, "B00000D":1.03, "B00002C":1.03, "B00035A":0.95,"B00061C":0.93,"B00076C":1.15, "B00000B":1.04,"B00032B":1.05,"B00091B":0.95}

    fig, ax = plt.subplots(figsize=(12,8))
    color_new = "red"
    color_old = "blue"
    color_official = "green"

    for det in data:

        weighted_av_noQC = ( (data[det]["noQC"]["FCCD_Ba"])/(data[det]["noQC"]["err_Ba"])**2 +  (data[det]["noQC"]["FCCD_Am"])/(data[det]["noQC"]["err_Am"])**2)/(1/(data[det]["noQC"]["err_Ba"])**2 + 1/(data[det]["noQC"]["err_Am"])**2)
        ax.errorbar([det], [weighted_av_noQC], color=color_new)
        ax.errorbar([det], old_HADES[det], color=color_old)
        ax.errorbar([det], official[det], color=color_official)
    
    



if __name__ == "__main__":
    main()