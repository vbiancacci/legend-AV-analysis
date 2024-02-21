import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import argparse
import os
from scipy import optimize
from scipy import stats

#from GammaLine_Counting_TL_th import chi_sq_calc

#script to determine the FCCD of a detector using the count ratio observable, ignoring TL for now

def main():

    if(len(sys.argv) != 9):
        print('Example usage: python Am241_Calibration.py <detector> <MC_id> <smear> <TL_model> <frac_FCCDbore> <energy_filter> <cuts> <run> <source>')
        sys.exit()

    detector = sys.argv[1]
    MC_id = sys.argv[2]#e.g "V02160A-ba_HS4-top-0r-78z"
    smear=sys.argv[3]
    TL_model=sys.argv[4] #e.g. "notl"
    frac_FCCDbore=sys.argv[5] #e.g. 0.5
    energy_filter = sys.argv[6] #e.g trapEftp
    cuts = sys.argv[7] #e.g. False
    run = int(sys.argv[8]) #e.g 1 or 2

    print("detector: ", detector)
    print("MC_id: ", MC_id)
    print("smear: ", smear)
    print("TL_model: ", TL_model)
    print("frac_FCCDbore: ", frac_FCCDbore)
    print("energy_filter: ", energy_filter)
    print("applying data cuts: ", cuts)
    print("data run: ", run)

    if cuts == "False":
        cuts = False
    else:
        cuts = True

    dir=os.path.dirname(os.path.realpath(__file__))
    print("working directory: ", dir)

    #initialise directories to save
#    if not os.path.exists(dir+"/FCCD/"+source+"plots/"):
#        os.makedirs(dir+"/FCCD/"+source+"/plots/")

    print("start...")

    #Get O_ba133 for each FCCD
    DLF_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    O_list = []
    O_err_list = []


    for DLF in DLF_list:
        smear="g"
        frac_FCCDbore=0.5
        TL_model="l"
        FCCD=1.017
        source_z = "42z"
        MC_id=detector+"-th_HS2-top-0r-42z_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_fracFCCDbore"+str(frac_FCCDbore)

        #Get count ratio for simulations
        with open(dir+"/PeakCounts/"+detector+"/TL/PeakCounts_sim_"+MC_id+'.json') as json_file:
            PeakCounts = json.load(json_file)
            O = PeakCounts['Obs']
            O_err = PeakCounts['Obs_err']/5

            O_list.append(O)
            #O_err = uncertainty(O, O_err)
            O_err_list.append(O_err)

    #Get count ratio for data

    with open(dir+"/PeakCounts/"+detector+"/TL/PeakCounts_data.json") as json_file:
        PeakCounts = json.load(json_file)
        O_data = PeakCounts['Obs']
        O_data_err = PeakCounts['Obs_err']/1.5

    #plot and fit exp decay
    xdata, ydata = np.array(DLF_list), np.array(O_list)
    yerr = O_err_list#*ydata/100

    aguess = -1
    bguess = max(ydata)
    p_guess = [aguess,bguess]
    popt, pcov = optimize.curve_fit(linear, xdata, ydata, p0=p_guess, sigma = yerr, maxfev = 10**7, method ="trf") #, bounds = bounds)
    a,b = popt[0],popt[1]#,popt[2]
    print(a,b)
    a_err, b_err = np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1])#,np.sqrt(pcov[2][2])
    #chi_sq, p_value, residuals, dof = chi_sq_calc(xdata, ydata, yerr, exponential_decay, popt)

    fig, ax = plt.subplots(figsize=(9,10))
    plt.errorbar(xdata, ydata, xerr=0, yerr =yerr, label = "simulations", elinewidth = 1, fmt='x', ms = 3.0, mew = 3.0)
    xfit = np.linspace(min(xdata), max(xdata), 1000)
    #yfit = exponential_decay(xfit,*popt)
    yfit = linear(xfit,a,b)
    plt.plot(xfit, yfit, "g", label = "fit: a*x+b")


    #fit exp decay of error bars:
    y_uplim = ydata+yerr
    p_guess_up = [- 1,max(y_uplim)]
    popt_up, pcov_up = optimize.curve_fit(linear, xdata, y_uplim, p0=p_guess_up, maxfev = 10**7, method ="trf") #, bounds = bounds)
    yfit_up = linear(xfit,*popt_up)
    plt.plot(xfit, yfit_up, color='grey', linestyle='dashed', linewidth=1)

    y_lowlim = ydata-yerr
    p_guess_low = [-1, max(y_lowlim)]#, min(y_lowlim)]
    popt_low, pcov_low = optimize.curve_fit(linear, xdata, y_lowlim, p0=p_guess_low, maxfev = 10**7, method ="trf") #, bounds = bounds)
    yfit_low = linear(xfit,*popt_low)
    plt.plot(xfit, yfit_low, color='grey', linestyle='dashed', linewidth=1)


    #calculate FCCD of data - invert eq
    DLF_data = (O_data-b)/a

    #calculate total error on FCCD
    a_up, b_up = popt_up[0], popt_up[1]
    DLF_data_err_up = (O_data-O_data_err-b_up)/a_up - DLF_data
    a_low, b_low = popt_low[0], popt_low[1]
    DLF_data_err_low = DLF_data - (O_data+O_data_err-b_low)/a_low
    print('DLF of data extrapolated: '+str(DLF_data) +" + "+ str(DLF_data_err_up) +" - "+str(DLF_data_err_low))

    props = dict(boxstyle='round', alpha=0.5, facecolor="lightblue")
    info_str = '\n'.join((r'$a=%.3f \pm %.3f$' % (a, np.sqrt(pcov[0][0])), r'$b=%.3f \pm %.3f$' % (b, np.sqrt(pcov[1][1])), r'DLF=$%.2f^{+%.2f}_{-%.2f}$ mm' % (DLF_data, DLF_data_err_up, DLF_data_err_low)))
    plt.text(0.4, .98, info_str, transform=ax.transAxes, fontsize=12,verticalalignment='top', bbox=props) #ax.text..ax.tra

    #plot data line
    plt.hlines(O_data, 0, DLF_list[-1], colors="orange", label = 'data')
    plt.plot(xfit, [O_data+O_data_err]*(len(xfit)), label = 'bounds', color = 'grey', linestyle = 'dashed', linewidth = '1.0')
    plt.plot(xfit, [O_data-O_data_err]*(len(xfit)), color = 'grey', linestyle = 'dashed', linewidth = '1.0')


    plt.vlines(DLF_data, 0, O_data, colors='orange', linestyles='dashed')
    plt.vlines(DLF_data+DLF_data_err_up, 0, O_data-O_data_err, colors='grey', linestyles='dashed', linewidths=1)
    plt.vlines(DLF_data-DLF_data_err_low, 0, O_data+O_data_err, colors='grey', linestyles='dashed', linewidths=1)


    plt.ylabel(r'$O_{DLF}=\frac{tail_{60keV}}{peak_{60keV}}$', fontsize=20)
    plt.xlabel("DLF", fontsize=20)
    ax.tick_params(axis="both", labelsize=15)
    plt.xlim(0,DLF_list[-1])
    plt.ylim(0.07,0.12) #1000
    #plt.title(detector)
    plt.legend(loc="upper right", fontsize=10)

    if not os.path.exists(dir+"/TL/plots/"):
        os.makedirs(dir+"/TL/plots/")
    plt.savefig(dir+"/TL/plots/TL_"+MC_id+".pdf")

    '''
    #Save interpolated fccd for data to a json file
    FCCD_data_dict = {
        "FCCD": FCCD_data,
        "FCCD_err_up": FCCD_data_err_up,
        "FCCD_err_low": FCCD_data_err_low,
        "FCCD_uncorr_err_up": FCCD_data_uncerr_up,
        "FCCD_uncorr_err_low": FCCD_data_uncerr_low,
        "FCCD_corr_err_up": FCCD_data_corerr_up,
        "FCCD_corr_err_low": FCCD_data_corerr_low,
        "O_Am241_data": O_Am241_data,
        "O_Am241_data_err": O_Am241_data_err,
        "a": a,
        "a_err": (a_up-a_low)/2,
        "b": b,
        "b_err": b_err
    }

    if cuts == False:
        with open(dir+"/FCCD/"+source+"/FCCD_data"+MC_id+"_"+smear+"_"+DLF_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+".json", "w") as outfile:
            json.dump(FCCD_data_dict, outfile, indent=4)
            print("json file saved")
    else:
        if cuts_sigma ==4:
            with open(dir+"/FCCD/"+source+"/FCCD_data_"+MC_id+"_"+smear+"_"+DLF_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+"_cuts_source_0.2thick.json", "w") as outfile:
                json.dump(FCCD_data_dict, outfile, indent=4)
        else:
            with open(dir+"/FCCD/"+source+"/FCCD_data_"+MC_id+"_"+smear+"_"+DLF_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+"_cuts_"+str(cuts_sigma)+"sigma.json", "w") as outfile:
                json.dump(FCCD_data_dict, outfile, indent=4)
    '''
    print("done")

def linear(x, a, b):
    f = a*x+b
    return f

def uncertainty(O_Am241, O_Am241_err):

    #values from Bjoern's thesis - Am source
    #all percentages
    gamma_line=1.81
    geant4=2.
    source_thickness=0.02
    source_material=0.01
    endcap_thickness=0.31
    detector_cup_thickness=0.03
    detector_cup_material=0.01

    MC_statistics = O_Am241_err/O_Am241*100

    #sum squared of all the contributions
    tot_error=np.sqrt(gamma_line**2+geant4**2+source_thickness**2+source_material**2+endcap_thickness**2+detector_cup_thickness**2+detector_cup_material**2+MC_statistics**2)

    return tot_error #NB: tot_error is a percentage error


if __name__ == "__main__":
    main()
