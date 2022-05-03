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

from GammaLine_Counting_Am241_HS1 import chi_sq_calc

#script to determine the FCCD of a detector using the count ratio observable, ignoring TL for now

def main():

    if(len(sys.argv) != 10):
        print('Example uage: python Am241_Calibration.py <detector> <MC_id> <smear> <TL_model> <frac_FCCDbore> <energy_filter> <cuts> <run> <source>')
        sys.exit()

    detector = sys.argv[1]
    MC_id = sys.argv[2]#e.g "V02160A-ba_HS4-top-0r-78z"
    smear=sys.argv[3]
    TL_model=sys.argv[4] #e.g. "notl"
    frac_FCCDbore=sys.argv[5] #e.g. 0.5
    energy_filter = sys.argv[6] #e.g trapEftp
    cuts = sys.argv[7] #e.g. False
    run = int(sys.argv[8]) #e.g 1 or 2
    source = sys.argv[9]

    print("detector: ", detector)
    print("MC_id: ", MC_id)
    print("smear: ", smear)
    print("TL_model: ", TL_model)
    print("frac_FCCDbore: ", frac_FCCDbore)
    print("energy_filter: ", energy_filter)
    print("applying data cuts: ", cuts)
    print("data run: ", run)
    print("source: ", source)

    if cuts == "False":
        cuts = False
    else:
        cuts = True

    dir=os.path.dirname(os.path.realpath(__file__))
    print("working directory: ", dir)

    calibration="BEGe"
    a=0
    a_err=0
    if calibration=="BEGe":
        a=57.751173630527326
        arr=2.892691574705301
    else: #IC
        a=58.18886615638756
        arr=2.1024957776256397
    #initialise directories to save
#    if not os.path.exists(dir+"/FCCD/"+source+"plots/"):
#        os.makedirs(dir+"/FCCD/"+source+"/plots/")

    print("start...")

    #Get O_ba133 for each FCCD
    FCCD_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0] #mm
    O_Am241_list = []
    O_Am241_err_pct_list = []
    DLF = 1.0 #considering 0 TL

    for FCCD in FCCD_list:

        #Get count ratio for simulations
        with open(dir+"/PeakCounts/"+detector+"/"+source+"/new/PeakCounts_sim_"+MC_id+"_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+'mm_DLF'+str(DLF)+'_fracFCCDbore'+frac_FCCDbore+'.json') as json_file:
            PeakCounts = json.load(json_file)
            C_60 = PeakCounts['C_60']
            C_99_103 = PeakCounts['C_99_103']
            O_Am241 = PeakCounts['O_Am241']
            O_Am241_err = PeakCounts ['O_Am241_err']
            O_Am241_list.append(O_Am241)

            #get errors:
            O_Am241_err_pct = uncertainty(O_Am241, O_Am241_err)
            O_Am241_err_pct_list.append(O_Am241_err_pct)

    #Get count ratio for data
    if cuts == False:
        with open(dir+"/PeakCounts/"+detector+"/"+source+"/new/PeakCounts_data_"+detector+"_"+energy_filter+"_run"+str(run)+".json") as json_file:
            PeakCounts = json.load(json_file)
            C_60 = PeakCounts['C_60']
            C_99_103 = PeakCounts['C_99_103']
            O_Am241_data = PeakCounts['O_Am241']
            O_Am241_data_err = PeakCounts ['O_Am241_err']
    else:
        cuts_sigma = 4 #default =4, change by hand here if interested
        print("data cuts sigma: ", str(cuts_sigma))
        if cuts_sigma == 4:
            with open(dir+"/PeakCounts/"+detector+"/"+source+"/new/PeakCounts_data_"+detector+"_cuts_"+energy_filter+"_run"+str(run)+".json") as json_file:
                PeakCounts = json.load(json_file)
                C_60 = PeakCounts['C_60']
                C_99_103 = PeakCounts['C_99_103']
                O_Am241_data = PeakCounts['O_Am241']
                O_Am241_data_err = PeakCounts ['O_Am241_err']
        else:
            with open(dir+"/PeakCounts/"+detector+"/"+source+"/new/PeakCounts_data_"+detector+"_cuts_"+energy_filter+"_run"+str(run)+"_"+str(sigma_cuts)+"sigma.json") as json_file:
                PeakCounts = json.load(json_file)
                C_60 = PeakCounts['C_60']
                C_99_103 = PeakCounts['C_99_103']
                O_Am241_data = PeakCounts['O_Am241']
                O_Am241_data_err = PeakCounts ['O_Am241_err']

    #plot and fit exp decay
    xdata, ydata = np.array(FCCD_list), np.array(O_Am241_list)
    print(ydata)
    yerr = O_Am241_err_pct_list*ydata/100 #get absolute error, not percentage

    aguess = max(ydata)
    bguess = 1
    #cguess = min(ydata)
    p_guess = [aguess,bguess]#,cguess]
    #bounds=([0, 0, 0, 0, -np.inf], [np.inf]*5)
    popt, pcov = optimize.curve_fit(exponential_decay, xdata, ydata, p0=p_guess, sigma = yerr, maxfev = 10**7, method ="trf") #, bounds = bounds)
    a,b = popt[0],popt[1]#,popt[2]
    print(a,b)
    a_err, b_err = np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1])#,np.sqrt(pcov[2][2])
    chi_sq, p_value, residuals, dof = chi_sq_calc(xdata, ydata, yerr, exponential_decay, popt)

    fig, ax = plt.subplots()
    plt.errorbar(xdata, ydata, xerr=0, yerr =yerr, label = "simulations", elinewidth = 1, fmt='x', ms = 3.0, mew = 3.0)
    xfit = np.linspace(min(xdata), max(xdata), 1000)
    #yfit = exponential_decay(xfit,*popt)
    #a=60.05239428860399  #58.18886615638756 #59.23103180210723 #57.04617122422738  #59.23103180210723 #57.52344368283454
    #59.23 is the IC calib, 57.52 is the IC calib without exp correction, 57.04 is the BEGe calib


    yfit = exponential_decay(xfit,a,b)
    plt.plot(xfit, yfit, "g", label = "fit: a*exp(-bx)")


    #fit exp decay of error bars:
    y_uplim = ydata+yerr
    p_guess_up = [max(y_uplim), 1]#, min(y_uplim)]
    popt_up, pcov_up = optimize.curve_fit(exponential_decay, xdata, y_uplim, p0=p_guess_up, maxfev = 10**7, method ="trf") #, bounds = bounds)
    #arr= 2.051958705758966 #2.944003054128919 #2.1673296109052482 #2.1571049512349667
    #2.16 is the IC calib, 2.15 is the IC calib without exp correction, 2.94 is the BEGe calib
    a_up=a+arr
    yfit_up = exponential_decay(xfit,a_up,b)
    plt.plot(xfit, yfit_up, color='grey', linestyle='dashed', linewidth=1)

    y_lowlim = ydata-yerr
    p_guess_low = [max(y_lowlim), 1]#, min(y_lowlim)]
    popt_low, pcov_low = optimize.curve_fit(exponential_decay, xdata, y_lowlim, p0=p_guess_low, maxfev = 10**7, method ="trf") #, bounds = bounds)
    a_low=a-arr
    yfit_low = exponential_decay(xfit,a_low,b)
    plt.plot(xfit, yfit_low, color='grey', linestyle='dashed', linewidth=1)


    #calculate FCCD of data - invert eq
    FCCD_data = (1/b)*np.log(a/O_Am241_data)

    #calculate error on FCCD
    b_up = popt_up[1]
    #a_up, b_up = popt_up[0], popt_up[1]
    FCCD_data_err_up = (1/b_up)*np.log(a_up/(O_Am241_data-O_Am241_data_err))-FCCD_data
    b_low = popt_low[1]
    #a_low, b_low = popt_low[0], popt_low[1]
    FCCD_data_err_low = FCCD_data - (1/b_low)*np.log(a_low/(O_Am241_data+O_Am241_data_err))
    print('FCCD of data extrapolated: '+str(FCCD_data) +" + "+ str(FCCD_data_err_up) +" - "+str(FCCD_data_err_low))
    #calculate uncorrelated error on FCCD
    FCCD_data_uncerr_up = (1/b)*np.log(a/(O_Am241_data-O_Am241_data_err))-FCCD_data
    FCCD_data_uncerr_low = FCCD_data - (1/b)*np.log(a/(O_Am241_data+O_Am241_data_err))
    print('uncorrelated error:  + '+ str(FCCD_data_uncerr_up) +" - "+str(FCCD_data_uncerr_low))
    #calculate correlated error on FCCD
    FCCD_data_corerr_up=np.sqrt(FCCD_data_err_up**2-FCCD_data_uncerr_up**2)
    FCCD_data_corerr_low=np.sqrt(FCCD_data_err_low**2-FCCD_data_uncerr_low**2)
    print('correlated error:  + '+ str(FCCD_data_corerr_up) +" - "+str(FCCD_data_corerr_low))



    props = dict(boxstyle='round', alpha=0.5)
    info_str = '\n'.join((r'$a=%.3f \pm %.3f$' % (a,arr), r'$b=%.3f \pm %.3f$' % (b, np.sqrt(pcov[1][1])), r'$\chi^2/dof=%.2f/%.0f$'%(chi_sq, dof), r'FCCD_data=$%.2f^{+%.2f}_{-%.2f}$ mm' % (FCCD_data, FCCD_data_err_up, FCCD_data_err_low)))
    plt.text(0.625, 0.775, info_str, transform=ax.transAxes, fontsize=9,verticalalignment='top', bbox=props) #ax.text..ax.tra

    #plot data line
    plt.hlines(O_Am241_data, 0, FCCD_list[-1], colors="orange", label = 'data')
    plt.plot(xfit, [O_Am241_data+O_Am241_data_err]*(len(xfit)), label = 'data bounds', color = 'grey', linestyle = 'dashed', linewidth = '1.0')
    plt.plot(xfit, [O_Am241_data-O_Am241_data_err]*(len(xfit)), color = 'grey', linestyle = 'dashed', linewidth = '1.0')


    plt.vlines(FCCD_data, 0, O_Am241_data, colors='orange', linestyles='dashed')
    plt.vlines(FCCD_data+FCCD_data_err_up, 0, O_Am241_data, colors='grey', linestyles='dashed', linewidths=1)
    plt.vlines(FCCD_data-FCCD_data_err_low, 0, O_Am241_data, colors='grey', linestyles='dashed', linewidths=1)


#    plt.ylabel(r'$O_{Am241} = C_{59.5}/(C_{99_103} C_{103}$')
    plt.ylabel(r'$O_{am\_HS1}=\frac{C_{60keV}}{C_{99keV}+C_{103keV}}$')
    plt.xlabel("FCCD [mm]")
    plt.xlim(0,FCCD_list[-1])
    plt.ylim(0,70)
    #plt.title(detector)
    plt.legend(loc="upper right", fontsize=8)

    if cuts == False:
        plt.savefig(dir+"/FCCD/"+source+"/plots/FCCD_OAm241_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+".png")
    else:
        plt.savefig(dir+"/FCCD/"+source+"/test/plots/FCCD_OAm241_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+"_cuts.png")


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
        "a_err": (a_up-a_low)/2
    }

    if cuts == False:
        with open(dir+"/FCCD/"+source+"/FCCD_data"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+".json", "w") as outfile:
            json.dump(FCCD_data_dict, outfile, indent=4)
    else:
        if cuts_sigma ==4:
            with open(dir+"/FCCD/"+source+"/FCCD_data_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+"_cuts.json", "w") as outfile:
                json.dump(FCCD_data_dict, outfile, indent=4)
        else:
            with open(dir+"/FCCD/"+source+"/test/FCCD_data_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+"_cuts_"+str(cuts_sigma)+"sigma.json", "w") as outfile:
                json.dump(FCCD_data_dict, outfile, indent=4)

    print("done")

def exponential_decay(x, a, b):
    f = a*np.exp(-b*x)
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

    #compute statistical error
    MC_statistics = O_Am241_err/O_Am241*100

    #sum squared of all the contributions
    tot_error=np.sqrt(gamma_line**2+geant4**2+source_thickness**2+source_material**2+endcap_thickness**2+detector_cup_thickness**2+detector_cup_material**2+MC_statistics**2)

    return tot_error #NB: tot_error is a percentage error




if __name__ == "__main__":
    main()
