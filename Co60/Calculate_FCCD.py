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

from GammaLine_Counting_Co60 import chi_sq_calc

#script to determine the FCCD of a detector using the count ratio observable, ignoring TL for now

def main():

    if(len(sys.argv) != 9):
        print('Example usage: python Calculate_FCCD.py <detector> <MC_id> <smear> <TL_model> <frac_FCCDbore> <energy_filter> <cuts> <run>')
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
    if not os.path.exists(dir+"/FCCD/plots/"):
        os.makedirs(dir+"/FCCD/plots/")

    print("start...")

    #Ratio of data/sim
    A_source_today = 2.7912E3#3.1834E3#2.76E3 #Bq
    if detector == "B00035B" or detector == "B00061C":
        data_live_time = 36000 #s, check here: /lfs/l1/legend/detector_char/enr/hades/char_data/B00061C/tier0/co_HS5_top_dlt/meta/char_data-B00061C-co_HS5_top_dlt-run0001.json
    elif detector == "B00000D" or detector == "B00035A" or detector == "B00076C":
        data_live_time = 14400 #s
    elif detector == "B00002C":
        data_live_time = 28800
    elif detector == "B00032B" or detector == "B00091B":
        data_live_time = 75600
    elif detector == "B00000B":
        data_live_time = 10800
    N_data = A_source_today*data_live_time
    N_sims = 10*10**7 #10 files with 10^7 events
    R = N_data/N_sims #need to scale sims by this number
    print("ratio data to MC: ",R)

    #Get O_ba133 for each FCCD
    FCCD_list = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0] #mm
    #FCCD_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    C_1173_list = []
    C_1332_list = []
    C_1173_err_pct_list = []
    C_1332_err_pct_list = []

    DLF = 1.0 #considering 0 TL

    for FCCD in FCCD_list:

        #Get count ratio for simulations
        with open(dir+"/PeakCounts/"+detector+"/PeakCounts_sim_"+MC_id+"_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+'mm_DLF'+str(DLF)+'_fracFCCDbore'+frac_FCCDbore+'.json') as json_file:
            PeakCounts = json.load(json_file)
            C_1173 = R*PeakCounts['C_1173']
            C_1332 = R*PeakCounts['C_1332']
            C_1173_list.append(C_1173)
            C_1332_list.append(C_1332)

            #get errors:
            C_1173_err = R*np.sqrt(C_1173)#/C_1173
            C_1332_err = R*np.sqrt(C_1332)#/C_1332

            #get errors:
            C_1173_err_pct = uncertainty(C_1173, C_1173_err)
            C_1173_err_pct_list.append(C_1173_err_pct)
            C_1332_err_pct = uncertainty(C_1332, C_1332_err)
            C_1332_err_pct_list.append(C_1332_err_pct)



    #Get count ratio for data
    if cuts == False:
        with open(dir+"/PeakCounts/"+detector+"/PeakCounts_data_"+detector+"_"+energy_filter+"_run"+str(run)+".json") as json_file:
            PeakCounts = json.load(json_file)
        C_1173_data = PeakCounts['C_1173']
        C_1332_data = PeakCounts['C_1332']
        C_1173_data_err = np.sqrt(C_1173_data)#PeakCounts['C_1173_err']
        C_1332_data_err = np.sqrt(C_1332_data) #PeakCounts['C_1332_err']
    else:
        cuts_sigma = 4 #default =4, change by hand here if interested
        print("data cuts sigma: ", str(cuts_sigma))
        if cuts_sigma == 4:
            with open(dir+"/PeakCounts/"+detector+"/PeakCounts_data_"+detector+"_cuts_"+energy_filter+"_run"+str(run)+".json") as json_file:
                PeakCounts = json.load(json_file)
            C_1173_data = PeakCounts['C_1173']
            C_1332_data = PeakCounts['C_1332']
        else:
            with open(dir+"/PeakCounts/"+detector+"/PeakCounts_data_"+detector+"_cuts_"+energy_filter+"_run"+str(run)+"_"+str(sigma_cuts)+"sigma.json") as json_file:
                PeakCounts = json.load(json_file)
            C_1173_data = PeakCounts['C_1173']
            C_1332_data = PeakCounts['C_1332']


    #========PLOT 1=============

    print("using Co60 observable 1: C_1173 keV")

    #plot and fit exp decay
    xdata, ydata = np.array(FCCD_list), np.array(C_1173_list)
    #yerr = C_1173_err_pct_list #get absolute error, not percentage
    yerr = C_1173_err_pct_list*ydata/100
    aguess = max(ydata)
    bguess = 1
    cguess = min(ydata)
    p_guess = [aguess,bguess,cguess]
    #bounds=([0, 0, 0, 0, -np.inf], [np.inf]*5)
    popt, pcov = optimize.curve_fit(exponential_decay, xdata, ydata, p0=p_guess, sigma = yerr, maxfev = 10**7, method ="trf") #, bounds = bounds)
    a,b,c = popt[0],popt[1],popt[2]
    a_err, b_err, c_err = np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1]), np.sqrt(pcov[2][2])
    chi_sq, p_value, residuals, dof = chi_sq_calc(xdata, ydata, yerr, exponential_decay, popt)

    fig, ax = plt.subplots()
    plt.errorbar(xdata, ydata, xerr=0, yerr =yerr, label = "simulations", elinewidth = 1, fmt='x', ms = 3.0, mew = 3.0)
    xfit = np.linspace(min(xdata), max(xdata), 1000)
    yfit = exponential_decay(xfit,*popt)
    plt.plot(xfit, yfit, "g", label = "fit: a*exp(-bx)+c")

    #fit exp decay of error bars:
    y_uplim = ydata+yerr
    p_guess_up = [max(y_uplim), 1, min(y_uplim)]
    popt_up, pcov_up = optimize.curve_fit(exponential_decay, xdata, y_uplim, p0=p_guess_up, maxfev = 10**7, method ="trf") #, bounds = bounds)
    yfit_up = exponential_decay(xfit,*popt_up)
    plt.plot(xfit, yfit_up, color='grey', linestyle='dashed', linewidth=1)

    y_lowlim = ydata-yerr
    p_guess_low = [max(y_lowlim), 1, min(y_lowlim)]
    popt_low, pcov_low = optimize.curve_fit(exponential_decay, xdata, y_lowlim, p0=p_guess_low, maxfev = 10**7, method ="trf") #, bounds = bounds)
    yfit_low = exponential_decay(xfit,*popt_low)
    plt.plot(xfit, yfit_low, color='grey', linestyle='dashed', linewidth=1)

    #calculate FCCD of data - invert eq
    FCCD_data_1173 = (1/b)*np.log(a/(C_1173_data-c))

    #calculate error on FCCD
    a_up, b_up, c_up = popt_up[0], popt_up[1], popt_up[2]
    FCCD_data_err_up_1173 = (1/b_up)*np.log(a_up/(C_1173_data-c_up))-FCCD_data_1173
    a_low, b_low, c_low = popt_low[0], popt_low[1], popt_low[2]
    FCCD_data_err_low_1173 = FCCD_data_1173 - (1/b_low)*np.log(a_low/(C_1173_data-c_low))
    #FCCD_data_err = np.sqrt((1/(a**2*b**4*(c-O_Ba133_data)**2))*(a**2*(b_err**2*(c-O_Ba133_data)**2)*(np.log(-a/(c-O_Ba133_data))**2) + b**2*(c_err**2+O_Ba133_err_data**2) + a_err**2*b**2*(c-O_Ba133_data)**2)) #wolfram alpha

    print('FCCD of data extrapolated: '+str(FCCD_data_1173) +" + "+ str(FCCD_data_err_up_1173) +" - "+str(FCCD_data_err_low_1173))

    props = dict(boxstyle='round', alpha=0.5)
    info_str = '\n'.join((r'$a=%.0f \pm %.0f$' % (a, np.sqrt(pcov[0][0])), r'$b=%.3f \pm %.3f$' % (b, np.sqrt(pcov[1][1])), r'$c=%.0f \pm %.0f$' % (c, np.sqrt(pcov[2][2])), r'$\chi^2/dof=%.2f/%.0f$'%(chi_sq, dof), r'FCCD_data=$%.2f^{+%.2f}_{-%.2f}$ mm' % (FCCD_data_1173, FCCD_data_err_up_1173, FCCD_data_err_low_1173)))
    plt.text(0.025, 0.275, info_str, transform=ax.transAxes, fontsize=9,verticalalignment='top', bbox=props) #ax.text..ax.tra

    #plot data line
    plt.hlines(C_1173_data, 0, FCCD_list[-1], colors="orange", label = 'data')
    plt.plot(xfit, [C_1173_data+C_1173_data_err]*(len(xfit)), color = 'grey', linestyle = 'dashed', linewidth = '1.0')
    plt.plot(xfit, [C_1173_data-C_1173_data_err]*(len(xfit)), color = 'grey', linestyle = 'dashed', linewidth = '1.0')

    plt.vlines(FCCD_data_1173, 0, C_1173_data, colors='orange', linestyles='dashed')
    plt.vlines(FCCD_data_1173+FCCD_data_err_up_1173, 0, C_1173_data, colors='grey', linestyles='dashed', linewidths=1)
    plt.vlines(FCCD_data_1173-FCCD_data_err_low_1173, 0, C_1173_data, colors='grey', linestyles='dashed', linewidths=1)

    plt.ylabel(r'$C_{1173 keV}$')
    plt.xlabel("FCCD [mm]")
    plt.xlim(0,FCCD_list[-1])
    plt.ylim(60000,None)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    #plt.title(detector)
    plt.legend(loc="upper right", fontsize=8)

    if cuts == False:
        plt.savefig(dir+"/FCCD/plots/FCCD_C1173_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+"_onlystat.pdf")
    else:
        plt.savefig(dir+"/FCCD/plots/FCCD_C1173_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+"_cuts.png")


    #========PLOT 2=============

    print("using Co60 observable 1: C_1332 keV")

    #plot and fit exp decay
    xdata, ydata = np.array(FCCD_list), np.array(C_1332_list)
    yerr = C_1332_err_pct_list*ydata/100
    #yerr = C_1332_err_pct_list #get absolute error, not percentage
    aguess = max(ydata)
    bguess = 1
    cguess = min(ydata)
    p_guess = [aguess,bguess,cguess]
    popt, pcov = optimize.curve_fit(exponential_decay, xdata, ydata, p0=p_guess, sigma = yerr, maxfev = 10**7, method ="trf") #, bounds = bounds)
    a,b,c = popt[0],popt[1],popt[2]
    a_err, b_err, c_err = np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1]), np.sqrt(pcov[2][2])
    chi_sq, p_value, residuals, dof = chi_sq_calc(xdata, ydata, yerr, exponential_decay, popt)

    fig, ax = plt.subplots()
    plt.errorbar(xdata, ydata, xerr=0, yerr =yerr, label = "simulations", elinewidth = 1, fmt='x', ms = 3.0, mew = 3.0)
    xfit = np.linspace(min(xdata), max(xdata), 1000)
    yfit = exponential_decay(xfit,*popt)
    plt.plot(xfit, yfit, "g", label = "fit: a*exp(-bx)+c")

    #fit exp decay of error bars:
    y_uplim = ydata+yerr
    p_guess_up = [max(y_uplim), 1, min(y_uplim)]
    popt_up, pcov_up = optimize.curve_fit(exponential_decay, xdata, y_uplim, p0=p_guess_up, maxfev = 10**7, method ="trf") #, bounds = bounds)
    yfit_up = exponential_decay(xfit,*popt_up)
    plt.plot(xfit, yfit_up, color='grey', linestyle='dashed', linewidth=1)

    y_lowlim = ydata-yerr
    p_guess_low = [max(y_lowlim), 1, min(y_lowlim)]
    popt_low, pcov_low = optimize.curve_fit(exponential_decay, xdata, y_lowlim, p0=p_guess_low, maxfev = 10**7, method ="trf") #, bounds = bounds)
    yfit_low = exponential_decay(xfit,*popt_low)
    plt.plot(xfit, yfit_low, color='grey', linestyle='dashed', linewidth=1)

    #calculate FCCD of data - invert eq
    FCCD_data_1332 = (1/b)*np.log(a/(C_1332_data-c))

    #calculate error on FCCD
    a_up, b_up, c_up = popt_up[0], popt_up[1], popt_up[2]
    FCCD_data_err_up_1332 = (1/b_up)*np.log(a_up/(C_1332_data-c_up))-FCCD_data_1332
    a_low, b_low, c_low = popt_low[0], popt_low[1], popt_low[2]
    FCCD_data_err_low_1332 = FCCD_data_1332 - (1/b_low)*np.log(a_low/(C_1332_data-c_low))

    print('FCCD of data extrapolated: '+str(FCCD_data_1332) +" + "+ str(FCCD_data_err_up_1332) +" - "+str(FCCD_data_err_low_1332))

    props = dict(boxstyle='round', alpha=0.5)
    info_str = '\n'.join((r'$a=%.0f \pm %.0f$' % (a, np.sqrt(pcov[0][0])), r'$b=%.3f \pm %.3f$' % (b, np.sqrt(pcov[1][1])), r'$c=%.0f \pm %.0f$' % (c, np.sqrt(pcov[2][2])), r'$\chi^2/dof=%.2f/%.0f$'%(chi_sq, dof), r'FCCD_data=$%.3f^{+%.3f}_{-%.3f}$ mm' % (FCCD_data_1332, FCCD_data_err_up_1332, FCCD_data_err_low_1332)))
    plt.text(0.025, 0.275, info_str, transform=ax.transAxes, fontsize=9,verticalalignment='top', bbox=props) #ax.text..ax.tra

    #plot data line
    plt.hlines(C_1332_data, 0, FCCD_list[-1], colors="orange", label = 'data')
    plt.plot(xfit, [C_1332_data+C_1332_data_err]*(len(xfit)), color = 'grey', linestyle = 'dashed', linewidth = '1.0')
    plt.plot(xfit, [C_1332_data-C_1332_data_err]*(len(xfit)), color = 'grey', linestyle = 'dashed', linewidth = '1.0')

    plt.vlines(FCCD_data_1332, 0, C_1332_data, colors='orange', linestyles='dashed')
    plt.vlines(FCCD_data_1332+FCCD_data_err_up_1332, 0, C_1332_data, colors='grey', linestyles='dashed', linewidths=1)
    plt.vlines(FCCD_data_1332-FCCD_data_err_low_1332, 0, C_1332_data, colors='grey', linestyles='dashed', linewidths=1)

    plt.ylabel(r'$C_{1332 keV}$')
    plt.xlabel("FCCD (mm)")
    plt.xlim(0,FCCD_list[-1])
    plt.ylim(0,None)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    #plt.title(detector)
    plt.legend(loc="upper left", fontsize=8)

    # plt.show()

    if cuts == False:
        plt.savefig(dir+"/FCCD/plots/FCCD_C1332_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+"_onlystat.png")
    else:
        plt.savefig(dir+"/FCCD/plots/FCCD_C1332_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+"_cuts.png")

    FCCD_av = (FCCD_data_1173+FCCD_data_1332)/2
    FCCD_av_err_up = 0.5*np.sqrt(FCCD_data_err_up_1173**2 + FCCD_data_err_up_1332**2)
    FCCD_av_err_low = 0.5*np.sqrt(FCCD_data_err_low_1173**2 + FCCD_data_err_low_1332**2)
    print("FCCD average: ", FCCD_av, " + ", FCCD_av_err_up, " - ", FCCD_av_err_low)

    #Save interpolated fccd for data to a json file
    FCCD_data_dict = {"FCCD_1173": FCCD_data_1173,"FCCD_err_up_1173": FCCD_data_err_up_1173, "FCCD_err_low_1173": FCCD_data_err_low_1173, "FCCD_1332": FCCD_data_1332,"FCCD_err_up_1332": FCCD_data_err_up_1332, "FCCD_err_low_1332": FCCD_data_err_low_1332, "FCCD_av": FCCD_av, "FCCD_av_err_up": FCCD_av_err_up, "FCCD_av_err_low": FCCD_av_err_low}

    if cuts == False:
        with open(dir+"/FCCD/FCCD_data_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+"_onlystat.json", "w") as outfile:
            json.dump(FCCD_data_dict, outfile, indent=4)
    else:
        if cuts_sigma ==4:
            with open(dir+"/FCCD/FCCD_data_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+"_cuts.json", "w") as outfile:
                json.dump(FCCD_data_dict, outfile, indent=4)
        else:
            with open(dir+"/FCCD/FCCD_data_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+"_cuts_"+str(cuts_sigma)+"sigma.json", "w") as outfile:
                json.dump(FCCD_data_dict, outfile, indent=4)

    print("done")

def exponential_decay(x, a, b ,c):
    f = a*np.exp(-b*x) + c
    return f

def uncertainty(O_C60, O_C60_err):

    #values from Bjoern's thesis - Barium source
    #all percentages
    gamma_line=0.03
    geant4=4.0
    source_distance=1.2
    source_thickness=0.02
    source_material=0.01
    source_activity=1.0
    detector_distance=1.2
    detector_dimensions=3
    endcap_thickness=0.15
    detector_cup_thickness=0.06
    detector_cup_material=0.03

    #compute statistical error
    MC_statistics = O_C60_err/O_C60*100

    #sum squared of all the contributions
    tot_error=np.sqrt(gamma_line**2+geant4**2+
        source_distance**2+source_thickness**2+source_material**2+source_activity**2+
        detector_distance**2+detector_dimensions**2+
        endcap_thickness**2+detector_cup_thickness**2+detector_cup_material**2+MC_statistics**2)

    #return tot_error #NB: tot_error is a percentage error
    return MC_statistics



if __name__ == "__main__":
    main()
