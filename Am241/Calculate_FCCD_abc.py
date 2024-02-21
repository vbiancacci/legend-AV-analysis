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

from GammaLine_Counting_Am241_HS6 import chi_sq_calc

#script to determine the FCCD of a detector using the count ratio observable, ignoring TL for now

def main():

    if(len(sys.argv) != 10):
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

    #initialise directories to save
#    if not os.path.exists(dir+"/FCCD/"+source+"plots/"):
#        os.makedirs(dir+"/FCCD/"+source+"/plots/")

    print("start...")

    #Get O_ba133 for each FCCD
    FCCD_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0] #mm
    O_Am241_list = []
    O_Am241_tot_err_list = []
    O_Am241_corr_err_list = []
    O_Am241_uncorr_err_list = []
    DLF = 1.0 #considering 0 TL

    for FCCD in FCCD_list:

        #Get count ratio for simulations
        with open(dir+"/PeakCounts/"+detector+"/"+source+"/new/PeakCounts_sim_"+MC_id+"_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+'mm_DLF'+str(DLF)+'_fracFCCDbore'+frac_FCCDbore+'.json') as json_file:
            PeakCounts = json.load(json_file)
            C_60 = PeakCounts['C_60']
            sigma_60 = PeakCounts['C_60_err']
            C_99_103 = PeakCounts['C_99_103']
            sigma_99_103 = PeakCounts['C_99_103_err']
            O_Am241 = PeakCounts['O_Am241']
            O_Am241_err = PeakCounts ['O_Am241_err']
            O_Am241_list.append(O_Am241)

            #get errors:
            O_Am241_tot_err, O_Am241_corr_err, O_Am241_uncorr_err = uncertainty(O_Am241, O_Am241_err)
            O_Am241_tot_err_list.append(O_Am241_tot_err)
            O_Am241_corr_err_list.append(O_Am241_corr_err)
            O_Am241_uncorr_err_list.append(O_Am241_uncorr_err)

    #Get count ratio for data
    if cuts == False:
        with open(dir+"/PeakCounts/"+detector+"/"+source+"/new/PeakCounts_data_"+detector+"_"+energy_filter+"_run"+str(run)+".json") as json_file:
            PeakCounts = json.load(json_file)
            C_60 = PeakCounts['C_60']
            sigma_60 = PeakCounts['C_60_err']
            C_99_103 = PeakCounts['C_99_103']
            sigma_99_103 = PeakCounts['C_99_103_err']
            O_Am241_data = PeakCounts['O_Am241']
            O_Am241_data_err = PeakCounts ['O_Am241_err']

    else:
        cuts_sigma = 4 #default =4, change by hand here if interested
        print("data cuts sigma: ", str(cuts_sigma))
        if cuts_sigma == 4:
            with open(dir+"/PeakCounts/"+detector+"/"+source+"/new/PeakCounts_data_"+detector+"_cuts_"+energy_filter+"_run"+str(run)+".json") as json_file:
                PeakCounts = json.load(json_file)
                C_60 = PeakCounts['C_60']
                sigma_60 = PeakCounts['C_60_err']
                C_99_103 = PeakCounts['C_99_103']
                sigma_99_103 = PeakCounts['C_99_103_err']
                O_Am241_data = PeakCounts['O_Am241']
                O_Am241_data_err = PeakCounts ['O_Am241_err']

        else:
            with open(dir+"/PeakCounts/"+detector+"/"+source+"/PeakCounts_data_"+detector+"_cuts_"+energy_filter+"_run"+str(run)+"_"+str(sigma_cuts)+"sigma.json") as json_file:
                PeakCounts = json.load(json_file)
                C_60 = PeakCounts['C_60']
                sigma_60 = PeakCounts['C_60_err']
                C_99_103 = PeakCounts['C_99_103']
                sigma_99_103 = PeakCounts['C_99_103_err']
                O_Am241_data = PeakCounts['O_Am241']
                O_Am241_data_err = PeakCounts ['O_Am241_err']


    #plot and fit exp decay
    xdata, ydata = np.array(FCCD_list), np.array(O_Am241_list)
    y_err=O_Am241_tot_err_list*ydata/100
    y_corr_err = O_Am241_corr_err_list*ydata/100 #get absolute error, not percentage
    y_uncorr_err= O_Am241_uncorr_err_list*ydata/100
    
    aguess = max(ydata)
    bguess = 1
    cguess = min(ydata)
    p_guess = [aguess,bguess, cguess]

    popt, pcov = optimize.curve_fit(exponential_decay, xdata, ydata, p0=p_guess, sigma = y_err, maxfev = 10**7, method ="trf") #, bounds = bounds)
    a,b,c = popt[0],popt[1],popt[2]
    print(a,b)
    #a_err, b_err = np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1])#,np.sqrt(pcov[2][2])
    chi_sq, p_value, residuals, dof = chi_sq_calc(xdata, ydata, y_err, exponential_decay, popt)

    fig, ax = plt.subplots()
    plt.errorbar(xdata, ydata, xerr=0, yerr =y_err, label = "simulations", elinewidth = 1, fmt='x', ms = 3.0, mew = 3.0)
    xfit = np.linspace(min(xdata), max(xdata), 1000)
    #yfit = exponential_decay(xfit,*popt)
    yfit = exponential_decay(xfit,a,b,c)
    plt.plot(xfit, yfit, "g", label = "fit: a*exp(-bx)")

    #calculate FCCD of data - invert eq
    FCCD_data = (1/b)*np.log(a/(O_Am241_data-c))
    print("FCCD_data ", FCCD_data)

    #fit exp decay of total error bars:
    y_uplim = ydata+y_err
    p_guess_up = [max(y_uplim), 1,min(y_uplim)]
    popt_up, pcov_up= optimize.curve_fit(exponential_decay, xdata, y_uplim, p0=p_guess_up, maxfev = 10**7, method ="trf") #, bounds = bounds)
    yfit_up = exponential_decay(xfit,*popt_up)
    #plt.plot(xfit, yfit_up, color='grey', linestyle='dashed', linewidth=1)

    y_lowlim = ydata-y_err
    p_guess_low = [max(y_lowlim), 1, min(y_lowlim)]
    popt_low, pcov_low = optimize.curve_fit(exponential_decay, xdata, y_lowlim, p0=p_guess_low, maxfev = 10**7, method ="trf") #, bounds = bounds)
    yfit_low = exponential_decay(xfit,*popt_low)
    #plt.plot(xfit, yfit_low, color='grey', linestyle='dashed', linewidth=1)

    a_up, b_up,c_up = popt_up[0], popt_up[1],  popt_up[2]
    a_low, b_low,c_low = popt_low[0], popt_low[1],  popt_low[2]


    #fit exp decay of correlated error bars:
    y_uplim = ydata+y_corr_err
    p_guess_up = [max(y_uplim), 1, min(y_uplim) ]
    popt_up_corr, pcov_up_corr = optimize.curve_fit(exponential_decay, xdata, y_uplim, p0=p_guess_up, maxfev = 10**7, method ="trf") #, bounds = bounds)
    yfit_up = exponential_decay(xfit,*popt_up_corr)
    #plt.plot(xfit, yfit_up, color='grey', linestyle='dashed', linewidth=1)

    y_lowlim = ydata-y_corr_err
    p_guess_low = [max(y_lowlim), 1, min(y_lowlim)]
    popt_low_corr, pcov_low_corr = optimize.curve_fit(exponential_decay, xdata, y_lowlim, p0=p_guess_low, maxfev = 10**7, method ="trf") #, bounds = bounds)
    yfit_low = exponential_decay(xfit,*popt_low_corr)
    #plt.plot(xfit, yfit_low, color='grey', linestyle='dashed', linewidth=1)

    a_up_corr, b_up_corr, c_up_corr = popt_up_corr[0], popt_up_corr[1], popt_up_corr[2]
    a_low_corr, b_low_corr, c_low_corr = popt_low_corr[0], popt_low_corr[1], popt_low_corr[2]


    #fit exp decay of uncorrelated error bars:
    y_uplim = ydata+y_uncorr_err
    p_guess_up = [max(y_uplim), 1, min(y_uplim)]
    popt_up_uncorr, pcov_up_uncorr = optimize.curve_fit(exponential_decay, xdata, y_uplim, p0=p_guess_up, maxfev = 10**7, method ="trf") #, bounds = bounds)
    yfit_up = exponential_decay(xfit,*popt_up_uncorr)
    plt.plot(xfit, yfit_up, color='grey', linestyle='dashed', linewidth=1)

    y_lowlim = ydata-y_uncorr_err
    p_guess_low = [max(y_lowlim), 1, min(y_lowlim)]
    popt_low_uncorr, pcov_low_uncorr = optimize.curve_fit(exponential_decay, xdata, y_lowlim, p0=p_guess_low, maxfev = 10**7, method ="trf") #, bounds = bounds)
    yfit_low = exponential_decay(xfit,*popt_low_uncorr)
    plt.plot(xfit, yfit_low, color='grey', linestyle='dashed', linewidth=1)

    a_up_uncorr, b_up_uncorr = popt_up_uncorr[0], popt_up_uncorr[1]
    a_low_uncorr, b_low_uncorr = popt_low_uncorr[0], popt_low_uncorr[1]

    #calculate total error on FCCD
    FCCD_err_up = (1/b_up)*np.log(a_up/(O_Am241_data-O_Am241_data_err))-FCCD_data
    FCCD_err_low = FCCD_data - (1/b_low)*np.log(a_low/(O_Am241_data+O_Am241_data_err))
    print('total error:  + '+ str(FCCD_err_up) +" - "+str(FCCD_err_low))


    #calculate uncorrelated error on FCCD
    FCCD_uncorr_up = (1/b_up_uncorr)*np.log(a_up_uncorr/(O_Am241_data-O_Am241_data_err))-FCCD_data
    FCCD_uncorr_low = FCCD_data - (1/b_low_uncorr)*np.log(a_low_uncorr/(O_Am241_data+O_Am241_data_err))
    print('uncorrelated error:  + '+ str(FCCD_uncorr_up) +" - "+str(FCCD_uncorr_low))

    #calculate correlated error on FCCD
    FCCD_corr_up = (1/b_up_corr)*np.log(a_up_corr/(O_Am241_data))-FCCD_data
    FCCD_corr_low = FCCD_data - (1/b_low_corr)*np.log(a_low_corr/(O_Am241_data))
    print('correlated error:  + '+ str(FCCD_corr_up) +" - "+str(FCCD_corr_low))


    props = dict(boxstyle='round', alpha=0.5)
    info_str = '\n'.join((r'$a=%.3f \pm %.3f$' % (a, np.sqrt(pcov[0][0])), r'$b=%.3f \pm %.3f$' % (b, np.sqrt(pcov[1][1])), r'$\chi^2/dof=%.2f/%.0f$'%(chi_sq, dof), r'FCCD_data=$%.2f^{+%.2f}_{-%.2f}$ mm' % (FCCD_data, FCCD_err_up, FCCD_err_low)))
    plt.text(0.625, 0.775, info_str, transform=ax.transAxes, fontsize=9,verticalalignment='top', bbox=props) #ax.text..ax.tra

    #plot data line
    plt.hlines(O_Am241_data, 0, FCCD_list[-1], colors="orange", label = 'data')
    plt.plot(xfit, [O_Am241_data+O_Am241_data_err]*(len(xfit)), label = 'bounds', color = 'grey', linestyle = 'dashed', linewidth = '1.0')
    plt.plot(xfit, [O_Am241_data-O_Am241_data_err]*(len(xfit)), color = 'grey', linestyle = 'dashed', linewidth = '1.0')


    plt.vlines(FCCD_data, 0, O_Am241_data, colors='orange', linestyles='dashed')
    plt.vlines(FCCD_data+FCCD_err_up, 0, O_Am241_data-O_Am241_data_err, colors='grey', linestyles='dashed', linewidths=1)
    plt.vlines(FCCD_data-FCCD_err_low, 0, O_Am241_data+O_Am241_data_err, colors='grey', linestyles='dashed', linewidths=1)
    #plt.vlines(FCCD_data+FCCD_data_uncerr_up, 0, O_Am241_data-O_Am241_data_err, colors='violet', linestyles='dashed', linewidths=1)
    #plt.vlines(FCCD_data-FCCD_data_uncerr_low, 0, O_Am241_data+O_Am241_data_err, colors='violet', linestyles='dashed', linewidths=1)



    plt.ylabel(r'$O_{am\_HS1}=\frac{C_{60keV}}{C_{99keV}+C_{103keV}}$')
    plt.xlabel("FCCD [mm]")
    plt.xlim(0,FCCD_list[-1])
    plt.ylim(0,1000) #1000
    #plt.title(detector)
    plt.legend(loc="upper right", fontsize=8)
    #plt.show()


    if cuts == False:
        plt.savefig(dir+"/FCCD/"+source+"/plots/FCCD_OAm241_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+".pdf")
    else:
        plt.savefig(dir+"/FCCD/"+source+"/plots/FCCD_OAm241_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+"_cuts_corr_uncorr.pdf")


    #Save interpolated fccd for data to a json file
    FCCD_data_dict = {
        "FCCD": FCCD_data,
        "FCCD_err_up": FCCD_err_up,
        "FCCD_err_low": FCCD_err_low,
        "FCCD_uncorr_err_up": FCCD_uncorr_up,
        "FCCD_uncorr_err_low": FCCD_uncorr_low,
        "FCCD_corr_err_up": FCCD_corr_up,
        "FCCD_corr_err_low": FCCD_corr_low,
        "O_Am241_data": O_Am241_data,
        "O_Am241_data_err": O_Am241_data_err,
        "a": a,
        "a_corr_err": (a_up_corr-a_low_corr)/2,
        "a_uncorr_err": (a_up_uncorr-a_low_uncorr)/2,
        "b": b,
        "b_corr_err": (b_up_corr-b_low_corr)/2,
        "b_uncorr_err": (b_up_uncorr-b_low_uncorr)/2
    }

    if cuts == False:
        with open(dir+"/FCCD/"+source+"/FCCD_data_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+".json", "w") as outfile:
            json.dump(FCCD_data_dict, outfile, indent=4)
            print("json file saved")
    else:
        if cuts_sigma ==4:
            with open(dir+"/FCCD/"+source+"/FCCD_data_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+"_cuts_corr_uncorr.json", "w") as outfile:
                json.dump(FCCD_data_dict, outfile, indent=4)
        else:
            with open(dir+"/FCCD/"+source+"/no_source_calibration/FCCD_data_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+"_cuts_"+str(cuts_sigma)+"sigma.json", "w") as outfile:
                json.dump(FCCD_data_dict, outfile, indent=4)

    print("done")

def exponential_decay(x, a, b,c):
    f = a*np.exp(-b*x)+c
    return f

def uncertainty(O_Am241, O_Am241_err):

    #values from my thesis - Am source
    #all percentages
    gamma_line=1.81
    geant4=2.
    source_thickness=0.01
    source_material=0.01
    endcap_thickness=0.37
    detector_cup_thickness=0.03
    detector_cup_material=0.01

    MC_statistics = O_Am241_err/O_Am241*100

    #sum squared of all the contributions
    tot_error=np.sqrt(gamma_line**2+geant4**2+source_thickness**2+source_material**2+endcap_thickness**2+detector_cup_thickness**2+detector_cup_material**2+MC_statistics**2)

     #correlated error
    corr_error=np.sqrt(gamma_line**2+geant4**2+source_thickness**2+source_material**2+endcap_thickness**2+detector_cup_thickness**2+detector_cup_material**2)


    #uncorrelated error
    uncorr_error=MC_statistics
    
   
    return tot_error, corr_error, uncorr_error #NB: tot_error is a percentage error


if __name__ == "__main__":
    main()
