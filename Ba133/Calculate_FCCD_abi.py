from ast import Invert
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import argparse
import os
#from pyrsistent import b
from scipy import optimize
from scipy import stats

from GammaLine_Counting_Ba133 import chi_sq_calc

#script to determine the FCCD of a detector using the count ratio observable, ignoring TL for now

def main():

    if(len(sys.argv) != 9):
        print('Example usage: python Ba133_Calibration.py <detector> <MC_id> <smear> <TL_model> <frac_FCCDbore> <energy_filter> <cuts> <run>')
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

    #Get O_ba133 for each FCCD
    FCCD_list = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 3.0] #mm ICPCs
    #FCCD_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0] 
    # FCCD_list = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0] #mm BEGes
    
    O_Ba133_list = []
    O_Ba133_tot_err_pct_list = [] 
    O_Ba133_corr_err_pct_list = [] #corr is actually all systematic errors
    O_Ba133_uncorr_err_pct_list = [] #uncorr is actually only statistical error

    DLF = 1.0 #considering 0 TL

    for FCCD in FCCD_list:

        #Get count ratio for simulations
        with open(dir+"/PeakCounts/"+detector+"/PeakCounts_sim_"+MC_id+"_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+'mm_DLF'+str(DLF)+'_fracFCCDbore'+frac_FCCDbore+'.json') as json_file:
            PeakCounts = json.load(json_file)
            C_356 = PeakCounts['C_356']
            C_79 = PeakCounts['C_79']
            C_81 = PeakCounts['C_81']
            O_Ba133 = PeakCounts['O_Ba133']
            O_Ba133_err = PeakCounts ['O_Ba133_err'] #stat fit error on MC
            O_Ba133_list.append(O_Ba133)

            #get errors:
            O_Ba133_tot_err_pct, O_Ba133_corr_err_pct, O_Ba133_uncorr_err_pct = error_O_Ba133_MC(O_Ba133, O_Ba133_err) #stat and syst error on MC
            O_Ba133_tot_err_pct_list.append(O_Ba133_tot_err_pct)
            O_Ba133_corr_err_pct_list.append(O_Ba133_corr_err_pct)
            O_Ba133_uncorr_err_pct_list.append(O_Ba133_uncorr_err_pct)

            
    #Get count ratio for data
    if cuts == False:
        with open(dir+"/PeakCounts/"+detector+"/PeakCounts_data_"+detector+"_"+energy_filter+"_run"+str(run)+".json") as json_file:
            PeakCounts = json.load(json_file)
            C_356 = PeakCounts['C_356']
            C_79 = PeakCounts['C_79']
            C_81 = PeakCounts['C_81']
            O_Ba133_data = PeakCounts['O_Ba133']
            O_Ba133_data_err = PeakCounts ['O_Ba133_err'] #stat error on data
    else:
        cuts_sigma = 4 #default =4, change by hand here if interested
        print("data cuts sigma: ", str(cuts_sigma))
        if cuts_sigma == 4:
            with open(dir+"/PeakCounts/"+detector+"/PeakCounts_data_"+detector+"_cuts_"+energy_filter+"_run"+str(run)+".json") as json_file:
                PeakCounts = json.load(json_file)
                C_356 = PeakCounts['C_356']
                C_79 = PeakCounts['C_79']
                C_81 = PeakCounts['C_81']
                O_Ba133_data = PeakCounts['O_Ba133']
                O_Ba133_data_err = PeakCounts ['O_Ba133_err']
        else:
            with open(dir+"/PeakCounts/"+detector+"/PeakCounts_data_"+detector+"_cuts_"+energy_filter+"_run"+str(run)+"_"+str(sigma_cuts)+"sigma.json") as json_file:
                PeakCounts = json.load(json_file)
                C_356 = PeakCounts['C_356']
                C_79 = PeakCounts['C_79']
                C_81 = PeakCounts['C_81']
                O_Ba133_data = PeakCounts['O_Ba133']
                O_Ba133_data_err = PeakCounts ['O_Ba133_err']

    #========= PLOTTING ===========

    plot_colors = {"MC":"black", "MC_fit": "black", "data": "orange", "data_err_stat": "green", "MC_err_stat": "green", "MC_err_syst": "pink", "MC_err_total": "red", "FCCD": "orange", "FCCD_err_total": "blue", "FCCD_err_MCstatsyst": "red", "FCCD_err_MCstat": "purple", "FCCD_err_MCsyst": "pink", "FCCD_err_datastat": "green", "FCCD_err_statMCstatdata": "grey"}

    #plot and fit exp decay
    print("fitting exp decay")
    xdata, ydata = np.array(FCCD_list), np.array(O_Ba133_list)
    yerr = O_Ba133_tot_err_pct_list*ydata/100 #absolute total error
    aguess = max(ydata)
    bguess = 1
    cguess = min(ydata)
    p_guess = [aguess,bguess,cguess]
    popt, pcov = optimize.curve_fit(exponential_decay, xdata, ydata, p0=p_guess, sigma = yerr,absolute_sigma=False, maxfev = 10**7, method ="trf") #, bounds = bounds)
    a,b,c = popt[0],popt[1],popt[2]
    a_err, b_err, c_err = np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1]), np.sqrt(pcov[2][2])
    chi_sq, p_value, residuals, dof = chi_sq_calc(xdata, ydata, yerr, exponential_decay, popt)

    fig, ax = plt.subplots()
    plt.errorbar(xdata, ydata, xerr=0, yerr =yerr, label = "MC", color= plot_colors["MC"], elinewidth = 1, fmt='x', ms = 3.0, mew = 3.0)
    xfit = np.linspace(min(xdata), max(xdata), 1000)
    yfit = exponential_decay(xfit,*popt)
    plt.plot(xfit, yfit, color=plot_colors["MC_fit"])


    #=====fit exp decay of error bars========

    # MC stat and syst
    print("fitting exp decay of total MC O_Ba133 errors")
    y_uplim = ydata+yerr
    p_guess_up = [max(y_uplim), 1, min(y_uplim)]
    popt_up, pcov_up = optimize.curve_fit(exponential_decay, xdata, y_uplim, p0=p_guess_up, maxfev = 10**7, method ="trf") #, bounds = bounds)
    yfit_up = exponential_decay(xfit,*popt_up)
    plt.plot(xfit, yfit_up, color=plot_colors["MC_err_total"], linestyle='dashed', linewidth=1, label="MC err (stat/corr + syst/uncorr)")

    y_lowlim = ydata-yerr
    p_guess_low = [max(y_lowlim), 1, min(y_lowlim)]
    popt_low, pcov_low = optimize.curve_fit(exponential_decay, xdata, y_lowlim, p0=p_guess_low, maxfev = 10**7, method ="trf") #, bounds = bounds)
    yfit_low = exponential_decay(xfit,*popt_low)
    plt.plot(xfit, yfit_low, color=plot_colors["MC_err_total"], linestyle='dashed', linewidth=1)

    a_up, b_up, c_up = popt_up[0], popt_up[1], popt_up[2]
    a_low, b_low, c_low = popt_low[0], popt_low[1], popt_low[2] 


    # MC corr
    print("fitting exp decay of MC stat") 
    yerr_corr = np.array(O_Ba133_corr_err_pct_list)*ydata/100
    y_uplim_corr = ydata+yerr_corr
    p_guess_up = [max(y_uplim), 1, min(y_uplim)]
    popt_up_corr, pcov_up_corr = optimize.curve_fit(exponential_decay, xdata, y_uplim_corr, p0=p_guess_up, maxfev = 10**7, method ="trf") #, bounds = bounds)
    yfit_up_corr = exponential_decay(xfit,*popt_up_corr)
    # plt.plot(xfit, yfit_up_corr, color=plot_colors["MC_err_corr"], linestyle='dashed', linewidth=1, label="MC stat")

    y_lowlim_corr = ydata-yerr_corr
    p_guess_low = [max(y_lowlim), 1, min(y_lowlim)]
    popt_low_corr, pcov_low_corr = optimize.curve_fit(exponential_decay, xdata, y_lowlim_corr, p0=p_guess_low, maxfev = 10**7, method ="trf") #, bounds = bounds)
    yfit_low_corr = exponential_decay(xfit,*popt_low_corr)
    # plt.plot(xfit, yfit_low_corr, color=plot_colors["MC_err_corr"],linestyle='dashed', linewidth=1)

    a_up_corr, b_up_corr, c_up_corr = popt_up_corr[0], popt_up_corr[1], popt_up_corr[2]
    a_low_corr, b_low_corr, c_low_corr = popt_low_corr[0], popt_low_corr[1], popt_low_corr[2] 

    # MC uncorr
    print("fitting exp decay of MC uncorr err")
    yerr_uncorr = O_Ba133_uncorr_err_pct_list*ydata/100
    y_uplim_uncorr = ydata+yerr_uncorr
    p_guess_up = [max(y_uplim), 1, min(y_uplim)]
    popt_up_uncorr, pcov_up_uncorr = optimize.curve_fit(exponential_decay, xdata, y_uplim_uncorr, p0=p_guess_up, maxfev = 10**7, method ="trf") #, bounds = bounds)
    yfit_up_uncorr = exponential_decay(xfit,*popt_up_uncorr)
    # plt.plot(xfit, yfit_up_uncorr, color=plot_colors["MC_err_uncorr"], linestyle='dashed', linewidth=1, label="MC uncorr")

    y_lowlim_uncorr = ydata-yerr_uncorr
    p_guess_low = [max(y_lowlim), 1, min(y_lowlim)]
    popt_low_uncorr, pcov_low_uncorr = optimize.curve_fit(exponential_decay, xdata, y_lowlim_uncorr, p0=p_guess_low, maxfev = 10**7, method ="trf") #, bounds = bounds)
    yfit_low_uncorr = exponential_decay(xfit,*popt_low_uncorr)
    # plt.plot(xfit, yfit_low_uncorr, color=plot_colors["MC_err_uncorr"],linestyle='dashed', linewidth=1)

    a_up_uncorr, b_up_uncorr, c_up_uncorr = popt_up_uncorr[0], popt_up_uncorr[1], popt_up_uncorr[2]
    a_low_uncorr, b_low_uncorr, c_low_uncorr = popt_low_uncorr[0], popt_low_uncorr[1], popt_low_uncorr[2] 


    #=========Compute FCCD and all errors ===============

    #calculate FCCD of data - invert eq
    FCCD_data = invert_exponential(O_Ba133_data,a,b,c)
    print('FCCD of data extrapolated: ')
    print(str(FCCD_data))

    #TOTAL ERROR = (statistical (uncorr) error on data, statistical (uncorr) and systematic (corr) on MC)
    FCCD_err_total_up = invert_exponential((O_Ba133_data-O_Ba133_data_err),a_up, b_up,c_up) - FCCD_data
    FCCD_err_total_low = FCCD_data - invert_exponential((O_Ba133_data+O_Ba133_data_err),a_low, b_low,c_low)
    print('Total error:  + '+ str(FCCD_err_total_up) +" - "+str(FCCD_err_total_low))
    
    # # uncorr/statistical error on data only
    # FCCD_err_statdata_up = FCCD_data - invert_exponential(O_Ba133_data+O_Ba133_data_err, a,b,c)
    # FCCD_err_statdata_low = invert_exponential(O_Ba133_data-O_Ba133_data_err, a,b,c) - FCCD_data
    # print('statistical/uncorr error on data:')  
    # print('+ '+str(FCCD_err_statdata_up)+" - "+str(FCCD_err_statdata_low))

    # # total error on MC (stat/uncorr and syst/corr) , none on data
    # FCCD_err_statsystMC_up = invert_exponential(O_Ba133_data,a_up,b_up,c_up) -FCCD_data
    # FCCD_err_statsystMC_low = FCCD_data - invert_exponential(O_Ba133_data,a_low,b_low,c_low)
    # print('total error (stat/uncorr and syst/corr) on MC, but not data:')
    # print("+ "+str(FCCD_err_statsystMC_up)+" - "+str(FCCD_err_statsystMC_low))

    # #stat/uncorr MC only
    # FCCD_err_statMC_up = invert_exponential(O_Ba133_data, a_up_uncorr, b_up_uncorr, c_up_uncorr) - FCCD_data
    # FCCD_err_statMC_low = FCCD_data - invert_exponential(O_Ba133_data, a_low_uncorr, b_low_uncorr, c_low_uncorr)
    # print('statistical/uncorr error on MC only')
    # print("+ "+str(FCCD_err_statMC_up) +" - "+str(FCCD_err_statMC_low))

    #TOTAL CORR ERROR == syst/corr MC only
    FCCD_err_systMC_up = invert_exponential(O_Ba133_data, a_up_corr, b_up_corr, c_up_corr) -FCCD_data
    FCCD_err_systMC_low = FCCD_data - invert_exponential(O_Ba133_data, a_low_corr, b_low_corr, c_low_corr)
    print('TOTAL CORR ERROR == systematic/corr error on MC only')
    print("+ "+str(FCCD_err_systMC_up) +" - "+str(FCCD_err_systMC_low))
    FCCD_err_corr_up, FCCD_err_corr_low = FCCD_err_systMC_up, FCCD_err_systMC_low

    #TOTAL UNCORR ERROR == stat/uncorr MC + stat/uncorr data
    FCCD_err_statMCstatdata_up = invert_exponential(O_Ba133_data-O_Ba133_data_err, a_up_uncorr, b_up_uncorr, c_up_uncorr) -FCCD_data
    FCCD_err_statMCstatdata_low = FCCD_data - invert_exponential(O_Ba133_data+O_Ba133_data_err, a_low_uncorr, b_low_uncorr, c_low_uncorr)
    print('TOTAL UNCORR ERROR == stat/uncorr error on MC and stat/uncorr error on data')
    print("+ "+str(FCCD_err_statMCstatdata_up) +" - "+str(FCCD_err_statMCstatdata_low))
    FCCD_err_uncorr_up, FCCD_err_uncorr_low = FCCD_err_statMCstatdata_up, FCCD_err_statMCstatdata_low

    print("sum in quadrature of corr and uncorr error:")
    FCCD_err_corr_uncorr_quadrature_up = np.sqrt(FCCD_err_uncorr_up**2 + FCCD_err_corr_up)
    FCCD_err_corr_uncorr_quadrature_low = np.sqrt(FCCD_err_uncorr_low**2 + FCCD_err_corr_low)
    print("+ ", FCCD_err_corr_uncorr_quadrature_up, ", - ", FCCD_err_corr_uncorr_quadrature_low)

    print("linear sum of corr and uncorr error:")
    print("+ ", FCCD_err_corr_up + FCCD_err_uncorr_up,", - ", FCCD_err_corr_low + FCCD_err_uncorr_low)

    #=============Complete Plot===================

    props = dict(boxstyle='round', alpha=0.5)
    info_str = '\n'.join((r'MC fit: $y=a*e^{-bx}+c$',r'$a=%.3f \pm %.3f$' % (a, np.sqrt(pcov[0][0])), r'$b=%.3f \pm %.3f$' % (b, np.sqrt(pcov[1][1])), r'$\chi^2/dof=%.2f/%.0f$'%(chi_sq, dof), r'FCCD_data=$%.3f^{+%.2f}_{-%.2f}$ mm' % (FCCD_data, FCCD_err_total_up, FCCD_err_total_low)))
    plt.text(0.02, 0.98, info_str, transform=ax.transAxes, fontsize=8,verticalalignment='top', bbox=props) #ax.text..ax.tra

    #plot horizontal data line and errors
    plt.hlines(O_Ba133_data, 0, FCCD_list[-1], color=plot_colors["data"], label = 'data')

    plt.plot(xfit, [O_Ba133_data+O_Ba133_data_err]*(len(xfit)), plot_colors["data_err_stat"], label = 'Data err (stat/uncorr)', linestyle = 'dashed', linewidth = '1.0')
    plt.plot(xfit, [O_Ba133_data-O_Ba133_data_err]*(len(xfit)), plot_colors["data_err_stat"], linestyle = 'dashed', linewidth = '1.0')

    #plot vertical lines
    plt.vlines(FCCD_data, 0, O_Ba133_data, color=plot_colors["FCCD"] , linestyles='dashed')

    #plot total error line
    plt.vlines(FCCD_data+FCCD_err_total_up, 0, O_Ba133_data-O_Ba133_data_err, color=plot_colors["FCCD_err_total"], linestyles='dashed', linewidths=1, label="FCCD total err")
    plt.vlines(FCCD_data-FCCD_err_total_low, 0, O_Ba133_data+O_Ba133_data_err, color=plot_colors["FCCD_err_total"], linestyles='dashed', linewidths=1)

    # #plot stat data error line
    # plt.vlines(FCCD_data+FCCD_err_statdata_up, 0, O_Ba133_data-O_Ba133_data_err, color=plot_colors["FCCD_err_datastat"], linestyles='dashed', linewidths=1, label="FCCD stat data err")
    # plt.vlines(FCCD_data-FCCD_err_statdata_low, 0, O_Ba133_data+O_Ba133_data_err, color=plot_colors["FCCD_err_datastat"], linestyles='dashed', linewidths=1)

    # plot syst MC/total corr error line
    plt.vlines(FCCD_data+FCCD_err_corr_up, 0, O_Ba133_data, color=plot_colors["FCCD_err_MCsyst"], linestyles='dashed', linewidths=1, label = "FCCD corr err")
    plt.vlines(FCCD_data-FCCD_err_corr_low, 0, O_Ba133_data, color=plot_colors["FCCD_err_MCsyst"], linestyles='dashed', linewidths=1)

    # # #plot stat MC error line
    # plt.vlines(FCCD_data+FCCD_err_statMC_up, 0, O_Ba133_data, color=plot_colors["FCCD_err_MCstat"], linestyles='dashed', linewidths=1, label = "FCCD stat MC err")
    # plt.vlines(FCCD_data-FCCD_err_statMC_low, 0, O_Ba133_data, color=plot_colors["FCCD_err_MCstat"], linestyles='dashed', linewidths=1)

    # #plot syst+stat MC error line
    # plt.vlines(FCCD_data+FCCD_err_statsystMC_up, 0, O_Ba133_data, color=plot_colors["FCCD_err_MCstatsyst"], linestyles='dashed', linewidths=1, label = "FCCD syst+stat MC err")
    # plt.vlines(FCCD_data-FCCD_err_statsystMC_low, 0, O_Ba133_data, color=plot_colors["FCCD_err_MCstatsyst"], linestyles='dashed', linewidths=1)

    # #plot stat MC + stat data/ total uncorr:
    plt.vlines(FCCD_data+FCCD_err_uncorr_up, 0, O_Ba133_data-O_Ba133_data_err, color=plot_colors["FCCD_err_statMCstatdata"], linestyles='dashed', linewidths=1, label = "FCCD uncorr err")
    plt.vlines(FCCD_data-FCCD_err_uncorr_low, 0, O_Ba133_data-O_Ba133_data_err, color=plot_colors["FCCD_err_statMCstatdata"], linestyles='dashed', linewidths=1)

    plt.ylabel(r'$O_{Ba133} = C_{79.6keV}+C_{91keV}/C_{356keV}$')
    plt.xlabel("FCCD (mm)")
    plt.xlim(0,FCCD_list[-1])
    plt.ylim(0.0,1.8)
    plt.title(detector)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()


    # #try propagation of errors -- ignore this
    # FCCD_err_prop = np.sqrt((1/b**2)*((a_err/a)**2 +  (1/b**2)*((np.log(a/(O_Ba133_data-c))**2)*b_err**2 + (1/(O_Ba133_data-c)**2))*(c_err**2 + O_Ba133_data_err**2)))
    # print("FCCD err from propagation of fit errors with data error")
    # print(FCCD_err_prop)
    # FCCD_err_prop_cov = propagation_of_errors(a,b,c,O_Ba133_data,pcov,O_Ba133_data_err)
    # print("FCCD err from propagation of fit errors with data error AND COVARIANCE")
    # print(FCCD_err_prop_cov)

    '''
    if cuts == False:
        plt.savefig(dir+"/FCCD/plots/FCCD_OBa133_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+".png")
    else:
        plt.savefig(dir+"/FCCD/plots/FCCD_OBa133_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+"_cuts.png")
    '''

    # Save interpolated fccd for data to a json file

    FCCD_data_dict = {
        "FCCD": FCCD_data,

        "FCCD_err_total_up": FCCD_err_total_up,
        "FCCD_err_total_low": FCCD_err_total_low,

        "FCCD_err_corr_up": FCCD_err_corr_up,
        "FCCD_err_corr_low": FCCD_err_corr_low,

        "FCCD_err_uncorr_up": FCCD_err_uncorr_up,
        "FCCD_err_uncorr_low": FCCD_err_uncorr_low,

        "FCCD_err_corr_uncorr_quadrature_up": FCCD_err_corr_uncorr_quadrature_up,
        "FCCD_err_corr_uncorr_quadrature_low": FCCD_err_corr_uncorr_quadrature_low,

        "O_Ba133_data": O_Ba133_data,
        "O_Ba133_data_err": O_Ba133_data_err,
        "a": a,
        "a_err": (a_up-a_low)/2,
        "b": b,
        "b_err": b_err
    }

    if cuts == False:
        with open(dir+"/FCCD/test_err/FCCD_data_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+".json", "w") as outfile:
            json.dump(FCCD_data_dict, outfile, indent=4)
    else:
        if cuts_sigma ==4:
            with open(dir+"/FCCD/test_err/FCCD_data_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+"_cuts.json", "w") as outfile:
                json.dump(FCCD_data_dict, outfile, indent=4)
        else:
            with open(dir+"/FCCD/test_err/FCCD_data_"+MC_id+"_"+smear+"_"+TL_model+"_fracFCCDbore"+frac_FCCDbore+"_"+energy_filter+"_run"+str(run)+"_cuts_"+str(cuts_sigma)+"sigma.json", "w") as outfile:
                json.dump(FCCD_data_dict, outfile, indent=4)

    print("done")

def exponential_decay(x, a, b ,c):
    f = a*np.exp(-b*x) + c
    return f

def error_O_Ba133_MC(O_Ba133, O_Ba133_err):
    "Returns the total, correlated and uncorrelated % errors on count ratio observable for the MC."

    #MC Systematics
    #values from Bjoern's thesis - Barium source, all percentages
    gamma_line=0.69 
    geant4=2.
    source_thickness=0.02
    source_material=0.01
    endcap_thickness=0.28
    detector_cup_thickness=0.07
    detector_cup_material=0.03

    #MC statistical
    MC_statistics = O_Ba133_err/O_Ba133*100

    #Total error: sum in quadrature of all contributions
    tot_error=np.sqrt(gamma_line**2+geant4**2+source_thickness**2+source_material**2+endcap_thickness**2+detector_cup_thickness**2+detector_cup_material**2+MC_statistics**2)

    correlated = [gamma_line, geant4, source_thickness, source_material, endcap_thickness, detector_cup_thickness, detector_cup_material]
    uncorrelated = [MC_statistics]

    #correlated error
    corr_error = np.sqrt(gamma_line**2+geant4**2+source_thickness**2+source_material**2+endcap_thickness**2+detector_cup_thickness**2+detector_cup_material**2)

    #uncorrelated error
    uncorr_error = MC_statistics
    
    return tot_error, corr_error, uncorr_error


def invert_exponential(x,a,b,c):
    #for f=a*exp(-b*x) +c
    # x = the O_ba133 value
    return (1/b)*np.log(a/(x-c))

# def propagation_of_errors(a,b,c,d,pcov,d_err):

#     dyda = 1/a*b
#     dydb = (-1/b**2)*(np.log(a/(d-c)))
#     dydc = (1/b)*(1/(d-c))
#     dydd = (1/b)*(1/(d-c))

#     err_sq = (dydd*d_err)**2 
#     err_sq =+ (dyda*dyda)*pcov[0,0] + (dyda*dydb)*pcov[0,1] + (dyda*dydc)*pcov[0,2] 
#     err_sq =+ (dydb*dyda)*pcov[1,0] + (dydb*dydb)*pcov[1,1] + (dydb*dydc)*pcov[1,2]
#     err_sq =+ (dydc*dyda)*pcov[2,0] + (dydc*dydb)*pcov[2,1] + (dydc*dydc)*pcov[2,2]

#     err = np.sqrt(err_sq)

#     return err

if __name__ == "__main__":
    main()
