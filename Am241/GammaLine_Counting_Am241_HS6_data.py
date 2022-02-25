import pandas as pd
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import argparse
from scipy import optimize
from scipy import stats
import glob
import json
from datetime import datetime
from scipy.integrate import quad
import fnmatch
from scipy.special import erf, erfc

from pygama.analysis import histograms
from pygama.analysis import peak_fitting

import pygama.genpar_tmp.cuts as cut

#Script to fit the gamma lines in the Am241 spectra, for data and/or MC


def main():

    par = argparse.ArgumentParser(description="fit and count gamma lines in Am241 spectrum",
                                  usage="python GammaLine_Counting.py [OPERATION: -d -s] [arguments: ]"
    )
    arg, st, sf = par.add_argument, "store_true", "store_false"
    arg("-d", "--data",  nargs=5, help="fit data, usage: python GammaLine_Counting.py --data <detector> <energy_filter> <cuts> <run>")
    arg("-s", "--sim", nargs=4, help="fit processed simulations, usage: python GammaLine_Counting.py --sim <detector> <sim_path> <MC_id>")

    args=vars(par.parse_args())

    #get path of this script:
    dir=os.path.dirname(os.path.realpath(__file__))
    print("working directory: ", dir)

    #========DATA MODE=======:
    if args["data"]:
        detector, energy_filter, cuts, run, source = args["data"][0], args["data"][1], args["data"][2], args["data"][3], args["data"][4]
        print("")
        print("MODE: Data")
        print("detector: ", detector)
        #print("calibration path: ", calibration)
        print("energy filter: ", energy_filter)
        print("applying cuts: ", cuts)
        print("run: ", run)
        print("source: ", source)
        print("")

        if cuts == "False":
            cuts = False
        else:
            cuts = True
            sigma_cuts = 4


        #initialise directories for detectors to save
        if not os.path.exists(dir+"/PeakCounts/"+detector+"/"+source+"/new/plots/data/"):
            os.makedirs(dir+"/PeakCounts/"+detector+"/"+source+"/new/plots/data/")

        #Get data and concoatonate into df
        df=pd.read_hdf(dir+"/data_calibration/"+detector+"/"+source+"/loaded_energy_"+detector+"_"+energy_filter+"_run"+str(run)+".hdf5", key='energy')
        energies=df['energy_filter']

        #Get Calibration
        #with open(calibration) as json_file:
        #   calibration_coefs = json.load(json_file)
        #m = calibration_coefs[energy_filter]["calibration"][0]
        #c = calibration_coefs[energy_filter]["calibration"][1]

        # energies = (energy_filter_data-c)/m
        #energies = energy_filter_data*m + c

    #========SIMULATION MODE:==============
    if args["sim"]:
        detector, sim_path, MC_id, source = args["sim"][0], args["sim"][1], args["sim"][2], args["sim"][3]
        print("")
        print("MODE: Simulations")
        print("detector: ", detector)
        print("sim path: ", sim_path)
        print("MC_id: ", MC_id)
        print("source: ", source)
        print("")

       #initialise directories for detectors to save
        if not os.path.exists(dir+"/PeakCounts/"+detector+"/"+source+"/new/plots/sim/"):
            os.makedirs(dir+"/PeakCounts/"+detector+"/"+source+"/new/plots/sim/")

        #get energies
        # MC_file = hdf5_path+"processed_detector_"+MC_file_id+'_FCCD'+str(FCCD)+'mm_DLF'+str(DLF)+'.hdf5'
        df =  pd.read_hdf(sim_path, key="procdf")
        energies = df['energy']


    #get total pygama histogram
    binwidth = 0.1 #keV
    bins = np.arange(0,140,binwidth)
    hist, bins, var = histograms.get_hist(energies, bins=bins)

    #_________Fit 99/103 double peak____________:

    print("99/103 keV")

    #prepare histogram
    xmin_99_103, xmax_99_103 = 97, 105
    bins_peak = np.arange(xmin_99_103,xmax_99_103,binwidth)
    #print(bins_peak)
    hist_peak, bins_peak, var_peak = histograms.get_hist(energies, bins=bins_peak)
    bins_centres_peak = histograms.get_bin_centers(bins_peak)
    #print(bins_centres_peak)

    #zeros = (hist_peak == 0)
    #mask = ~(zeros)
    #sigma = np.sqrt(hist_peak)
    #hist_peak = hist_peak[mask]

    #bins_centres_peak = histograms.get_bin_centers(bins_peak)[mask]
    #print(bins_centres_peak)

    #fit function initial guess
    R =  0.0203/0.0195
    mu_99_guess, sigma_99_guess, a_99_guess, bkg_99_guess, s_99_guess = 99., 0.8, max(hist_peak)*R, min(hist_peak), min(hist_peak)
    mu_103_guess, sigma_103_guess, a_103_guess = 103., 0.8, max(hist_peak)
    mu_small_guess, sigma_small_guess, a_small_guess = 101., 0.4, max(hist_peak)*0.05
    double_guess = [a_99_guess, mu_99_guess, sigma_99_guess,
                    a_103_guess, mu_103_guess, sigma_103_guess,
                    a_small_guess, mu_small_guess, sigma_small_guess,
                    bkg_99_guess,s_99_guess]
    try:

        coeff, cov_matrix = peak_fitting.fit_hist(Am_double, hist_peak, bins_peak, var=None, guess=double_guess, poissonLL=False, integral=None, method=None, bounds=None)

        a_99, mu_99, sigma_99, a_103, mu_103, sigma_103, a_small, mu_small, sigma_small = coeff[0], coeff[1], coeff[2], coeff[3], coeff[4], coeff[5], coeff[6], coeff[7], coeff[8]
        bkg_99, s_99 = coeff[9], coeff[10]
        a_99_err, mu_99_err, sigma_99_err, a_103_err, mu_103_err, sigma_103_err, a_small_err, mu_small_err, sigma_small_err = np.sqrt(cov_matrix[0][0]), np.sqrt(cov_matrix[1][1]), np.sqrt(cov_matrix[2][2]), np.sqrt(cov_matrix[3][3]), np.sqrt(cov_matrix[4][4]), np.sqrt(cov_matrix[5][5]), np.sqrt(cov_matrix[6][6]), np.sqrt(cov_matrix[7][7]), np.sqrt(cov_matrix[8][8])
        bkg_99_err , s_99_err  = np.sqrt(cov_matrix[9][9]), np.sqrt(cov_matrix[10][10])
        #compute chi sq of fit
        chi_sq, p_value, residuals, dof = chi_sq_calc(bins_centres_peak, hist_peak, np.sqrt(hist_peak), Am_double, coeff)
        print("r chi sq: ", chi_sq/dof)

        #Counting - integrate gaussian signal part

        C_99_103, C_99_103_err = double_gauss_count(a_99, mu_99,sigma_99, a_103, mu_103, sigma_103, binwidth)
        print("peak count 99-103 = ", str(C_99_103)," +/- ", str(C_99_103_err))

        #plot with fit
        xfit = np.linspace(xmin_99_103, xmax_99_103, 1000)
        yfit = Am_double(xfit, *coeff)
        #yfit_step = peak_fitting.step(xfit,mu_99, sigma_99, bkg_99, s_99)
        #yfit_doubleG= double_gauss(xfit, a_99, mu_99, sigma_99, a_103, mu_103, sigma_103)

        fig, ax = plt.subplots()
        histograms.plot_hist(hist_peak, bins_peak, var=None, show_stats=False, stats_hloc=0.75, stats_vloc=0.85)
        plt.plot(xfit, yfit, label=r'gauss($x,\mu_{99},\sigma_{99},a_{99}$)+gauss($x,\mu_{103},\sigma_{103},a_{103}$)+step($x,\mu_{99},\sigma_{99},bkg,s$)')
        #plt.plot(xfit, yfit_doubleG, "--", color='red', label =r'double_gauss($x,\mu_{99},\sigma_{99},a_{99},\mu_{103},\sigma_{103},a_{103}$)')
        #plt.plot(xfit, yfit_step, "--", label =r'step($x,\mu_{99},\sigma_{99},bkg,s$))')


        plt.xlim(xmin_99_103, xmax_99_103)
        #ax.set_ylim(min(hist_peak),max(hist_peak))
        plt.yscale("log")
        plt.xlabel("Energy (keV)")
        plt.ylabel("Counts")
        plt.legend(loc="lower left", prop={'size': 8.5})

        props = dict(boxstyle='round', alpha=0.5)
        info_str = '\n'.join((r'$\mu_{99}=%.3g \pm %.3g$' % (mu_99, mu_99_err), r'$\mu_{103}=%.3g \pm %.3g$' % (mu_103, mu_103_err), r'$\sigma_{99}=%.3g \pm %.3g$' % (sigma_99, sigma_99_err), r'$\sigma_{103}=%.3g \pm %.3g$' % (sigma_103, sigma_103_err), r'$a_{99}=%.3g \pm %.3g$' % (a_99, a_99_err), r'$a_{103}=%.3g \pm %.3g$' % (a_103, a_103_err), r'$\chi^2/dof=%.2f/%.0f$'%(chi_sq, dof)))
        plt.text(0.02, 0.98, info_str, transform=ax.transAxes, fontsize=8,verticalalignment='top', bbox=props)


    except RuntimeError:
        print("Error - curve_fit failed")

        #counting - nan values
        C_99_103, C_99_103_err = np.nan, np.nan
        print("peak count 99-103 = ", str(C_99_103)," +/- ", str(C_99_103_err))

        #plot without fit
        fig, ax = plt.subplots()
        histograms.plot_hist(hist_peak, bins_peak, var=None, show_stats=False, stats_hloc=0.75, stats_vloc=0.85)
        plt.xlim(xmin_99_103, xmax_99_103)
        plt.yscale("log")
        plt.xlabel("Energy (keV)")
        plt.ylabel("Counts")
        plt.legend(loc="upper left", prop={'size': 8.5})


    #Save fig
    if args["sim"]:
        ax.set_title(MC_id, fontsize=9)
        plt.savefig(dir+"/PeakCounts/"+detector+"/"+source+"/new/plots/sim/"+MC_id+"_103keV.png")

    if args["data"]:
        ax.set_title("Data: "+detector, fontsize=9)
        if cuts == False:
            plt.savefig(dir+"/PeakCounts/"+detector+"/"+source+"/plots/data/"+detector+"_103keV_"+energy_filter+"_run"+str(run)+".png")
        else:
            if sigma_cuts ==4:
                plt.savefig(dir+"/PeakCounts/"+detector+"/"+source+"/new/plots/data/"+detector+"_103keV_cuts_"+energy_filter+"_run"+str(run)+".png")


    #___________Fit single peaks_________________:
    peak_counts = []
    peak_counts_err = []
    peak_ranges =[[52,62]]# , [122, 124], [124, 126], [207,209], [334,336], [661,663]] #Rough by eye
    peaks = [60] # 123, 125, 208, 335, 662] #

    for index, i in enumerate(peak_ranges):

        #prepare histogram
        print(str(peaks[index]), " keV")
        xmin, xmax = i[0], i[1]
        bins_peak = np.arange(xmin,xmax,binwidth)
        hist_peak, bins_peak, var_peak = histograms.get_hist(energies, bins=bins_peak)
        bins_centres_peak = histograms.get_bin_centers(bins_peak)


        #fit function initial guess
        mu_59_guess, sigma_59_guess, a_59_guess = 59.5, 0.5, max(hist_peak)
        mu_57_guess, sigma_57_guess, a_57_guess = 57.8, 0.5, max(hist_peak)*0.8    #1.2, *0.8
        mu_53_guess, sigma_53_guess, a_53_guess = 53, 0.5, max(hist_peak)*0.5      #1.2, *0.8
        bkg_guess, s_guess = min(hist_peak), min(hist_peak)
        gauss_step_guess = [mu_59_guess, sigma_59_guess, a_59_guess, mu_57_guess, sigma_57_guess, a_57_guess, mu_53_guess, sigma_53_guess, a_53_guess, bkg_guess, s_guess]
        bounds = ([0, 0, 0, 0, 0, 0, 0, 0, 0, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

        #fit - gauss step
        try:
            coeff, cov_matrix = peak_fitting.fit_hist(Am_60, hist_peak, bins_peak, var=None, guess=gauss_step_guess, poissonLL=False, integral=None, method=None, bounds=bounds)
            mu_59, sigma_59, a_59 = coeff[0], coeff[1], coeff[2]
            mu_57, sigma_57, a_57= coeff[3], coeff[4], coeff[5]
            mu_53, sigma_53, a_53= coeff[6], coeff[7], coeff[8]
            bkg, s = coeff[9], coeff[10]
            mu_59_err, sigma_59_err, a_59_err = np.sqrt(cov_matrix[0][0]), np.sqrt(cov_matrix[1][1]), np.sqrt(cov_matrix[2][2])
            mu_57_err, sigma_57_err, a_57_err = np.sqrt(cov_matrix[3][3]), np.sqrt(cov_matrix[4][4]), np.sqrt(cov_matrix[5][5])
            mu_53_err, sigma_53_err, a_53_err = np.sqrt(cov_matrix[6][6]), np.sqrt(cov_matrix[7][7]), np.sqrt(cov_matrix[8][8])
            bkg_err, s_err = np.sqrt(cov_matrix[9][9]), np.sqrt(cov_matrix[10][10])
            #compute chi sq of fit
            chi_sq, p_value, residuals, dof = chi_sq_calc(bins_centres_peak, hist_peak, np.sqrt(hist_peak), Am_60, coeff)
            print("r chi sq: ", chi_sq/dof)

            #Counting - integrate gaussian signal part +/- 3 sigma
            C, C_err = gauss_count(a_59, mu_59 ,sigma_59, binwidth)
            print("peak counts = ", str(C)," +/- ", str(C_err))
            peak_counts.append(C)
            peak_counts_err.append(C_err)

            #plot
            xfit = np.linspace(xmin, xmax, 1000)
            yfit = Am_60(xfit, *coeff)
            #yfit_step = peak_fitting.step(xfit ,mu_59, sigma_59, bkg, s)
            #yfit_gaus53 = peak_fitting.gauss(xfit ,mu_53, sigma_53, a_53)
            #yfit_gaus57 = peak_fitting.gauss(xfit ,mu_57, sigma_57, a_57)
            yfit_gaus59 = peak_fitting.gauss(xfit ,mu_59, sigma_59, a_59)

            fig, ax = plt.subplots()
            histograms.plot_hist(hist_peak, bins_peak, var=None, show_stats=False, stats_hloc=0.75, stats_vloc=0.85)
            plt.plot(xfit, yfit, label=r'total_fit: gauss($x,\mu_{59},\sigma_{59},a_{59})+step(x,\mu_{59},\sigma_{59},bkg,s)+gauss_{53}+gauss_{57}$')
            #plt.plot(xfit, yfit_step, "--", label =r'step($x,\mu,\sigma,bkg,s$)')
            #plt.plot(xfit, yfit_gaus53, "--", color='blue', label =r'gaus53($x,\mu,\sigma,a)')
            #plt.plot(xfit, yfit_gaus57, "--", color='green', label =r'gaus57($x,\mu,\sigma,a)')
            #plt.plot(xfit, yfit_gaus59, "--", color='red', label =r'gauss59: gauss($x,\mu_{59},\sigma_{59},a_{59}$)')

            plt.xlim(xmin, xmax)
            #ax.set_ylim(min(hist_peak)*0.8,max(hist_peak)*1.2)
            plt.yscale("log")
            plt.xlabel("Energy (keV)")
            plt.ylabel("Counts")
            plt.legend(loc="lower left", prop={'size': 8.5})

            props = dict(boxstyle='round', alpha=0.5)
            info_str = '\n'.join((r'$\mu_{59}=%.3g \pm %.3g$' % (mu_59, mu_59_err), r'$\sigma_{59}=%.3g \pm %.3g$' % (sigma_59, sigma_59_err),r'$a_{59}=%.3g \pm %.3g$' % (a_59, a_59_err), r'$\chi^2/dof=%.2f/%.0f$'%(chi_sq, dof)))
            plt.text(0.02, 0.98, info_str, transform=ax.transAxes, fontsize=8,verticalalignment='top', bbox=props)

        except RuntimeError:
            print("Error - curve_fit failed")

            #counting - nan values
            C, C_err = np.nan, np.nan
            print("peak counts = ", str(C)," +/- ", str(C_err))
            peak_counts.append(C)
            peak_counts_err.append(C_err)

            #plot without fit
            fig, ax = plt.subplots()
            histograms.plot_hist(hist_peak, bins_peak, var=None, show_stats=False, stats_hloc=0.75, stats_vloc=0.85)
            plt.xlim(xmin, xmax)
            plt.yscale("log")
            plt.xlabel("Energy (keV)")
            plt.ylabel("Counts")
            plt.legend(loc="upper left", prop={'size': 8.5})


        #Save fig
        if args["sim"]:
            ax.set_title(MC_id, fontsize=9)
            plt.savefig(dir+"/PeakCounts/"+detector+"/"+source+"/new/plots/sim/"+MC_id+"_"+str(peaks[index])+'keV.png')

        if args["data"]:
            ax.set_title("Data: "+detector, fontsize=9)
            if cuts == False:
                plt.savefig(dir+"/PeakCounts/"+detector+"/"+source+"/plots/data/"+detector+"_"+str(peaks[index])+'keV_'+energy_filter+'_run'+str(run)+'.png')
            else:
                if sigma_cuts ==4:
                    plt.savefig(dir+"/PeakCounts/"+detector+"/"+source+"/new/plots/data/"+detector+"_"+str(peaks[index])+'keV_cuts_'+energy_filter+'_run'+str(run)+'.png')



    #Comput count ratio O_Am241
    print("")
    C_60, C_60_err = peak_counts[0], peak_counts_err[0]
    if (C_60 == np.nan) or (C_99_103 == np.nan):
        O_Am241, O_Am241_err = np.nan, np.nan
    else:
        O_Am241 = C_60/C_99_103
        O_Am241_err = np.sqrt((C_60_err/C_60)**2 + (C_99_103_err/C_99_103)**2)*O_Am241
    print("O_Am241 = " , O_Am241, " +/- ", O_Am241_err)


    #Save count values to json file
    PeakCounts = {
        "C_60" : peak_counts[0],
        "C_60_err" : peak_counts_err[0],
        "C_99_103" : C_99_103,
        "C_99_103_err" : C_99_103_err,
#        "C_103" : C_103,
#        "C_103_err" : C_103_err,
#        "C_208" : peak_counts[3],
#        "C_208_err" : peak_counts_err[3],
#        "C_335" : peak_counts[4],
#        "C_335_err" : peak_counts_err[4],
#        "C_662" : peak_counts[5],
#        "C_662_err" : peak_counts_err[5],
        "O_Am241" : O_Am241,
        "O_Am241_err" : O_Am241_err,
    }

    if args["sim"]:
        with open(dir+"/PeakCounts/"+detector+"/"+source+"/new/PeakCounts_sim_"+MC_id+".json", "w") as outfile:
            json.dump(PeakCounts, outfile, indent=4)
    if args["data"]:
        if cuts == False:
            with open(dir+"/PeakCounts/"+detector+"/"+source+"/PeakCounts_data_"+detector+"_"+energy_filter+"_run"+str(run)+".json", "w") as outfile:
                json.dump(PeakCounts, outfile, indent=4)
        else:
            if sigma_cuts ==4:
                with open(dir+"/PeakCounts/"+detector+"/"+source+"/new/PeakCounts_data_"+detector+"_cuts_"+energy_filter+"_run"+str(run)+".json", "w") as outfile:
                    json.dump(PeakCounts, outfile, indent=4)
            else:
                with open(dir+"/PeakCounts/"+detector+"/"+source+"/PeakCounts_data_"+detector+"_cuts_"+energy_filter+"_run"+str(run)+"_"+str(sigma_cuts)+"sigma.json", "w") as outfile:
                    json.dump(PeakCounts, outfile, indent=4)


def chi_sq_calc(xdata, ydata, yerr, fit_func, coeff):
    "calculate chi sq and p-val of a fit given the data points and fit parameters, e.g. fittype ='linear'"

    y_obs = ydata
    y_exp = []

    y_exp = []
    for index, y_i in enumerate(y_obs):
        x_obs = xdata[index]
        y_exp_i = fit_func(x_obs, *coeff)
        y_exp.append(y_exp_i)

    #chi_sq, p_value = stats.chisquare(y_obs, y_exp)#this is without errors
    chi_sq = 0.
    residuals = []
    for i in range(len(y_obs)):
        if yerr[i] != 0:
            residual = (y_obs[i]-y_exp[i])/(yerr[i])
        else:
            residual = 0.
        chi_sq += (residual)**2
        residuals.append(residual)

    N = len(y_exp) #number of data points
    dof = N-1
    chi_sq_red = chi_sq/dof

    p_value = 1-stats.chi2.cdf(chi_sq, dof)

    return chi_sq, p_value, residuals, dof

def gauss_count(a,mu,sigma, bin_width):
    "count/integrate gaussian peak"

    #height = a/sigma/np.sqrt(2*np.pi)
    #integral = a/bin_width
    #integral_err = a_err/bin_width

    #_____3sigma_____
    #integral_60_3sigma_list = quad(peak_fitting.gauss,mu-3*sigma, mu+3*sigma, args=(mu,sigma,a))
    integral_list = quad(peak_fitting.gauss,0,120, args=(mu,sigma,a))
    integral = integral_list[0]/bin_width
    integral_err = integral_list[1]/bin_width

    return integral, integral_err

def double_gauss_count(a_99,mu_99,sigma_99, a_103, mu_103, sigma_103, bin_width):
    "count/integrate double gaussian peak"

    integral_list = quad(double_gauss, 0, 120, args=(a_99,mu_99,sigma_99, a_103, mu_103, sigma_103))
    integral = integral_list[0]/bin_width
    integral_err = integral_list[1]/bin_width

    return integral, integral_err

def double_gauss(x, a_99,mu_99,sigma_99, a_103, mu_103, sigma_103):
    peak_99 = peak_fitting.gauss(x,mu_99, sigma_99, a_99)
    peak_103 = peak_fitting.gauss(x,mu_103, sigma_103, a_103)
    double_peak = peak_99 + peak_103

    return double_peak

def Am_60(x, mu_59, sigma_59, a_59, mu_57, sigma_57, a_57, mu_53, sigma_53, a_53, bkg, s):

    R =  0.0203/0.0195 #intensity ratio for Ba-133 double peak

    peak_59 = peak_fitting.gauss(x,mu_59, sigma_59, a_59)

    peak_57 = peak_fitting.gauss(x,mu_57, sigma_57, a_57)

    bump_53 = peak_fitting.gauss(x, mu_53, sigma_53, a_53)

    step = peak_fitting.step(x,mu_59,sigma_59,bkg,s)

    f = peak_59 + peak_57 + bump_53 + step

    return f


def Am_double(x,a1,mu1,sigma1,a2,mu2,sigma2,a3,mu3,sigma3,b,s,
              components=False) :
    """
    A Fit function exclusevly for a 241Am 99keV and 103keV lines situation
    Consists of
     - three gaussian peaks (two lines + one bkg line in between)
     - two steps (for the two lines)
     - two tails (for the two lines)
    """

    step1 = peak_fitting.step(x,mu1,sigma1,b,s)
    #step2 = peak_fitting.step(x,mu2,sigma2,b,s)

    gaus1 = peak_fitting.gauss(x,mu1,sigma1,a1)
    gaus2 = peak_fitting.gauss(x,mu2,sigma2,a2)
    gaus3 = peak_fitting.gauss(x,mu3,sigma3,a3)



    double_f = step1+ gaus1 + gaus2 + gaus3

    if components:
       return double_f, gaus1, gaus2, gaus3, step1,
    else:
       return double_f

if __name__ =="__main__":
    main()
