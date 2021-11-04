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

import pygama.io.lh5 as lh5
from pygama.analysis import histograms
from pygama.analysis import peak_fitting

import pygama.genpar_tmp.cuts as cut

#Script to fit the gamma lines in the Am241 spectra, for data and/or MC


def main():

    par = argparse.ArgumentParser(description="fit and count gamma lines in Am241 spectrum",
                                  usage="python GammaLine_Counting.py [OPERATION: -d -s] [arguments: ]"
    )
    arg, st, sf = par.add_argument, "store_true", "store_false"
    arg("-d", "--data",  nargs=6, help="fit data, usage: python GammaLine_Counting.py --data <detector> <data_path> <calibration> <energy_filter> <cuts> <run>")
    arg("-s", "--sim", nargs=3, help="fit processed simulations, usage: python GammaLine_Counting.py --sim <detector> <sim_path> <MC_id>")

    args=vars(par.parse_args())

    #get path of this script:
    dir=os.path.dirname(os.path.realpath(__file__))
    print("working directory: ", dir)

    #========DATA MODE=======:
    if args["data"]:
        detector, data_path, calibration, energy_filter, cuts, run = args["data"][0], args["data"][1], args["data"][2], args["data"][3], args["data"][4], int(args["data"][5])
        print("")
        print("MODE: Data")
        print("detector: ", detector)
        print("data path: ", data_path)
        print("calibration path: ", calibration)
        print("energy filter: ", energy_filter)
        print("applying cuts: ", cuts)
        print("run: ", run)
        print("")

        if cuts == "False":
            cuts = False
        else:
            cuts = True

        #initialise directories for detectors to save
        if not os.path.exists(dir+"/PeakCounts/"+detector+"/plots/data/"):
            os.makedirs(dir+"/PeakCounts/"+detector+"/plots/data/")

        #Get data and concoatonate into df
        if cuts == False:
            df_total_lh5 = read_all_dsp_lh5(data_path,cuts,run=run)
        else:
            sigma_cuts = 4
            print("sigma cuts: ", str(sigma_cuts))
            df_total_lh5 = read_all_dsp_lh5(data_path,cuts,run=run, sigma=sigma_cuts)


        print("df_total_lh5: ", df_total_lh5)
        energy_filter_data = df_total_lh5[energy_filter]

        #Get Calibration
        with open(calibration) as json_file:
            calibration_coefs = json.load(json_file)
        # m = calibration_coefs[energy_filter]["Calibration_pars"][0]
        # c = calibration_coefs[energy_filter]["Calibration_pars"][1]
        m = calibration_coefs[energy_filter]["calibration"][0]
        c = calibration_coefs[energy_filter]["calibration"][1]

        # energies = (energy_filter_data-c)/m
        energies = energy_filter_data*m + c

    #========SIMULATION MODE:==============
    if args["sim"]:
        detector, sim_path, MC_id = args["sim"][0], args["sim"][1], args["sim"][2]
        print("")
        print("MODE: Simulations")
        print("detector: ", detector)
        print("sim path: ", sim_path)
        print("MC_id: ", MC_id)
        print("")

       #initialise directories for detectors to save
        if not os.path.exists(dir+"/PeakCounts/"+detector+"/plots/sim/"):
            os.makedirs(dir+"/PeakCounts/"+detector+"/plots/sim/")

        #get energies
        # MC_file = hdf5_path+"processed_detector_"+MC_file_id+'_FCCD'+str(FCCD)+'mm_DLF'+str(DLF)+'.hdf5'
        df =  pd.read_hdf(sim_path, key="procdf")
        energies = df['energy']


    #get total pygama histogram
    binwidth = 0.3 #keV
    bins = np.arange(0,700,binwidth)
    hist, bins, var = histograms.get_hist(energies, bins=bins)

    #_________Fit 99/103 double peak____________:

    print("99/103 keV")

    #prepare histogram
    xmin_99_103, xmax_99_103 = 95, 107
    bins_peak = np.arange(xmin_99_103,xmax_99_103,binwidth)
    hist_peak, bins_peak, var_peak = histograms.get_hist(energies, bins=bins_peak)
    bins_centres_peak = histograms.get_bin_centers(bins_peak)

    zeros = (hist_peak == 0)
    mask = ~(zeros)
    sigma = np.sqrt(hist_peak)
    hist_peak = hist_peak[mask]

    bins_centres_peak = histograms.get_bin_centers(bins_peak)[mask]


    #fit function initial guess
    R =  0.0203/0.0195
    mu_99_guess, sigma_99_guess, a_99_guess, bkg_99_guess, s_99_guess = 99., 0.5, max(hist_peak)*R, min(hist_peak), min(hist_peak)
    mu_103_guess, sigma_103_guess, a_103_guess, bkg_103_guess, s_103_guess = 103., 0.5, max(hist_peak), min(hist_peak), min(hist_peak)
    mu_small_guess, sigma_small_guess, a_small_guess = 101., 0.5, max(hist_peak)*0.1
    double_guess = [a_99_guess, mu_99_guess, sigma_99_guess,
                    a_103_guess, mu_103_guess, sigma_103_guess,
                    a_small_guess, mu_small_guess, sigma_small_guess,
                    bkg_99_guess, bkg_103_guess, s_99_guess, s_103_guess]
    try:

        coeff, cov_matrix = peak_fitting.fit_hist(peak_fitting.Am_double, hist_peak, bins_peak, var=None, guess=double_guess, poissonLL=False, integral=None, method=None, bounds=None)

        a_99, mu_99, sigma_99, a_103, mu_103, sigma_103, a_small, mu_small, sigma_small = coeff[0], coeff[1], coeff[2], coeff[3], coeff[4], coeff[5], coeff[6], coeff[7], coeff[8]
        bkg_99, bkg_103, s_99, s_103 = coeff[9], coeff[10], coeff[11], coeff[12]
        a_99_err, mu_99_err, sigma_99_err, a_103_err, mu_103_err, sigma_103_err, a_small_err, mu_small_err, sigma_small_err = np.sqrt(cov_matrix[0][0]), np.sqrt(cov_matrix[1][1]), np.sqrt(cov_matrix[2][2]), np.sqrt(cov_matrix[3][3]), np.sqrt(cov_matrix[4][4]), np.sqrt(cov_matrix[5][5]), np.sqrt(cov_matrix[6][6]), np.sqrt(cov_matrix[7][7]), np.sqrt(cov_matrix[8][8])
        bkg_99_err , bkg_103_err , s_99_err , s_103_err = np.sqrt(cov_matrix[9][9]), np.sqrt(cov_matrix[10][10]), np.sqrt(cov_matrix[11][11]), np.sqrt(cov_matrix[12][12])

        #compute chi sq of fit
        chi_sq, p_value, residuals, dof = chi_sq_calc(bins_centres_peak, hist_peak, np.sqrt(hist_peak), peak_fitting.Am_double, coeff)
        print("r chi sq: ", chi_sq/dof)

        #Counting - integrate gaussian signal part +/- 3 sigma -- change!

        C_99_103, C_99_103_err = double_gauss_count(a_99, mu_99,sigma_99, a_103, mu_103, sigma_103, binwidth)
        print("peak count 99-103 = ", str(C_99_103)," +/- ", str(C_99_103_err))

        #plot with fit
        xfit = np.linspace(xmin_99_103, xmax_99_103, 1000)
        yfit = peak_fitting.Am_double(xfit, *coeff)
        #yfit_step = peak_fitting.step(xfit,mu_99, sigma_99, bkg_99, s_99)

        fig, ax = plt.subplots()
        histograms.plot_hist(hist_peak, bins_peak, var=None, show_stats=False, stats_hloc=0.75, stats_vloc=0.85)
        plt.plot(xfit, yfit, label=r'gauss($x,\mu_{99},\sigma_{99},Ra$)+gauss($x,\mu_{103},\sigma_{103},a$)+step($x,\mu_{99},\sigma_{99},bkg,s$)')
        #plt.plot(xfit, yfit_step, "--", label =r'step($x,\mu_{99},\sigma_{99},bkg,s$))')

        plt.xlim(xmin_99_103, xmax_99_103)
        plt.yscale("log")
        plt.xlabel("Energy (keV)")
        plt.ylabel("Counts")
        plt.legend(loc="upper left", prop={'size': 8.5})

        props = dict(boxstyle='round', alpha=0.5)
        info_str = '\n'.join((r'$\mu_{99}=%.3g \pm %.3g$' % (mu_99, mu_99_err), r'$\mu_{103}=%.3g \pm %.3g$' % (mu_103, mu_103_err), r'$\sigma_{99}=%.3g \pm %.3g$' % (sigma_99, sigma_99_err), r'$\sigma_{103}=%.3g \pm %.3g$' % (sigma_103, sigma_103_err), r'$\chi^2/dof=%.2f/%.0f$'%(chi_sq, dof)))
        plt.text(0.02, 0.8, info_str, transform=ax.transAxes, fontsize=8,verticalalignment='top', bbox=props)


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
        plt.savefig(dir+"/PeakCounts/"+detector+"/plots/sim/"+MC_id+"_103keV.png")

    if args["data"]:
        ax.set_title("Data: "+detector, fontsize=9)
        if cuts == False:
            plt.savefig(dir+"/PeakCounts/"+detector+"/plots/data/"+detector+"_103keV_"+energy_filter+"_run"+str(run)+".png")
        else:
            if sigma_cuts ==4:
                plt.savefig(dir+"/PeakCounts/"+detector+"/plots/data/"+detector+"_103keV_cuts_"+energy_filter+"_run"+str(run)+".png")


    #___________Fit single peaks_________________:
    peak_counts = []
    peak_counts_err = []
    peak_ranges =[[50,63]]# , [122, 124], [124, 126], [207,209], [334,336], [661,663]] #Rough by eye
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
        mu_57_guess, sigma_57_guess, a_57_guess = 57.8, 0.5, max(hist_peak)*0.8
        mu_53_guess, sigma_53_guess, a_53_guess = 53, 1.0, max(hist_peak)*0.5
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
            yfit_step = peak_fitting.step(xfit ,mu_59, sigma_59, bkg, s)
            yfit_gaus53 = peak_fitting.gauss(xfit ,mu_53, sigma_53, a_53)
            yfit_gaus57 = peak_fitting.gauss(xfit ,mu_57, sigma_57, a_57)
            yfit_gaus59 = peak_fitting.gauss(xfit ,mu_59, sigma_59, a_59)

            fig, ax = plt.subplots()
            histograms.plot_hist(hist_peak, bins_peak, var=None, show_stats=False, stats_hloc=0.75, stats_vloc=0.85)
            plt.plot(xfit, yfit, label=r'gauss_step: gauss($x,\mu,\sigma,a$)+step($x,\mu,\sigma,bkg,s$)')
            #plt.plot(xfit, yfit_step, "--", label =r'step($x,\mu,\sigma,bkg,s$)')
            #plt.plot(xfit, yfit_gaus53, "--", color='blue', label =r'gaus53($x,\mu,\sigma,a)')
            #plt.plot(xfit, yfit_gaus57, "--", color='green', label =r'gaus57($x,\mu,\sigma,a)')
            plt.plot(xfit, yfit_gaus59, "--", color='red', label =r'gaus59($x,\mu,\sigma,a)')

            plt.xlim(xmin, xmax)
            plt.yscale("log")
            ax.set_ylim([10**-3,10**+7])
            plt.xlabel("Energy (keV)")
            plt.ylabel("Counts")
            plt.legend(loc="lower left", prop={'size': 8.5})

            props = dict(boxstyle='round', alpha=0.5)
            info_str = '\n'.join((r'$a=%.3g \pm %.3g$' % (a_59, a_59_err), r'$\mu=%.3g \pm %.3g$' % (mu_57, mu_57_err), r'$\chi^2/dof=%.2f/%.0f$'%(chi_sq, dof)))
            plt.text(0.02, 0.5, info_str, transform=ax.transAxes, fontsize=8,verticalalignment='top', bbox=props)

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
            plt.savefig(dir+"/PeakCounts/"+detector+"/plots/sim/"+MC_id+"_"+str(peaks[index])+'keV.png')

        if args["data"]:
            ax.set_title("Data: "+detector, fontsize=9)
            if cuts == False:
                plt.savefig(dir+"/PeakCounts/"+detector+"/plots/data/"+detector+"_"+str(peaks[index])+'keV_'+energy_filter+'_run'+str(run)+'.png')
            else:
                if sigma_cuts ==4:
                    plt.savefig(dir+"/PeakCounts/"+detector+"/plots/data/"+detector+"_"+str(peaks[index])+'keV_cuts_'+energy_filter+'_run'+str(run)+'.png')



    #Comput count ratio O_Am241
    print("")
    C_60, C_60_err = peak_counts[0], peak_counts_err[0]
    if (C_60 == np.nan) or (C_99_103 == np.nan):
        O_Am241, O_Am241_err = np.nan, np.nan
    else:
        O_Am241 = C_60/C_99_103
        O_Am241_err = np.sqrt((C_60_err/C_99_103)**2 + (C_60*C_99_103_err/C_99_103**2)**2)
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
        with open(dir+"/PeakCounts/"+detector+"/PeakCounts_sim_"+MC_id+".json", "w") as outfile:
            json.dump(PeakCounts, outfile, indent=4)
    if args["data"]:
        if cuts == False:
            with open(dir+"/PeakCounts/"+detector+"/PeakCounts_data_"+detector+"_"+energy_filter+"_run"+str(run)+".json", "w") as outfile:
                json.dump(PeakCounts, outfile, indent=4)
        else:
            if sigma_cuts ==4:
                with open(dir+"/PeakCounts/"+detector+"/PeakCounts_data_"+detector+"_cuts_"+energy_filter+"_run"+str(run)+".json", "w") as outfile:
                    json.dump(PeakCounts, outfile, indent=4)
            else:
                with open(dir+"/PeakCounts/"+detector+"/PeakCounts_data_"+detector+"_cuts_"+energy_filter+"_run"+str(run)+"_"+str(sigma_cuts)+"sigma.json", "w") as outfile:
                    json.dump(PeakCounts, outfile, indent=4)




def read_all_dsp_lh5(t2_folder, cuts, cut_file_path=None, run="all", sigma=4):

    sto = lh5.Store()
    files = os.listdir(t2_folder)
    files = fnmatch.filter(files, "*lh5")
    if run == 1:
        files = fnmatch.filter(files, "*run0001*")
    if run == 2:
        files = fnmatch.filter(files, "*run0002*")

    df_list = []
    if cuts == False:
        for file in files:

            #get data, no cuts
            tb = sto.read_object("raw",t2_folder+file)[0]
            df = lh5.Table.get_dataframe(tb)
            df_list.append(df)

        df_total = pd.concat(df_list, axis=0, ignore_index=True)
        return df_total

    else: #apply cuts
        files = [t2_folder+file for file in files] #get list of full paths
        lh5_group = "raw"
        df_total_cuts, failed_cuts = cut.load_df_with_cuts(files, lh5_group, cut_file = cut_file_path, cut_parameters= {'bl_mean':sigma,'bl_std':sigma, 'pz_std':sigma}, verbose=True)

        return df_total_cuts

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
    integral_list = quad(peak_fitting.gauss,0, 120, args=(mu,sigma,a))
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

if __name__ =="__main__":
    main()
