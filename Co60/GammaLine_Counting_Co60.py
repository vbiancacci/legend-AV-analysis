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

# import pygama.io.lh5 as lh5
import pygama.lh5 as lh5
from pygama.analysis import histograms
from pygama.analysis import peak_fitting

import pygama.genpar_tmp.cuts as cut

#Script to fit the gamma lines in the Ba133 spectra, for data and/or MC 

dir=os.path.dirname(os.path.realpath(__file__))

def main():

    par = argparse.ArgumentParser(description="fit and count gamma lines in Ba133 spectrum",
                                  usage="python GammaLine_Counting.py [OPERATION: -d -s] [arguments: ]"
    )
    arg, st, sf = par.add_argument, "store_true", "store_false"
    arg("-d", "--data",  nargs=6, help="fit data, usage: python GammaLine_Counting.py --data <detector> <data_path> <calibration> <energy_filter> <cuts> <run>")
    arg("-s", "--sim", nargs=3, help="fit processed simulations, usage: python GammaLine_Counting.py --sim <detector> <sim_path> <MC_id>")
    
    args=vars(par.parse_args())

    #test: python GammaLine_Counting_Co60.py --data B00035B /lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-full_dl-v01/gen/B00035B/tier2/co_HS5_top_dlt/ /lfs/l1/legend/users/aalexander/legend-AV-analysis/Co60/data_calibration/B00035B/calibration_run1_cuts.json cuspEmax_ctc True 1

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
        df =  pd.read_hdf(sim_path, key="procdf")
        energies = df['energy']
 

    #get total pygama histogram
    binwidth = 0.25 #keV
    bins = np.arange(0,1500,binwidth)
    hist, bins, var = histograms.get_hist(energies, bins=bins)


    #___________Fit single peaks_________________:
    peak_counts = []
    peak_counts_err = []
    peak_ranges = [[1165, 1180],[1325,1340]] #Rough by eye
    peaks = [1173, 1332]

    for index, i in enumerate(peak_ranges):

        #prepare histogram
        print(str(peaks[index]), " keV")
        xmin, xmax = i[0], i[1]
        bins_peak = np.arange(xmin,xmax,binwidth)
        hist_peak, bins_peak, var_peak = histograms.get_hist(energies, bins=bins_peak)
        bins_centres_peak = histograms.get_bin_centers(bins_peak)

        
        #fit function initial guess
        mu_guess, sigma_guess, a_guess, bkg_guess, s_guess = peaks[index], 1, max(hist_peak), min(hist_peak), min(hist_peak)
        gauss_step_guess = [a_guess, mu_guess, sigma_guess, bkg_guess, s_guess]
        bounds = ([0, 0, 0, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf, np.inf])
        
        #fit - gauss step
        try:
            coeff, cov_matrix = peak_fitting.fit_hist(peak_fitting.gauss_step, hist_peak, bins_peak, var=None, guess=gauss_step_guess, poissonLL=False, integral=None, method=None, bounds=bounds)
            a, mu, sigma, bkg, s = coeff[0], coeff[1], coeff[2], coeff[3], coeff[4]
            a_err, mu_err, sigma_err, bkg_err, s_err = np.sqrt(cov_matrix[0][0]), np.sqrt(cov_matrix[1][1]), np.sqrt(cov_matrix[2][2]), np.sqrt(cov_matrix[3][3]), np.sqrt(cov_matrix[4][4])
            
            #compute chi sq of fit
            chi_sq, p_value, residuals, dof = chi_sq_calc(bins_centres_peak, hist_peak, np.sqrt(hist_peak), peak_fitting.gauss_step, coeff)
            print("r chi sq: ", chi_sq/dof)

            #Counting - integrate gaussian signal part +/- 3 sigma
            C, C_err = gauss_count(a,mu,sigma, a_err, mu_err,sigma_err, binwidth)
            print("peak counts = ", str(C)," +/- ", str(C_err))
            peak_counts.append(C)
            peak_counts_err.append(C_err)

            #plot
            xfit = np.linspace(xmin, xmax, 1000)
            yfit = peak_fitting.gauss_step(xfit, *coeff)
            yfit_step = peak_fitting.step(xfit,mu, sigma, bkg, s)
            
            fig, ax = plt.subplots()
            histograms.plot_hist(hist_peak, bins_peak, var=None, show_stats=False, stats_hloc=0.75, stats_vloc=0.85)
            plt.plot(xfit, yfit, label=r'gauss_step: gauss($x,\mu,\sigma,a$)+step($x,\mu,\sigma,bkg,s$)')
            plt.plot(xfit, yfit_step, "--", label =r'step($x,\mu,\sigma,bkg,s$)')
            
            plt.xlim(xmin, xmax)
            plt.yscale("log")
            plt.xlabel("Energy (keV)")
            plt.ylabel("Counts")
            plt.legend(loc="upper left", prop={'size': 8.5})

            props = dict(boxstyle='round', alpha=0.5)
            info_str = '\n'.join((r'$a=%.3g \pm %.3g$' % (a, a_err), r'$\mu=%.3g \pm %.3g$' % (mu, mu_err), r'$\sigma=%.3g \pm %.3g$' % (sigma, sigma_err), r'$bkg=%.3g \pm %.3g$' % (bkg, bkg_err),r'$s=%.3g \pm %.3g$' % (s, s_err), r'$\chi^2/dof=%.2f/%.0f$'%(chi_sq, dof)))
            plt.text(0.02, 0.8, info_str, transform=ax.transAxes, fontsize=8,verticalalignment='top', bbox=props) 
        
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
           
        
    #Save count values to json file
    PeakCounts = {
        "C_1173" : peak_counts[0],
        "C_1173_err" : peak_counts_err[0],
        "C_1332" : peak_counts[1],
        "C_1332_err" : peak_counts_err[1]
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
        # df_total_cuts, failed_cuts = cut.load_df_with_cuts(files, lh5_group, cut_file = cut_file_path, cut_parameters= {'bl_mean':sigma,'bl_std':sigma, 'pz_std':sigma}, verbose=True)
        df_total_cuts, failed_cuts = cut.load_df_with_cuts(files, lh5_group, cut_parameters= {'bl_mean':sigma,'bl_std':sigma, 'pz_std':sigma}, verbose=True)
        
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

def gauss_count(a,mu,sigma, a_err, mu_err,sigma_err, bin_width):
    "count/integrate gaussian peak"

    height = a/sigma/np.sqrt(2*np.pi) #gauss= height*np.exp(-(x - mu)**2 / (2. * sigma**2))

    #____analytical integration___ - -inf to +inf
    #integral = height*sigma*np.sqrt(2*np.pi)/bin_width
    #integral_err = integral*np.sqrt((a_err/a)**2 + (sigma_err/sigma)**2)
    integral = a/bin_width
    integral_err = a_err/bin_width

    #_____3sigma_____
    integral_356_3sigma_list = quad(peak_fitting.gauss,mu-3*sigma, mu+3*sigma, args=(mu,sigma,a))
    integral = integral_356_3sigma_list[0]/bin_width
    intergral_err = integral_356_3sigma_list[1]/bin_width

    return integral, integral_err


if __name__ =="__main__":
    main()