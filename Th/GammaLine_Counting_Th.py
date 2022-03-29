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
import sys

# import pygama.io.lh5 as lh5
import pygama.lh5 as lh5
from pygama.analysis import histograms
from pygama.analysis import peak_fitting

import pygama.genpar_tmp.cuts as cut

sys.path.insert(1,'/lfs/l1/legend/users/bianca/IC_geometry/analysis/myplot')
from myplot import *
myStyle = True
#Script to fit the gamma lines in the Ba133 spectra, for data and/or MC


def main():

    par = argparse.ArgumentParser(description="fit and count gamma lines in Ba133 spectrum",
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

        df=pd.read_hdf("data_calibration/"+detector+"/calibrated_energy_"+detector+"_"+energy_filter+"_run"+str(run)+".hdf5", key='energy')
        print(df.keys())
        energies=df['calib_energy']
        '''
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
        '''
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
    binwidth = 0.1 #keV
    bins = np.arange(0,450,binwidth)
    hist, bins, var = histograms.get_hist(energies, bins=bins)

    #___________Fit single peaks_________________:
    peak_counts = []
    peak_counts_err = []
    peak_ranges = [[2595, 2630]] #,[159,162],[221.5,225],[274,279],[300,306],[381,386.5]] #Rough by eye
    peaks = [2614.5]

    for index, i in enumerate(peak_ranges):

        #prepare histogram
        print(str(peaks[index]), " keV")
        xmin, xmax = i[0], i[1]
        bins_peak = np.arange(xmin,xmax,binwidth)
        hist_peak, bins_peak, var_peak = histograms.get_hist(energies, bins=bins_peak)
        bins_centres_peak = histograms.get_bin_centers(bins_peak)


        #fit function initial guess
        #mu, sigma, n_sig, htail, tau, n_bkg, hstep = peaks[index], 1, 2.61675369e+04, 2261.485421, 6.164781, 1.344282, 61.598749
        mu, sigma, n_sig  = peaks[index], 8.80409552e-01, 2.44480498e+03 #1,  2.61675369e+04
        n_bkg, hstep = 1.14283195e+01, 1.06247713e+01  #1.14283203e+02
        htail,tau = 1.07818564e+01, 6.06127626e-01 #2.63595614e+04
  #2.61675369e+02,  1.47658e+02
        #mu, sigma, n_sig, hstep = peaks[index], 1, 2.61675369e+04,



#0.1,0.1,0.1,0.1 # min(hist_peak), min(hist_peak), min(hist_peak), min(hist_peak)
        gauss_step_guess = [n_sig, mu, sigma, n_bkg, hstep, htail, tau]
        #peak_fitting.gauss_step_guess = [ mu, sigma, n_bkg, hstep] #n_bkg, hstep]
        #bounds = ([0, 0, 0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        bounds = ([0,0,0,0.,0.,0.,0.],[np.inf,np.inf,np.inf, np.inf,np.inf,np.inf,np.inf])
        #fit - gauss step
        try:
            coeff, cov_matrix = peak_fitting.fit_hist(peak_fitting.radford_peak_wrapped, hist_peak, bins_peak, var=None, guess=gauss_step_guess, poissonLL=False, integral=None, method=None, bounds=bounds)
            n_sig_, mu_, sigma_ = coeff[0], coeff[1], coeff[2]
            n_bkg_, h_step_ = coeff[3], coeff[4]
            htail_, tau_ = coeff[5], coeff[6]
            #a, mu, sigma, bkg, s = coeff[0], coeff[1], coeff[2], coeff[3], coeff[4]
            #a_err, mu_err, sigma_err, bkg_err, s_err = np.sqrt(cov_matrix[0][0]), np.sqrt(cov_matrix[1][1]), np.sqrt(cov_matrix[2][2]), np.sqrt(cov_matrix[3][3]), np.sqrt(cov_matrix[4][4])
            a_err, mu_err, sigma_err = np.sqrt(cov_matrix[0][0]), np.sqrt(cov_matrix[1][1]), np.sqrt(cov_matrix[2][2])
            #compute chi sq of fit
            chi_sq, p_value, residuals, dof = chi_sq_calc(bins_centres_peak, hist_peak, np.sqrt(hist_peak), peak_fitting.radford_peak_wrapped, coeff)
            print("r chi sq: ", chi_sq/dof)

            #Counting - integrate gaussian signal part +/- 3 sigma
            #C, C_err = gauss_count(a,mu,sigma, a_err, mu_err,sigma_err, binwidth)
            #print("peak counts = ", str(C)," +/- ", str(C_err))
            #peak_counts.append(C)
            #peak_counts_err.append(C_err)

            #plot
            xfit = np.linspace(xmin, xmax, 1000)
            yfit = peak_fitting.radford_peak_wrapped(xfit, *coeff)
            y_a_tail, y_gauss, y_bkg, y_step, y_tail = peak_fitting.radford_peak_wrapped(xfit, *coeff, components=True)
            print(coeff)
            y_gauss*=y_a_tail
            #yfit_step = peak_fitting.step(xfit,mu_, sigma_, n_bkg_, h_step_)

            p =Plot((12,8),n=1)
            a=p.ax
            #fig, ax = plt.subplots()
            histograms.plot_hist(hist_peak, bins_peak,label="Data", var=None, show_stats=False, stats_hloc=0.75, stats_vloc=0.85)
            plt.plot(xfit, yfit,color='r', label=r'Total Fit Function')
            plt.plot(xfit, y_gauss, "--", color='g', label =r'Gaussian Function')
            plt.plot(xfit, y_step, "--", label =r'Step Function')
            plt.plot(xfit, y_tail, "--", color='magenta', label =r'Tail Function')
            plt.plot(xfit, y_bkg*np.ones(1000), "--", color='purple', label =r'Constant Function')

            a.set_xlim(xmin, xmax)
            a.set_ylim(0.1, 3*10**4)
            a.set_yscale("log")
            a.set_xlabel("Energy [keV]", family='serif')
            a.set_ylabel("Counts", family='serif')
            p.legend(pos="lower left", ncol=3, out=True) #, prop={'size': 8.5})
            p.pretty(grid=False,large=8)
            p.figure(dir+"/PeakCounts/"+detector+"/plots/data/"+detector+"_"+str(peaks[index])+'keV_cuts_'+energy_filter+'_run'+str(run)+'_myplot.png')

            #props = dict(boxstyle='round', alpha=0.5)
            #info_str = '\n'.join((r'$a=%.3g \pm %.3g$' % (n_sig, a_err), r'$\mu=%.3g \pm %.3g$' % (mu, mu_err), r'$\sigma=%.3g \pm %.3g$' % (sigma, sigma_err)))#, r'$bkg=%.3g \pm %.3g$' % (bkg, bkg_err),r'$s=%.3g \pm %.3g$' % (s, s_err), r'$\chi^2/dof=%.2f/%.0f$'%(chi_sq, dof)))
            #plt.text(0.02, 0.8, info_str, transform=ax.transAxes, fontsize=8,verticalalignment='top', bbox=props)

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
            plt.xlabel("Energy [keV]", family='serif')
            plt.ylabel("Counts", family='serif')
            plt.legend(loc="upper left", prop={'size': 8.5})



        #Save fig
        if args["sim"]:
            ax.set_title(MC_id, fontsize=9)
            plt.savefig(dir+"/PeakCounts/"+detector+"/plots/sim/"+MC_id+"_"+str(peaks[index])+'keV.png')

        if args["data"]:
            #ax.set_title("Data: "+detector, fontsize=9)
            if cuts == False:
                plt.savefig(dir+"/PeakCounts/"+detector+"/plots/data/"+detector+"_"+str(peaks[index])+'keV_'+energy_filter+'_run'+str(run)+'.png')
            else:
                print('ciao')
                #plt.savefig(dir+"/PeakCounts/"+detector+"/plots/data/"+detector+"_"+str(peaks[index])+'keV_cuts_'+energy_filter+'_run'+str(run)+'.png')





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

def gauss_amp(x, mu, sigma, a):
    """
    Gaussian with height as a parameter for fwhm etc. args mu sigma, amplitude
    """
    return a * peak_fitting.gauss(x,mu,sigma)

def gauss_tail_with_pdf(x, mu, sigma, a, htail, tau):
    """
    Gaussian with height as a parameter for fwhm etc. args mu sigma, amplitude
    """
    return a * peak_fitting.gauss_tail(x,mu,sigma,htail,tau)

def gauss_step_pdf (x, mu, sigma,n_bkg, hstep):
    return n_bkg*peak_fitting.peak_fitting.gauss_step(x,a,mu,sigma,hstep)

if __name__ =="__main__":
    main()
