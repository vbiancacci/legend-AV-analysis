import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pygama.analysis.histograms as pgh
import pygama.analysis.calibration as cal
import pygama.analysis.peak_fitting as pgp
import json
import fnmatch
import h5py
import pandas as pd

import pygama.lh5 as lh5
import pygama.genpar_tmp.cuts as cut
#Script to calibrate Am241 pygama data and obtain resolution fit coefficients

CodePath=os.path.dirname(os.path.realpath(__file__))

def main():

    if(len(sys.argv) != 7):
        print('Example usage: python Calibration_Am241.py <detector> <data_path> <energy_filter> <cuts> <run>')
        sys.exit()

    detector = sys.argv[1]
    data_path = sys.argv[2]
    energy_filter = sys.argv[3]
    cuts = sys.argv[4]
    run = int(sys.argv[5])
    source = sys.argv[6]

    print("")
    print("detector: ", detector)
    #print("data path: ", data_path)
    print("energy_filter: ", energy_filter)
    print("applying cuts: ", cuts)
    print("data run: ", run)
    print("source: ", source)

    if cuts == "False":
        cuts = False
    else:
        cuts = True

    #initialise directories for detectors to save
    if not os.path.exists(CodePath+"/data_calibration/"+detector+"/"+source+"/plots/"):
        os.makedirs(CodePath+"/data_calibration/"+detector+"/"+source+"/plots/")

    #====Load data======
    print(" ")
    print("Loading data")
    if cuts == False:
        df_total_lh5 = read_all_dsp_lh5(data_path,cuts,run=run)
        failed_cuts = np.zeros(len(df_total_lh5[energy_filter]))
    else:
        sigma_cuts = 4
        print("sigma cuts: ", str(sigma_cuts))
        df_total_lh5, failed_cuts = read_all_dsp_lh5(data_path,cuts,run=run, sigma=sigma_cuts)
        failed_cuts = failed_cuts[energy_filter]

    print("df_total_lh5: ", df_total_lh5)
    energy_filter_data = df_total_lh5[energy_filter]
    '''
    energy_load_data=energy_filter_data.to_frame(name='energy_filter')
    energy_load_failed=failed_cuts.to_frame(name='failed_cuts')

    output_file=CodePath+"/data_calibration/"+detector+"/"+source+"/loaded_energy_"+detector+"_"+energy_filter+"_run"+str(run)+".hdf5"

    energy_load_data.to_hdf(output_file, key='energy', mode='w')
    energy_load_failed.to_hdf(output_file, key='failed')
    '''

    #========Compute calibration coefficients===========

    print("Calibrating...")
    #df=pd.read_hdf(CodePath+"/data_calibration/"+detector+"/"+source+"/loaded_energy_"+detector+"_"+energy_filter+"_run"+str(run)+".hdf5", key='energy')
    #energy_filter_data=df['energy_filter']

    glines=[59.5409, 98.97,102.98,123.05]
    range_keV=[(3., 3.),(1.5,1.5),(1.5,1.5),(1.5,1.5)]

    #guess = 2621./(energy_filter_data.quantile(0.99))
    #guess=102.9/(np.nanpercentile(energy_filter_data,0.99))
    #print(np.nanpercentile(energy_filter_data,0.99))
    #print(energy_filter_data.quantile(0.99))
    #print(guess)
    if detector=='V02160A' or detector=='V05268A':
        guess= 0.1#0.045#0.1 #V02160A #0.057   0.032 if V07647A  #0.065 7298B
    elif detector=='V05266A':
        guess=0.08
    elif detector=='V05267B':
        guess=0.07
    elif detector=='V04545A':
        guess=0.07
    #elif detector=='V04549B': #only for am_HS&
    #    guess=0.1
    else:
        guess=0.045

    print("Find peaks and compute calibration curve...",end=' ')
    pars, cov, results = cal.hpge_E_calibration(energy_filter_data, glines, guess, deg=1, range_keV = range_keV, funcs = [pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step],verbose=True)
    print("cal pars: ", pars)


    #======Plot calibrated energy=======

    ecal_pass = pgp.poly(energy_filter_data, pars)
    ecal_cut  = pgp.poly(failed_cuts,  pars)

    xpb = 0.1
    xlo = 0
    xhi = 120

    nb = int((xhi-xlo)/xpb)
    hist_pass, bin_edges = np.histogram(ecal_pass, range=(xlo, xhi), bins=nb)
    bins_pass = pgh.get_bin_centers(bin_edges)
    hist_cut, bin_edges = np.histogram(ecal_cut, range=(xlo, xhi), bins=nb)
    bins_cut = pgh.get_bin_centers(bin_edges)
    plot_title = detector+' - run '+str(run)
    plt.plot(bins_pass, hist_pass, label='QC pass', lw=1, c='b')
    plt.plot(bins_cut,  hist_cut,  label='QC fail', lw=1, c='r')
    plt.plot(bins_cut,  hist_cut+hist_pass,  label='no QC', lw=1)

    plt.xlabel("Energy (keV)",     ha='right', x=1)
    plt.ylabel("Counts / keV",     ha='right', y=1)
    plt.title(plot_title)
    plt.yscale('log')
    plt.tight_layout()
    plt.legend(loc='upper right')

    plt.savefig(CodePath+"/data_calibration/"+detector+"/"+source+"/plots/calibrated_energy_"+energy_filter+"_run"+str(run)+".png")

    #=========Plot Calibration Curve===========

    fitted_peaks = results['fitted_keV']
    pk_pars      = results['pk_pars']
    mus          = np.array([pars[1] for pars in pk_pars]).astype(float)
    fwhms        = np.array([pars[2] for pars in pk_pars]).astype(float)*pars[0]*2.*math.sqrt(2.*math.log(2.))
    pk_covs      = results['pk_covs']
    dfwhms       = np.array([], dtype=np.float32)
    for i,covsi in enumerate(pk_covs):
        covsi    = np.asarray(covsi, dtype=float)
        parsigsi = np.sqrt(covsi.diagonal())
        dfwhms   = np.append(dfwhms,parsigsi[2]*pars[0]*2.*math.sqrt(2.*math.log(2.)))

    fwhm_peaks   = np.array([], dtype=np.float32)
    for i,peak in enumerate(fitted_peaks):
        fwhm_peaks = np.append(fwhm_peaks,peak)


    param_guess  = [0.2,0.001,0.000001]
    param_bounds = (0, [10., 1. ,0.1])
    fit_pars, fit_covs = curve_fit(fwhm_slope, fwhm_peaks, fwhms, sigma=dfwhms, p0=param_guess, bounds=param_bounds)
    print('FWHM curve fit: ',fit_pars)
    fit_vals = fwhm_slope(fwhm_peaks,fit_pars[0],fit_pars[1],fit_pars[2])
    print('FWHM fit values: ',fit_vals)
    fit_qbb = fwhm_slope(2039.0,fit_pars[0],fit_pars[1],fit_pars[2])
    print('FWHM energy resolution at Qbb: %1.2f keV' % fit_qbb)


    fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    ax1.errorbar(fwhm_peaks,fwhms,yerr=dfwhms, marker='o',lw=0, c='b')
    ax1.plot(fwhm_peaks,fit_vals,lw=1, c='g')
    ax1.set_ylabel("FWHM energy resolution (keV)", ha='right', y=1)

    ax2.plot(fitted_peaks,pgp.poly(mus, pars)-fitted_peaks, lw=1, c='b')
    ax2.set_xlabel("Energy (keV)",    ha='right', x=1)
    ax2.set_ylabel("Residuals (keV)", ha='right', y=1)

    fig.suptitle(plot_title)

    plt.savefig(CodePath+"/data_calibration/"+detector+"/"+source+"/plots/calibration_curve_"+energy_filter+"_run"+str(run)+".png")

    #=========Save Calibration Coefficients==========
    dict = {energy_filter: {"resolution": list(fit_pars), "calibration": list(pars)}}
    print(dict)
    if cuts == False:
        with open(CodePath+"/data_calibration/"+detector+"/"+source+"/calibration_run"+str(run)+".json", "w") as outfile:
            json.dump(dict, outfile, indent=4)
    else:
        with open(CodePath+"/data_calibration/"+detector+"/"+source+"/calibration_run"+str(run)+"_cuts.json", "w") as outfile:
            json.dump(dict, outfile, indent=4)

    print("done")
    print("")

    #===================Store calibrated data =====================
    energy_calib_data=ecal_pass.to_frame(name='energy_filter')
    energy_calib_failed=ecal_cut.to_frame(name='failed_cuts')

    output_file=CodePath+"/data_calibration/"+detector+"/"+source+"/loaded_energy_"+detector+"_"+energy_filter+"_run"+str(run)+".hdf5"

    energy_calib_data.to_hdf(output_file, key='energy', mode='w')
    energy_calib_failed.to_hdf(output_file, key='failed')



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

        return df_total_cuts, failed_cuts


def fwhm_slope(x, m0, m1, m2):
    """
    Fit the energy resolution curve
    """
    return np.sqrt(m0 + m1*x +(m2*(x**2)))


if __name__ == "__main__":
    main()
