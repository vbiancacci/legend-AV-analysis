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
     
    energy_load_data=energy_filter_data.to_frame(name='energy_filter')
    energy_load_failed=failed_cuts.to_frame(name='failed_cuts')

    output_file=CodePath+"/data_calibration/"+detector+"/"+source+"/loaded_energy_"+detector+"_"+energy_filter+"_run"+str(run)+"_test.hdf5"

    energy_load_data.to_hdf(output_file, key='energy', mode='w')
    energy_load_failed.to_hdf(output_file, key='failed')
        
    #========Compute calibration coefficients===========
    
    print("Calibrating...")
    #df=pd.read_hdf(CodePath+"/data_calibration/"+detector+"/"+source+"/loaded_energy_"+detector+"_"+energy_filter+"_run"+str(run)+"_test.hdf5", key='energy')
    #energy_filter_data=df['energy_filter']
    
    glines=[59.5409, 98.97, 102.98 ,123.05]
    range_keV=[(3., 3.),(1.5,1.5),(1.5,1.5),(1.5,1.5)]

    #V07646A=0.053 for am_HS6, for 0.045 for am_HS1    
    #V08682A =0.07 for am_HS6, 0.053 for am_HS1
    #V08682B =0.053 for am_HS6, 0.03 for am_HS1
    #V09372A = 0.1 for am_HS6, 0.07 for am_HS1
    #V09374A = 0.07 for am_HS6, 0.053 for am_HS1
    #V09724A = 0.1 for am_HS6, 0.045 for am_HS1

    if detector=='V02160A' or detector=='V05268A'   or detector=='B00035A' or detector=='V09724A': # or detector=='B00032B':
        guess= 0.1#0.045#0.1 #V02160A #0.057   0.032 if V07647A  #0.065 7298B
    #elif detector=='V05266A' or  detector=="B00061C":
    #    guess=0.08
    #elif detector=='V05267B'or detector=='V04545A'or detector=='V09372A' or detector=="B00035B" or detector=='B00002C' or detector=='V08682A':
    #    guess=0.07
    #elif detector=='V08682B'or detector=='V09372A'  or detector=='B00076C' or detector=='B00032B' or detector=='V07646A': #32B and 91B without cut
    #    guess=0.053
    #elif detector=='V04549B' or detector=='V09372A': #only for am_HS6
    #    guess=0.1
    elif detector=='V07646A': 
       guess=0.035
    #elif detector=='B00091B' or detector=='V09374A' :
    #    guess=0.07
    else:
        guess=0.045
    
    print("Find peaks and compute calibration curve...",end=' ')
    try:
        pars, cov, results = cal.hpge_E_calibration(energy_filter_data, glines, guess, deg=1, range_keV = range_keV, funcs = [pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step],verbose=True)
        print("cal pars: ", pars)
        ''' 
        
#======Plot calibrated energy=======
    try:
        calibration_ = "/lfs/l1/legend/users/bianca/IC_geometry/analysis/post-proc-python/second_fork/legend-AV-analysis/Am241/data_calibration/"+detector+"/am_HS6/calibration_run1_cuts.json"
        with open(calibration_) as json_file:
            calibration_coefs = json.load(json_file)
    # print(calibration_coefs)
        m_Th = calibration_coefs[energy_filter]["calibration"][0]#["Calibration_pars"][0]
        c_Th = calibration_coefs[energy_filter]["calibration"][1]#["Calibration_pars"][1]
        a_Th = calibration_coefs[energy_filter]["resolution"][0]#["m0"]
        b_Th = calibration_coefs[energy_filter]["resolution"][1]#["m1"]

        pars = [m_Th, c_Th]
        print("cal pars: ", pars)
        resolution_pars = [a_Th, b_Th]
        print("res pars: ", resolution_pars)
        ''' 
        ecal_pass = pgp.poly(energy_filter_data, pars)
        ecal_cut  = pgp.poly(failed_cuts,  pars)
        calib_pars=True
    
    except (IndexError, TypeError):
        calibration="/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-full_dl-v01/pargen/dsp_ecal/"+detector+".json"
        #calibration="/lfs/l1/legend/users/bianca/IC_geometry/analysis/post-proc-python/second_fork/legend-AV-analysis/Am241/data_calibration/V05261A/"+detector+".json"
        print("calibration with Th file:", calibration)
        with open(calibration) as json_file:
            calibration_coefs = json.load(json_file)
        m = calibration_coefs[energy_filter]["Calibration_pars"][0]
        c = calibration_coefs[energy_filter]["Calibration_pars"][1]
        ecal_pass = energy_filter_data*m + c
        ecal_cut = failed_cuts*m+c
        calib_pars=False

    xpb = 0.1
    xlo = 0
    xhi =720
    nb = int((xhi-xlo)/xpb)
    fig, ax1 = plt.subplots(figsize=(12,8))
    hist_pass, bin_edges = np.histogram(ecal_pass, range=(xlo, xhi), bins=nb)
    bins_pass = pgh.get_bin_centers(bin_edges)
    hist_cut, bin_edges = np.histogram(ecal_cut, range=(xlo, xhi), bins=nb)
    bins_cut = pgh.get_bin_centers(bin_edges)
    plot_title = detector+' - run '+str(run)
    ax1.plot(bins_pass, hist_pass,lw=1, label='QC pass', c='b')
    plt.plot(bins_cut,  hist_cut,  label='QC fail', lw=1, c='r')
    plt.plot(bins_cut,  hist_cut+hist_pass,  label='no QC', lw=1)

    ax1.set_xlabel("Energy [keV]", fontsize=30)
    ax1.set_ylabel("Counts / 0.1keV", fontsize=30)
    ax1.set_ylim(1,10**6)
    ax1.margins(x=0)
    ax1.tick_params(axis="both", labelsize=25)    
    plt.title(plot_title)
    ax1.set_yscale('log')

    plt.legend(loc='lower left', fontsize=25)

    #left, bottom, width, height = [0.3, 0.22, 0.3, 0.3]
    left, bottom, width, height = [0.6, 0.63, 0.3, 0.3]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.set_xlabel("Energy [keV]", fontsize=22)
    ax2.set_ylabel("Counts / 0.1keV", fontsize=22)
    ax2.set_yscale('log')
    ax2.margins(x=0)
    ax2.tick_params(axis="both", labelsize=20)
    hist_pass_tiny, bin_edges_tiny = np.histogram(ecal_pass, range=(25, 120), bins=950)
    bins_pass_tiny = pgh.get_bin_centers(bin_edges_tiny)
    ax2.plot(bins_pass_tiny, hist_pass_tiny,c='b',lw=1)

    plt.tight_layout()

    if cuts == True:
        plt.savefig(CodePath+"/data_calibration/"+detector+"/"+source+"/plots/calibrated_energy_"+energy_filter+"_run"+str(run)+"_test.pdf")
    else:
        plt.savefig(CodePath+"/data_calibration/"+detector+"/"+source+"/plots/calibrated_energy_"+energy_filter+"_nocuts_run"+str(run)+"_test.png")
     
    #=========Plot Calibration Curve===========
    if calib_pars==True:
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

        if cuts==True:
            plt.savefig(CodePath+"/data_calibration/"+detector+"/"+source+"/plots/calibration_curve_"+energy_filter+"_run"+str(run)+"_test.png")
        else:
            plt.savefig(CodePath+"/data_calibration/"+detector+"/"+source+"/plots/calibration_curve_"+energy_filter+"_nocuts_run"+str(run)+"_test.png")

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
   
    
    '''    
    #===================Store calibrated data =====================
    energy_calib_data=ecal_pass.to_frame(name='energy_filter')
    energy_calib_failed=ecal_cut.to_frame(name='failed_cuts')

    output_file=CodePath+"/data_calibration/"+detector+"/"+source+"/loaded_energy_"+detector+"_"+energy_filter+"_run"+str(run)+".hdf5"

    energy_calib_data.to_hdf(output_file, key='energy', mode='w')
    energy_calib_failed.to_hdf(output_file, key='failed', mode='w')
    
    
    if cuts== True :
        output_file=CodePath+"/data_calibration/"+detector+"/"+source+"/loaded_energy_"+detector+"_"+energy_filter+"_run"+str(run)+".hdf5"
    else:
        output_file=CodePath+"/data_calibration/"+detector+"/"+source+"/loaded_energy_"+detector+"_"+energy_filter+"_nocuts_run"+str(run)+".hdf5"
    energy_calib_data=ecal_pass.to_frame(name='energy_filter')
    energy_calib_data.to_hdf(output_file, key='energy', mode='w')

    #if cuts==True:
    #    energy_calib_failed=ecal_cut.to_frame(name='failed_cuts')
    #    energy_calib_failed.to_hdf(output_file, key='failed', mode='w')
    '''


def read_all_dsp_lh5(t2_folder, cuts, cut_file_path=None, run="all", sigma=4):

    sto = lh5.Store()
    files = os.listdir(t2_folder)
    files = fnmatch.filter(files, "*lh5")
    if run == 1:
        files = fnmatch.filter(files, "*run0001*")
    if run == 2:
        files = fnmatch.filter(files, "*run0002*")
    if run == 3:
        files = fnmatch.filter(files, "*run0003*")
    if run == 4:
        files = fnmatch.filter(files, "*run0004*")

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
