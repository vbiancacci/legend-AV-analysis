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
import pandas as pd

# import pygama.io.lh5 as lh5
import pygama.lh5 as lh5
import pygama.genpar_tmp.cuts as cut

#Script to calibrate Ba133 pygama data and obtain resolution fit coefficients

CodePath=os.path.dirname(os.path.realpath(__file__))

def main():

    if(len(sys.argv) !=7 ):
        print('Example usage: python Ba133_Calibration.py <detector> <data_path> <energy_filter> <cuts> <run>')
        sys.exit()

    detector = sys.argv[1]
    data_path = sys.argv[2]
    energy_filter = sys.argv[3]
    cuts = sys.argv[4]
    run = int(sys.argv[5])
    source = sys.argv[6]

    print("")
    print("detector: ", detector)
    print("data path: ", data_path)
    print("energy_filter: ", energy_filter)
    print("applying cuts: ", cuts)
    print("data run: ", run)
    print("source: ", source)


    #calibration="/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/genpar/dsp_ecal/"+detector+".json"
    #calibration="/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/genpar/dsp_ecal/I02160A.json"
    calibration = CodePath+"/data_calibration/"+detector+"/calibration_run"+str(run)+"_cuts.json"

    if cuts == "False":
        cuts = False
    else:
        cuts = True

    #initialise directories for detectors to save
    if not os.path.exists(CodePath+"/data_calibration/"+detector+"/plots/"):
        os.makedirs(CodePath+"/data_calibration/"+detector+"/plots/")
    '''
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

    # plt.hist(energy_filter_data, bins=5000)
    # plt.show()

    energy_load_data=energy_filter_data.to_frame(name='energy_filter')
    energy_load_failed=failed_cuts.to_frame(name='failed_cuts')

    output_file=CodePath+"/data_calibration/"+detector+"/loaded_energy_"+detector+"_"+energy_filter+"_run"+str(run)+".hdf5"

    energy_load_data.to_hdf(output_file, key='energy', mode='w')
    energy_load_failed.to_hdf(output_file, key='failed')
    '''


    #========Compute calibration coefficients===========
    print("Calibrating...")
    df=pd.read_hdf(CodePath+"/data_calibration/"+detector+"/loaded_energy_"+detector+"_"+energy_filter+"_run"+str(run)+".hdf5", key='energy')
    energy_filter_data=df['energy_filter']
    glines    = [300.089, 1592.511, 1620.738, 2103.511, 2614.511] # gamma lines used for calibration
    range_keV = [(2,2),(2,2),(2,2),(2,2),(4,4)] # side bands width
    print("gline")
    #glines    = [238.632, 300.089, 510.74, 583.187, 727.330, 763.45, 785.37, 860.53, 893.408, 1078.63, 1512.70, 1592.511, 1620.738, 2103.511, 2614.511] # gamma lines used for calibration
    #range_keV = [(1,1),(1.5,1.5),(2,2),(2,2),(2,2),(2,2),(2,2),(2,2),(2,2),(2,2),(2,2),(2.5,2.5),(3,3),(3,3),(5,5)] # side bands width
    # glines    = [160.61, 223.24, 276.40, 302.85, 356.01, 383.85] # gamma lines used for calibration
    # range_keV = [(2,2),(3,3),(4,4),(4,4),(4,4),(4,4)] # side bands width

    guess = 0.05# 2614.5/(energy_filter_data.quantile(0.9))#2614.5/45403. #old Th guess
    print(guess)
    # print(energy_filter_data.quantile(0.9))
    #guess = 383/(energy_filter_data.quantile(0.9))

    print("Find peaks and compute calibration curve...")#,end=' ')

    pars, cov, results = cal.hpge_E_calibration(energy_filter_data,glines,guess,deg=1,range_keV = range_keV,funcs = [pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step],verbose=True)
    # pars, cov, results = cal.hpge_E_calibration(energy_filter_data,glines,guess,deg=1,range_keV = range_keV,funcs = [pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step],verbose=False)
    print("cal pars: ", pars)

    #======Plot calibrated energy=======

    #Get data and concoatonate into df
    #if cuts == False:
    #    df_total_lh5 = read_all_dsp_lh5(data_path,cuts,run=run)
    #else:
#        sigma_cuts = 4
#        print("sigma cuts: ", str(sigma_cuts))
        #df_total_lh5 = read_all_dsp_lh5(data_path,cuts,run=run, sigma=sigma_cuts)


    #print("df_total_lh5: ", df_total_lh5)
    df=pd.read_hdf(CodePath+"/data_calibration/"+detector+"/loaded_energy_"+detector+"_"+energy_filter+"_run"+str(run)+".hdf5", key='energy')
    energy_filter_data=df['energy_filter']
    #energy_filter_data = df_total_lh5[energy_filter]

    #Get Calibration
    with open(calibration) as json_file:
        calibration_coefs = json.load(json_file)
    m = calibration_coefs[energy_filter]["calibration"][0] #["Calibration_pars"][0]
    c = calibration_coefs[energy_filter]["calibration"][0] #["Calibration_pars"][1]

    #m=0.0634603499212275 #for PhysicsList
    #c=-0.611497784803706# for PhysicsList

    xpb = 1
    xlo = 0
    xhi = 3000

    nb = int((xhi-xlo)/xpb)
    # energies = (energy_filter_data-c)/m
    energies = energy_filter_data*m + c
    hist_pass, bin_edges = np.histogram(energies, range=(xlo, xhi), bins=nb)
    bins_pass = pgh.get_bin_centers(bin_edges)
    plt.plot(bins_pass, hist_pass, label='QC pass', lw=1, c='b')
    plot_title = detector+' - run '+str(run)

    plt.xlabel("Energy (keV)",     ha='right', x=1)
    plt.ylabel("Counts / keV",     ha='right', y=1)
    plt.title(plot_title)
    plt.yscale('log')
    plt.tight_layout()
    plt.legend(loc='upper right')

    plt.savefig(CodePath+"/data_calibration/"+detector+"/plots/calibrated_energy_"+energy_filter+"_run"+str(run)+".png")

    output_file=CodePath+"/data_calibration/"+detector+"/calibrated_energy_"+detector+"_"+energy_filter+"_run"+str(run)+".hdf5"
    energies_df=energies.to_frame(name='calib_energy')
    energies_df.to_hdf(output_file, key='energy', mode='w')


    ecal_pass = pgp.poly(energy_filter_data, pars)
    #ecal_cut  = pgp.poly(failed_cuts,  pars)

    xpb = 0.1
    xlo = 0
    xhi = 3000

    nb = int((xhi-xlo)/xpb)
    hist_pass, bin_edges = np.histogram(ecal_pass, range=(xlo, xhi), bins=nb)
    bins_pass = pgh.get_bin_centers(bin_edges)
    #hist_cut, bin_edges = np.histogram(ecal_cut, range=(xlo, xhi), bins=nb)
    bins_cut = pgh.get_bin_centers(bin_edges)
    plot_title = detector+' - run '+str(run)
    plt.plot(bins_pass, hist_pass, label='QC pass', lw=1, c='b')
    #plt.plot(bins_cut,  hist_cut,  label='QC fail', lw=1, c='r')
    #plt.plot(bins_cut,  hist_cut+hist_pass,  label='no QC', lw=1)

    plt.xlabel("Energy (keV)",     ha='right', x=1)
    plt.ylabel("Counts / keV",     ha='right', y=1)
    plt.title(plot_title)
    plt.yscale('log')
    plt.tight_layout()
    plt.legend(loc='upper right')

    plt.savefig(CodePath+"/data_calibration/"+detector+"/plots/calibrated_energy_"+energy_filter+"_run"+str(run)+".png")



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
    # ax1.plot(qbb_line_hx,qbb_line_hy,lw=1, c='r')
    # ax1.plot(qbb_line_vx,qbb_line_vy,lw=1, c='r')
    ax1.set_ylabel("FWHM energy resolution (keV)", ha='right', y=1)

    ax2.plot(fitted_peaks,pgp.poly(mus, pars)-fitted_peaks, lw=1, c='b')
    ax2.set_xlabel("Energy (keV)",    ha='right', x=1)
    ax2.set_ylabel("Residuals (keV)", ha='right', y=1)

    fig.suptitle(plot_title)

    plt.savefig(CodePath+"/data_calibration/"+detector+"/plots/calibration_curve_"+energy_filter+"_run"+str(run)+".png")

    #=========Save Calibration Coefficients==========
    dict = {energy_filter: {"resolution": list(fit_pars), "calibration": list(pars)}}
    print(dict)
    if cuts == False:
        with open(CodePath+"/data_calibration/"+detector+"/calibration_run"+str(run)+".json", "w") as outfile:
            json.dump(dict, outfile, indent=4)
    else:
        with open(CodePath+"/data_calibration/"+detector+"/calibration_run"+str(run)+"_cuts.json", "w") as outfile:
            json.dump(dict, outfile, indent=4)

    print("done")
    print("")

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

        return df_total_cuts, failed_cuts

def fwhm_slope(x, m0, m1, m2):
    """
    Fit the energy resolution curve
    """
    return np.sqrt(m0 + m1*x +(m2*(x**2)))


if __name__ == "__main__":
    main()
