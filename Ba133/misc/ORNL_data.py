import cProfile
import pygama.lh5 as lh5
import os,json
import matplotlib.pyplot as plt
import numpy as np
import fnmatch
import pandas as pd
import pygama.genpar_tmp.cuts as cts
import pygama.analysis.calibration as cal
import pygama.analysis.peak_fitting as pgp
import pygama.analysis.histograms as pgh
import pygama.lh5.store
from pygama.dsp.WaveformBrowser import WaveformBrowser
from pygama.analysis import histograms

import pygama.genpar_tmp.cuts as cut

def main():

    #details:
    # det      = 'V01403A'
    det = "V01406A"
    datatype = 'ba133/'
    # runs = np.arange(1,92,1) #runs 0001-0091 from z = 220 mm to z = 310 mm - runs [0001 - 0091]
    # z_positions = 220*np.ones(91) + runs*((310-220)/90)

    #get tier 2 file list
    datapath_t2   = '/lfs/l1/legend/users/aalexander/ORNL_detchar_data/'+det+'/'+datatype
    files_t2      = os.listdir(datapath_t2)
    files_t2.sort()

    #test file - position 2 files
    # https://elog.legend-exp.org/ORNL/18
    # https://elog.legend-exp.org/ORNL/26

    if det == "V01403A":

        #====== CROS TOP SCANS ========
        run = "71" #x-y cross top scans
        cycs1 = list(range(580,585,1)) #position 1, left of centre
        cycs2 = list(range(585,590,1)) 
        cycs3 = list(range(590,595,1)) 
        cycs4 = list(range(595,600,1)) # centre?
        cycs5 = list(range(600,605,1)) #right of centre
        cycs6 = list(range(605,610,1)) 
        cycs7 = list(range(610,615,1)) 
        cycs8 = list(range(615,625,1)) #above centre
        cycs9 = list(range(620,625,1)) 
        cycs10 = list(range(625,630,1)) 
        cycs11 = list(range(630,635,1)) #centre
        cycs12 = list(range(635,640,1)) #below centre
        cycs13 = list(range(640,645,1)) 
        cycs14 = list(range(645,650,1)) 

        cycs = cycs1 + cycs2 + cycs3 + cycs4 + cycs5 + cycs6 + cycs7 +cycs8 + cycs9 + cycs10 + cycs11 + cycs12 + cycs13 + cycs14

        test_files_t2 = []
        for i,cyc in enumerate(cycs):
            file_i = fnmatch.filter(files_t2, "*cyc"+str(cyc)+"*")
            file_i = os.path.join(datapath_t2,file_i[0])
            test_files_t2.append(file_i)

        #====== CIRCLE TOP SCANS ========
        run = "72" #x-y circle top scans
        cycs1 = list(range(525,530,1)) 
        cycs2 = list(range(530,535,1))  
        cycs3 = list(range(535,540,1)) 
        cycs4 = list(range(540,545,1)) 
        cycs5 = list(range(545,550,1)) 
        cycs6 = list(range(550,555,1)) 
        cycs7 = list(range(555,560,1)) 
        cycs8 = list(range(560,565,1)) 
        cycs9 = list(range(565,570,1))  
        cycs10 = list(range(570,575,1)) 
        cycs11 = list(range(575,580,1))  

        cycs = cycs1 + cycs2 + cycs3 + cycs4 + cycs5 + cycs6 + cycs7 +cycs8 + cycs9 + cycs10 + cycs11 

        for i,cyc in enumerate(cycs):
            file_i = fnmatch.filter(files_t2, "*cyc"+str(cyc)+"*")
            file_i = os.path.join(datapath_t2,file_i[0])
            test_files_t2.append(file_i)

    elif det == "V01406A":

        #RUNS 85 AND 86 only ??
        test_files_t2 = []
        for i, file_i in enumerate(files_t2[10:15]):
            file_i = os.path.join(datapath_t2,file_i)
            test_files_t2.append(file_i)

    print(test_files_t2)

    dfs_uncal = []
    for i, file_t2 in enumerate(test_files_t2):
        sto=lh5.Store()
        # print(sto.ls(file_t2, 'icpc1/dsp/'))
        if det == "V01403A":
            df_uncal = lh5.load_dfs(file_t2, par_list = ["trapEmax", "cuspE", "zacE"], lh5_group='icpcs/icpc1/dsp/', verbose=False)
        elif det == "V01406A":
            df_uncal = lh5.load_dfs(file_t2, par_list = ["trapEmax"], lh5_group='icpc1/dsp', verbose=False)
        dfs_uncal.append(df_uncal)

    df_uncal_total = pd.concat(dfs_uncal, axis=0, ignore_index=True)

    plt.figure()
    plt.title("ORNL - "+det)
    # counts, bins, bars = plt.hist(uncal_data, bins=5000, histtype='step', label=energy_filter)
    counts, bins, bars = plt.hist(df_uncal_total["trapEmax"], bins=10000, histtype='step', label="trapEmax")
    if det == "V01403A":
        counts, bins, bars = plt.hist(df_uncal_total["cuspE"], bins=10000, histtype='step', label="cuspE")
        counts, bins, bars = plt.hist(df_uncal_total["zacE"], bins=10000, histtype='step', label="zacE")
    plt.legend()
    # plt.xlim(0,4000)
    plt.xlabel("adc")
    plt.ylabel("Counts")
    plt.yscale('log')

    energy_filter = "trapEmax"
    uncal_data = df_uncal_total[energy_filter]

    #cut high energy end
    df_uncal_cut = df_uncal_total.loc[(df_uncal_total[energy_filter] < 5000)]
    uncal_data_cut = df_uncal_cut[energy_filter] 

    plt.figure()
    plt.title("ORNL - "+ det)
    counts, bins, bars = plt.hist(uncal_data_cut, bins=5000, histtype='step', label=energy_filter)
    plt.legend()
    # plt.xlim(0,4000)
    plt.xlabel("adc")
    plt.ylabel("Counts")
    plt.yscale('log') 

    #compare with a hades uncal:
    t2_hades_folder = "/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/gen/V08682A/tier2/ba_HS4_top_dlt/"
    df_hades = read_all_dsp_lh5(t2_hades_folder, cuts=False, run=2)
    df_hades_cut = df_hades.loc[(df_hades["trapEmax"] < 5000)]
    plt.figure()
    plt.title("HADES - V08682A")
    plt.hist(df_hades_cut["trapEmax"], bins=5000, histtype='step', label=energy_filter)
    # plt.xlim(0,4000)
    plt.xlabel("adc")
    plt.ylabel("Counts")
    plt.yscale('log')

    print("no.events ORNL: ", str(df_uncal_total.shape))
    print("no.events hades: ", str(df_hades.shape))




    #calibrate 
    glines    = [80.9979,160.61, 223.24, 276.40, 302.85, 356.01, 383.85] # gamma lines used for calibration
    range_keV = [(1,1),(1.5,1.5),(2,2),(2.5,2.5),(3,3),(3,3),(3,3)] # side bands width
    # glines    = [160.61, 223.24, 276.40, 302.85, 356.01, 383.85] # gamma lines used for calibration
    # range_keV = [(1.5,1.5),(2,2),(2.5,2.5),(3,3),(3,3),(3,3)] # side bands width

    guess = 383/(uncal_data.quantile(0.9))
    print(guess)
    idx = np.where(counts == np.max(counts))[0][0]
    max_x = bins[idx]
    guess = 81/max_x
    print(guess)

    try:
        pars, cov, results = cal.hpge_E_calibration(uncal_data_cut,glines,guess,deg=1,range_keV = range_keV, funcs = [pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step],verbose=True)
        # pars, cov, results = cal.hpge_E_calibration(uncal_data,glines,guess,deg=1,range_keV = range_keV, funcs = [pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step,pgp.gauss_step],verbose=True)
        
        print("cal pars: ", pars)

        ecal_pass = pgp.poly(uncal_data, pars)

        xpb = 0.1
        xlo = 0
        xhi = 450

        plt.figure()
        nb = int((xhi-xlo)/xpb)
        hist_pass, bin_edges = np.histogram(ecal_pass, range=(xlo, xhi), bins=nb)
        bins_pass = pgh.get_bin_centers(bin_edges)
        plt.plot(bins_pass, hist_pass, label='QC pass', lw=1, c='b')
        plt.xlabel("Energy (keV)",     ha='right', x=1)
        plt.ylabel("Counts / keV",     ha='right', y=1)
        plt.yscale('log')
        plt.tight_layout()
        plt.legend(loc='upper right')
    except:
        print("could not calibrate")

    plt.show()

    # #get 60keV fit params
    # coef_60keV = results["pk_pars"][0]
    # A_60keV_adc = coef_60keV[0]
    # A_60keV.append(A_60keV_adc)
    # mu_60keV_adc = coef_60keV[1]
    # sigma_60keV_adc = coef_60keV[2]
    # peak_range_adc = [mu_60keV_adc - sigma_60keV_adc, mu_60keV_adc + sigma_60keV_adc]

    # plt.close("all")



    #     df = uncal_pass
    #     df_60keV_t2 = df.loc[(df['cuspEmax_ctc'] > peak_range_adc[0])&(df['cuspEmax_ctc'] < peak_range_adc[1])]
    #     indices_60keV = df_60keV_t2["index"].tolist()
    #     tp_0_est_60keV = df_60keV_t2["tp_0_est"].tolist()

    #     #truncate indices to 3000 values
    #     if len(indices_60keV) > 3000:
    #         indices_60keV = indices_60keV[:3000]
    #         tp_0_est_60keV = tp_0_est_60keV[:3000]
    #     else:
    #         print(file)
    #         print(z)

        
    # print("problem runs:")
    # print(problem_runs)
    # plt.figure()
    # plt.scatter(z_positions, A_60keV)
    # plt.xlabel("z (mm)")
    # plt.ylabel("A_60kev")
    # plt.savefig("A_60keV.png")
    # plt.show()


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


if __name__ =="__main__":
    main()