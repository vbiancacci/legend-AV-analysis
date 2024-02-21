import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime
import json
import argparse
import os
from scipy import optimize
from scipy import stats
import fnmatch
import h5py


sys.path.insert(1,'/lfs/l1/legend/users/bianca/IC_geometry/analysis/myplot')
from myplot import *

myStyle = True


#Script to plot spectra of MC (best fit FCCD) and data

def main():
    all = True
    isotopes = False
    lists  = False
    scatterplot = False

    if(len(sys.argv) != 8):
        print('Example usage: python AV_postproc.py <detector> <MC_id> <sim_path> <FCCD> <DLF> <data_path> <calibration> <cuts> <run>')
        sys.exit()

    detector = sys.argv[1] #raw MC folder path
    data_path = sys.argv[2] #path to data
    calibration = sys.argv[3] #path to data calibration
    energy_filter = sys.argv[4] #energy filter - e.g. trapEftp
    cuts = sys.argv[5] #e.g. False
    run = int(sys.argv[6]) #data run, e.g. 1 or 2

    print("detector: ", detector)
    print("data_path: ", data_path)
    print("calibration: ", calibration)
    print("energy_filter: ", energy_filter)
    print("applying data cuts: ", cuts)
    print("run: ", run)


    CodePath=os.path.dirname(os.path.realpath(__file__))
    DLF=1.0
    smear="g"
    frac_FCCDbore=0.5
    TL_model="notl"
    FCCD = 0.0
    sim_path="/lfs/l1/legend/detector_char/enr/hades/simulations/legend-g4simple-simulation/simulations/"+detector+"/th_HS2/top_0r_38z/hdf5/AV_processed/"
    MC_id=detector+"-th_HS2-top-0r-38z"+"_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_fracFCCDbore"+str(frac_FCCDbore)
    isotope=".hdf5"

    dir=os.path.dirname(os.path.realpath(__file__))
    print("working directory: ", dir)

    #initialise directories to save spectra
    if not os.path.exists(dir+"/Spectra/"+detector+"/"):
        os.makedirs(dir+"/Spectra/"+detector+"/")

    if all:
        print("start...")

        #GET DATA
        #Get data and concoatonate into df
        df=pd.read_hdf(CodePath+"/data_calibration/"+detector+"/calibrated_energy_"+detector+"_"+energy_filter+"_run"+str(run)+".hdf5", key='energy')
        print(df.keys())
        energy_data=df['calib_energy']

        #GET MC
        df =  pd.read_hdf(sim_path+MC_id+isotope, key="procdf")
        energy_MC = df['energy']
        print("opened MC")

        #Plot data and scaled MC
        binwidth = 1 #keV
        bins = np.arange(0,2700,binwidth)

        p =Plot((12,8),n=1)
        ax0=p.ax

        counts_data, bins, bars_data = ax0.hist(energy_data, bins=bins,  label = "Data", histtype = 'step', linewidth = '0.55')
        counts_MC, bins = np.histogram(energy_MC, bins = bins)

        C_2100_data = area_between_vals(counts_data, bins, 1000, 2200)
        C_2100_MC = area_between_vals(counts_MC, bins, 1000, 2200)
        print(C_2100_MC, C_2100_data)

        scaled_counts_MC, bins, bars = ax0.hist(energy_MC, bins = bins,  label = "g4simple simulation", weights=(C_2100_data/C_2100_MC)*np.ones_like(energy_MC), histtype = 'step', linewidth = '0.55')


        print("basic histos complete")

        #counts_res, bins, bars_res

        Data_MC_ratios = []
        Data_MC_ratios_err = []
        for index, bin in enumerate(bins[1:]):
            data = counts_data[index]
            MC = scaled_counts_MC[index] #This counts has already been scaled by weights
            if MC == 0:
                ratio = 0.
                error = 0.
            else:
                residual = data-MC
            Data_MC_ratios.append(residual)

        print("errors")

        #ax1.errorbar(bins[1:], Data_MC_ratios, yerr=0,color="green", elinewidth = 1, fmt='x', ms = 1.0, mew = 1.0)
        #ax1.hlines(1, 0, 3000, colors="gray", linestyles='dashed')
        plt.xlabel("Energy [keV]",  family='serif')
        ax0.set_ylabel("Counts", family='serif')
        ax0.set_yscale("log")
        #ax0.set_title(detector, family='serif')

        p.legend(ncol=1, out=False, pos = "lower left")
        p.pretty(large=8, grid=False)
        p.figure(dir+"/Spectra/"+detector+"/DataMC_"+MC_id+"_"+energy_filter+"_run"+str(run)+"_cuts_top.png")


        print("done")


    if isotopes:
        # plot of three isotopes
        p_3 =Plot((12,8),n=1)
        ax0_3=p_3.ax
        binwidth = 1 #keV
        bins = np.arange(0,2700,binwidth)
        #fig = plt.figure()
        #gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        #ax0 = plt.subplot(gs[0])
        #ax1 = plt.subplot(gs[1], sharex = ax0)

        df_tl =  pd.read_hdf(sim_path+MC_id+"_LBE_tl.hdf5", key="procdf")
        df_bi =  pd.read_hdf(sim_path+MC_id+"_LBE_bi.hdf5", key="procdf")
        df_pb =  pd.read_hdf(sim_path+MC_id+"_LBE_pb.hdf5", key="procdf")
        energy_MC_tl = df_tl['energy']
        energy_MC_bi = df_bi['energy']
        energy_MC_pb = df_pb['energy']
        counts_MC_is, bins, bars = ax0_3.hist([energy_MC_tl,energy_MC_bi,energy_MC_pb], bins = bins,  label = ['$^{208}$Tl', '$^{212}$Bi','$^{212}$Pb'], linewidth = '0.55', histtype ='step',color=['blueviolet','mediumpurple','darkslateblue'])
        #counts_MC_bi, bins, bars = ax0_3.hist(energy_MC_bi, bins = bins, histtype = 'step', linewidth = '0.35')
        #counts_MC_pb, bins, bars = ax0_3.hist(energy_MC_pb, bins = bins, histtype = 'step', linewidth = '0.35')
        plt.xlabel("Energy [keV]",  family='serif')
        ax0_3.set_ylabel("Counts", family='serif')
        ax0_3.set_yscale("log")
        #ax0_3.set_title(detector, family='serif')
        p_3.legend(ncol=1, out=False)
        p_3.pretty(large=8, grid=False)
        p_3.figure(dir+"/Spectra/"+detector+"/three_isotopes.eps")



    if lists:
        # plot of three isotopes
        p =Plot((12,8),n=1)
        ax0=p.ax
        binwidth = 1 #keV
        bins = np.arange(0,3000,binwidth)
        #fig = plt.figure()
        #gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        #ax0 = plt.subplot(gs[0])
        #ax1 = plt.subplot(gs[1], sharex = ax0)
        df=pd.read_hdf(CodePath+"/data_calibration/"+detector+"/calibrated_energy_test_"+detector+"_"+energy_filter+"_run"+str(run)+".hdf5", key='energy')
        print(df.keys())
        energy_data=df['calib_energy']

        df_Sh =  pd.read_hdf(sim_path+MC_id+"_Shielding.hdf5", key="procdf")
        df_ShL =  pd.read_hdf(sim_path+MC_id+"_ShieldingLIV.hdf5", key="procdf")
        df_LBE =  pd.read_hdf(sim_path+MC_id+"_LBE.hdf5", key="procdf")
        energy_MC_Sh = df_Sh['energy']
        energy_MC_ShL = df_ShL['energy']
        energy_MC_LBE = df_LBE['energy']

        counts_data, bins = np.histogram(energy_data, bins=bins)
        counts_MC_Sh, bins = np.histogram(energy_MC_Sh, bins = bins)
        counts_MC_ShL, bins = np.histogram(energy_MC_ShL, bins = bins)
        counts_MC_LBE, bins = np.histogram(energy_MC_LBE, bins = bins)

        C_2100_data = area_between_vals(counts_data, bins, 1000, 2200)
        C_2100_MC_Sh = area_between_vals(counts_MC_Sh, bins, 1000, 2200)
        C_2100_MC_ShL = area_between_vals(counts_MC_ShL, bins, 1000, 2200)
        C_2100_MC_LBE = area_between_vals(counts_MC_LBE, bins, 1000, 2200)
        weights=[(C_2100_data/C_2100_MC_LBE)*np.ones_like(energy_MC_LBE),(C_2100_data/C_2100_MC_ShL)*np.ones_like(energy_MC_ShL),(C_2100_data/C_2100_MC_Sh)*np.ones_like(energy_MC_Sh)]

        bins_new=np.arange(50,110,binwidth)

        counts_MC_list, bins, bars = ax0.hist([energy_MC_LBE,energy_MC_ShL,energy_MC_Sh], bins = bins_new, weights=weights, label = ['LBE List', 'Shielding_LIV List','Shielding List'], linewidth = 0.55, histtype ='step',color=['tab:orange','tab:grey','mediumseagreen'])
        counts_data, bins, bars_data = ax0.hist(energy_data, bins=bins_new,  label = "Data", histtype = 'step', linewidth = '0.65', color='tab:blue')
        #counts_MC_bi, bins, bars = ax0_3.hist(energy_MC_bi, bins = bins, histtype = 'step', linewidth = '0.35')
        #counts_MC_pb, bins, bars = ax0_3.hist(energy_MC_pb, bins = bins, histtype = 'step', linewidth = '0.35')
        plt.xlabel("Energy [keV]",  family='serif')
        ax0.set_ylabel("Counts", family='serif')
        ax0.set_yscale("log")
        #ax0_3.set_title(detector, family='serif')
        p.legend(ncol=1, out=False)
        p.pretty(large=8, grid=False)
        p.figure(dir+"/Spectra/"+detector+"/PhysicsList.eps")


    if scatterplot:
        g4sdf = combine_simulations("/lfs/l1/legend/detector_char/enr/hades/simulations/legend-g4simple-simulation/simulations/"+detector+"/th_HS2/top_0r_42z/hdf5/LBE/", MC_id)
        print("combined done")
        detector_hits = g4sdf.loc[(g4sdf.Edep>0)&(g4sdf.volID==1)]
        print("selection")
        procdf = pd.DataFrame(detector_hits.groupby(['x', 'y', 'z'], as_index=False))
        x_MC= procdf['x']
        y_MC= procdf['y']
        z_MC= procdf['z']
        print("preparation figure")
        fig, ax = plt.subplots()
        ax.scatter(x_MC, y_MC)
        ax.set_xlabel("x [mm]",    ha='right', x=1)
        ax.set_ylabel("y [mm]", ha='right', y=1)
        plt.savefig("scatterplot_xy.eps")
        fig2,ax2 = plt.subplots()
        ax2.scatter(x_MC, z_MC)
        ax.set_xlabel("x [mm]",    ha='right', x=1)
        ax.set_ylabel("z [mm]", ha='right', y=1)
        plt.savefig("scatterplot_xz.eps")
        #plt.show()


        #x_MC = read_all_MC_hdf5(sim_path+MC_id+isotope,'x')
        #y_MC = read_all_MC_hdf5(sim_path+MC_id+isotope,'y')
        #z_MC = read_all_MC_hdf5(sim_path+MC_id+isotope,'z')



def combine_simulations(MC_raw_path, MC_id):
    "combine all g4simple .hdf5 simulations within a folder into one dataframe"

    #read in each hdf5 file
    files = os.listdir(MC_raw_path)
    files = fnmatch.filter(files, "*.hdf5")
    df_list = []
    for file in files:

        # print("file: ", str(file))
        file_no = file[-7]+file[-6]
        # print("raw MC file_no: ", file_no)

        g4sfile = h5py.File(MC_raw_path+file, 'r')

        g4sntuple = g4sfile['default_ntuples']['g4sntuple']
        g4sdf = pd.DataFrame(np.array(g4sntuple), columns=['event'])

        # # build a pandas DataFrame from the hdf5 datasets we will use
        g4sdf = pd.DataFrame(np.array(g4sntuple['event']['pages']), columns=['event'])
        g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['Edep']['pages']), columns=['Edep']),lsuffix = '_caller', rsuffix = '_other')
        g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['volID']['pages']),columns=['volID']), lsuffix = '_caller', rsuffix = '_other')
        g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['x']['pages']),columns=['x']), lsuffix = '_caller', rsuffix = '_other')
        g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['y']['pages']),columns=['y']), lsuffix = '_caller', rsuffix = '_other')
        g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['z']['pages']),columns=['z']), lsuffix = '_caller', rsuffix = '_other')

        #add new column to each df for the raw MC file no
        g4sdf["raw_MC_fileno"] = file_no

        df_list.append(g4sdf)

    #concatonate
    df_total = pd.concat(df_list, axis=0, ignore_index=True)
    return df_total


def read_all_MC_hdf5(path_MC, key):

    #sto = lh5.Store()
    files = os.listdir(t2_folder)
    files = fnmatch.filter(files, "*hdf5")

    df_list = []

    for file in files:
        #get data, no cuts
        df=pd.read_hdf(file,key=key)
        #tb = sto.read_object("raw",t2_folder+file)[0]
        #df = lh5.Table.get_dataframe(tb)
        df_list.append(df)

    df_total = pd.concat(df_list, axis=0, ignore_index=True)
    return df_total


def find_bin_idx_of_value(bins, value):
    """Finds the bin which the value corresponds to."""
    array = np.asarray(value)
    idx = np.digitize(array,bins)
    return idx-1

def area_between_vals(counts, bins, val1, val2):
    """Calculates the area of the hist between two certain values"""
    left_bin_edge_index = find_bin_idx_of_value(bins, val1)
    right_bin_edge_index = find_bin_idx_of_value(bins, val2)
    bin_width = np.diff(bins)[0]
    area = sum(bin_width * counts[left_bin_edge_index:right_bin_edge_index])
    return area

if __name__ == "__main__":
    main()
