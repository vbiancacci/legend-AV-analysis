import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import gridspec
from datetime import datetime
import json
import argparse
import os
from scipy import optimize
from scipy.stats import poisson, norm
from scipy import stats
import math
import pygama.analysis.histograms as pgh


sys.path.insert(1,'/lfs/l1/legend/users/bianca/IC_geometry/analysis/myplot')
from myplot import *

myStyle = True
#Script to plot spectra of MC (best fit FCCD) and data

def main():
    bands = False

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
    DLF_list= [0.9] #[0.3]#, 1.0]
    smear="g"
    frac_FCCDbore=0.5
    TL_model="l"
    FCCD = 1.017
    sim_path="/lfs/l1/legend/detector_char/enr/hades/simulations/legend-g4simple-simulation/simulations/"+detector+"/th_HS2/top_0r_42z/hdf5/AV_processed/"

    isotope=".hdf5"

    dir=os.path.dirname(os.path.realpath(__file__))
    print("working diboxory: ", dir)

    #initialise diboxories to save spectra
    if not os.path.exists(dir+"/Spectra/"+detector+"/"):
        os.makedirs(dir+"/Spectra/"+detector+"/")

    print("start...")
    p =Plot((15,5),n=1)
    ax0=p.ax
    binwidth = 10 #keV
    bins = np.arange(0,2700,binwidth)
    #GET DATA
    df = pd.read_hdf(dir+"/data_calibration/"+detector+"/calibrated_energy_"+detector+"_"+energy_filter+"_run"+str(run)+".hdf5", key='energy')
    energy_data = df['calib_energy']
    counts_data, bins, bars_data = ax0.hist(energy_data, bins=bins, histtype = 'step', linewidth = '0.75')
    C_2100_data = area_between_vals(counts_data, bins, 2250, 2500)
    counts_data, bins, bars_data = ax0.hist(energy_data, bins=bins,  label = "Data", histtype = 'step', linewidth = '0.55', color='tab:blue')

    #GET MC
    for DLF in DLF_list:
        MC_id=detector+"-th_HS2-top-0r-42z"+"_"+smear+"_"+TL_model+"_FCCD"+str(FCCD)+"mm_DLF"+str(DLF)+"_fracFCCDbore"+str(frac_FCCDbore)
        df_sim =  pd.read_hdf(sim_path+MC_id+isotope, key="procdf")
        print(sim_path+MC_id+isotope)
        energy_MC = df_sim['energy']
        print("opened MC")

    #Plot data and scaled MC

        counts_MC, bins = np.histogram(energy_MC, bins = bins)

        C_2100_MC = area_between_vals(counts_MC, bins, 2250, 2500)
        print(C_2100_MC, C_2100_data)

        scaled_counts_MC, bins, bars = ax0.hist(energy_MC, bins = bins,  label = "g4simple simulation", weights=(C_2100_data/C_2100_MC)*np.ones_like(energy_MC), histtype = 'step', linewidth = '0.65')  #label="DLF "+str(DLF)

    plt.xlabel("Energy [keV]",  family='serif', fontsize=20)
    ax0.set_ylabel("Counts / 1keV", family='serif', fontsize=20)
    ax0.tick_params(axis="both", labelsize=15)
    ax0.set_yscale("log")
    ax0.set_xlim(0,2700)
    #ax0.set_title(detector, family='serif')

    p.legend(ncol=1, out=False, pos = "lower left")
    p.pretty(large=8, grid=False)
    #plt.show()
    p.figure(dir+"/Spectra/"+detector+"/DataMC_"+MC_id+"_"+energy_filter+"_run"+str(run)+"_cuts_top.pdf")



    '''
    print("basic histos complete")

    Data_MC_ratios = []
    Data_MC_ratios_err = []
    for index, bin in enumerate(bins[1:]):
        data = counts_data[index]
        MC = counts_MC[index] #This counts has already been scaled by weights
        if MC == 0:
            ratio = 0.
            error = 0.
        else:
            try:
                ratio = data/MC
                try:
                    error = np.sqrt(1/data + 1/MC)
                except:
                    error = 0.
            except:
                ratio = 0 #if MC=0 and dividing by 0
        Data_MC_ratios.append(ratio)
        Data_MC_ratios_err.append(error)

    print("errors")

    ax1.errorbar(bins[1:], Data_MC_ratios, yerr=Data_MC_ratios_err,color="green", elinewidth = 1, fmt='x', ms = 1.0, mew = 1.0)
    ax1.hlines(1, xmin, xmax, colors="gray", linestyles='dashed')
    '''



    if bands==True:
        fig2 = plt.subplots()
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax2a = plt.subplot(gs[0])
        ax2b = plt.subplot(gs[1], sharex = ax2a)
        ax2a.set_yscale("log")
        ax2b.set_ylim(0.8,1.2)
        print('making figure')
        counts_dataa, binsa, bars_dataa = ax2a.hist(energy_data, bins=bins,  label = "Data", histtype = 'step', linewidth = '0.35')
        scaled_counts_MC, bins, bars = ax2a.hist(energy_MC, bins = bins,  label = "g4simple simulation", weights=(C_2100_data/C_2100_MC)*np.ones_like(energy_MC), histtype = 'step', linewidth = '0.55')
        print('before dividing')
        ratio = np.divide(counts_dataa,
                  scaled_counts_MC,
                  where=(scaled_counts_MC != 0))
        ax2b.errorbar(binsa[1:], ratio, fmt='.',color='b')
        print('after dividing')
        residual = False
        for c,b,l, in zip(scaled_counts_MC, binsa, pgh.get_bin_widths(binsa)):
            print(b)
            box1a, box2a, box3a = draw_poisson_bands(c,b,l,residual)
            ax2a.add_patch(box3a)
            ax2a.add_patch(box2a)
            ax2a.add_patch(box1a)
            #ax2a.set_xlabel("Energy [keV]",  family='serif')
            ax2a.set_ylabel("Counts / 10keV", family='serif', fontsize=15)
        residual2 = True
        print('after first plot')
        for c,b,l, in zip(counts_MC, bins, pgh.get_bin_widths(bins)):
            print(b)
            box1b, box2b, box3b = draw_poisson_bands(c,b,l,residual2)
            ax2b.add_patch(box3b)
            ax2b.add_patch(box2b)
            ax2b.add_patch(box1b)
            ax2b.set_xlabel("Energy [keV]",  family='serif', fontsize=15)
            ax2b.set_ylabel("Residual", family='serif', fontsize=15)

    plt.show()

    # plt.subplots_adjust(hspace=.0)

    if cuts == False:
        plt.savefig(dir+"/Spectra/"+detector+"/DataMC_"+MC_id+"_"+energy_filter+"_run"+str(run)+".png")
    else:
        plt.savefig(dir+"/Spectra/"+detector+"/DataMC_"+MC_id+"_"+energy_filter+"_run"+str(run)+"_cuts.png")

    print("done")





def smallest_poisson_interval(cov, mu):
    #if (cov > 1 or cov < 0 or mu < 0):
    #    throw std::runtime_error("smallest_poisson_interval(): bad input");
    res = [mu,mu]
    if (mu>50):#gaussian pproximation
        res = [
        mu + norm.ppf((1-cov)/2)*np.sqrt(mu)-0.5,
        mu - norm.ppf((1-cov)/2)*np.sqrt(mu)+0.5
        ]
    else:
        mode = math.floor(mu) #start from the mode=integer part of the mean
        l = mode
        u = mode
        prob = poisson.pmf(mode,mu)
        while (prob < cov):
            prob_u = poisson.pmf(u+1, mu)
            l_new=l-1 if l>0 else 0
            prob_l = poisson.pmf(l_new, mu)
            # we expand on the right if:
            #- the lower edge is already at zero
            #- the prob of the right point is higher than the left
            if (l == 0 or prob_u > prob_l):
                u += 1
                prob += prob_u
            #otherwhise we expand on the left
            elif (prob_u < prob_l):
                l-= 1
                prob += prob_l
            #if prob_u == prob_l we expand on both sides
            else:
                u += 1
                l -= 1
                prob += prob_u + prob_l
        l_n=0 if l==0 else l-0.5
        res = [l_n, u+0.5]

    return res


def draw_poisson_bands(mu, x_low, x_size, residuals = False):
    #int col_idx = 9000
    #if (!col_defined):
    #    new TColor(col_idx  , 238./255, 136./255, 102./255, "tol-lig-orange")
    #    new TColor(col_idx+1, 238./255, 221./255, 136./255, "tol-lig-lightyellow")
    #    new TColor(col_idx+2, 187./255, 204./255,  51./255, "tol-lig-pear")
    #    col_defined = true

    sig1 = smallest_poisson_interval(0.682, mu)
    sig2 = smallest_poisson_interval(0.954, mu)
    sig3 = smallest_poisson_interval(0.997, mu)

    if (residuals) :
        if (mu != 0):
            sig1[0] /= mu
            sig1[1] /= mu
            sig2[0] /= mu
            sig2[1] /= mu
            sig3[0] /= mu
            sig3[1] /= mu
        else:
            sig1[0] = sig1[1] = 1
            sig2[0] = sig2[1] = 1
            sig3[0] = sig3[1] = 1

    cent_b1 = (sig1[1] + sig1[0])/2
    cent_b2 = (sig2[1] + sig2[0])/2
    cent_b3 = (sig3[1] + sig3[0])/2

    xdw = x_low
    xup = x_low + x_size

    ''''
    if (h != nullptr):
        auto xc1 = gPad->GetUxmin()
        auto xc2 = gPad->GetUxmax()
        auto yc1 = gPad->GetUymin()
        auto yc2 = gPad->GetUymax()

        if (sig1.[0]  < yc1):
            sig1.[0]  = yc1
        if (sig2.[0]  < yc1):
            sig2.[0]  = yc1
        if (sig3.[0]  < yc1):
            sig3.[0]  = yc1
        if (sig1.[1] > yc2):
            sig1.[1] = yc2
        if (sig2.[1] > yc2):
            sig2.[1] = yc2
        if (sig3.[1] > yc2):
            sig3.[1] = yc2
        if (xdw < xc1):
            xdw = xc1
        if (xup > xc2):
            xup = xc2
    '''

    box_b1 = patches.Rectangle((xdw, sig1[0]), width=abs(xup-xdw),height=abs(sig1[1]-sig1[0]),color='yellowgreen')
    box_b2 = patches.Rectangle((xdw, sig2[0]), width=abs(xup-xdw),height=abs(sig2[1]-sig2[0]), color='gold')
    box_b3 = patches.Rectangle((xdw, sig3[0]), width=abs(xup-xdw),height=abs(sig3[1]-sig3[0]), color='orange')


    return box_b1, box_b2, box_b3

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
