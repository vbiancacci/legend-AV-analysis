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


#Script to plot spectra of MC (best fit FCCD) and data

def main():
    bands = False

    if(len(sys.argv) != 10):
        print('Example usage: python AV_postproc.py <detector> <MC_id> <sim_path> <FCCD> <DLF> <cuts> <run>')
        sys.exit()

    detector = sys.argv[1] #raw MC folder path - e.g. "/lfs/l1/legend/users/aalexander/legend-g4simple-simulation/legend/simulations/V02160A/ba_HS4/top_0r_78z/hdf5/"
    MC_id = sys.argv[2]     #file id, including fccd config - e.g. ${detector}-ba_HS4-top-0r-78z_${smear}_${TL_model}_FCCD${FCCD}mm_DLF${DLF}_fracFCCDbore${frac_FCCDbore}
    sim_path = sys.argv[3] #path to AV processed sim, e.g. /lfs/l1/legend/users/aalexander/legend-g4simple-simulation/legend/simulations/${detector}/ba_HS4/top_0r_78z/hdf5/AV_processed/${MC_id}.hdf5
    FCCD = sys.argv[4] #FCCD of MC - e.g. 0.69
    DLF = sys.argv[5] #DLF of MC - e.g. 1.0
    energy_filter = sys.argv[6] #energy filter - e.g. trapEftp
    cuts = sys.argv[7] #e.g. False
    run = int(sys.argv[8]) #data run, e.g. 1 or 2
    source = sys.argv[9]

    print("detector: ", detector)
    print("MC_id: ", MC_id)
    print("sim_path: ", sim_path)
    print("FCCD: ", str(FCCD))
    print("DLF: ", str(DLF))
    print("energy_filter: ", energy_filter)
    print("applying data cuts: ", cuts)
    print("run: ", run)
    print("source: ", source)

    if cuts == "False":
        cuts = False
    else:
        cuts = True

    dir=os.path.dirname(os.path.realpath(__file__))
    print("working diboxory: ", dir)

    #initialise diboxories to save spectra
    if not os.path.exists(dir+"/Spectra/"+detector+"/"+source+"/"):
        os.makedirs(dir+"/Spectra/"+detector+"/"+source+"/")

    print("start...")

    #GET DATA
    df = pd.read_hdf(dir+"/data_calibration/"+detector+"/"+source+"/loaded_energy_"+detector+"_"+energy_filter+"_run"+str(run)+".hdf5", key='energy')
    energy_data = df['energy_filter']

    #GET MC
    df_sim =  pd.read_hdf(sim_path, key="procdf")
    energy_MC = df_sim['energy']
    print("opened MC")

    #Get peak counts C_356 for scaling
    if cuts == False:
        PeakCounts_data = dir+"/PeakCounts/"+detector+"/"+source+"/new/PeakCounts_data_"+detector+"_"+energy_filter+"_run"+str(run)+".json"
    else:
        PeakCounts_data = dir+"/PeakCounts/"+detector+"/"+source+"/new/PeakCounts_data_"+detector+"_cuts_"+energy_filter+"_run"+str(run)+".json"

    PeakCounts_MC = dir+"/PeakCounts/"+detector+"/"+source+"/new/PeakCounts_sim_"+MC_id+".json"

    with open(PeakCounts_data) as json_file:
        PeakCounts = json.load(json_file)
        C_60_data = PeakCounts['C_60']

    with open(PeakCounts_MC) as json_file:
        PeakCounts = json.load(json_file)
        C_60_MC = PeakCounts['C_60']

    print("got peak counts")

    #Plot data and scaled MC
    binwidth = 0.1 #keV
    xmin=0
    xmax=120
    bins = np.arange(xmin,xmax,binwidth)


    fig, ax = plt.subplots()
    #gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    #ax = plt.subplot(gs[0])
    #ax1 = plt.subplot(gs[1], sharex = ax)

    counts_data, bins, bars_data = ax.hist(energy_data, bins=bins,  label = "Data", histtype = 'step', linewidth = '0.35')
    counts_MC, bins, bars = ax.hist(energy_MC, bins = bins, weights=(C_60_data/C_60_MC)*np.ones_like(energy_MC), label = "G4simple simulation", histtype = 'step', linewidth = '0.35')

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

    #ax1.errorbar(bins[1:], Data_MC_ratios, yerr=Data_MC_ratios_err,color="green", elinewidth = 1, fmt='x', ms = 1.0, mew = 1.0)
    #ax1.hlines(1, xmin, xmax, colors="gray", linestyles='dashed')


    plt.xlabel("Energy [keV]")
    ax.set_ylabel("Counts / 0.1keV")
    ax.set_yscale("log")
    ax.legend(frameon=False, loc = "upper left")
    #ax.set_title(detector)
    #ax1.set_ylabel("data/MC")
    #ax1.set_yscale("log")
    #ax1.set_xlim(xmin,xmax)
    ax.set_xlim(xmin,xmax)


    if bands==True:
        fig2 = plt.subplots()
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax2a = plt.subplot(gs[0])
        ax2b = plt.subplot(gs[1], sharex = ax2a)

        counts_dataa, binsa, bars_dataa = ax2a.hist(energy_data, bins=bins,  label = "Data", histtype = 'step', linewidth = '0.35')
        counts_MCa, binsa, barsa = ax2a.hist(energy_MC, bins = bins, weights=(C_60_data/C_60_MC)*np.ones_like(energy_MC), label = "G4simple simulation", histtype = 'step', linewidth = '0.35')
        ratio = np.divide(counts_dataa,
                  counts_MCa,
                  where=(counts_MCa != 0))
        ax2b.errorbar(binsa[1:], ratio, fmt='.',color='b')
        residual = False
        for c,b,l, in zip(counts_MCa, binsa, pgh.get_bin_widths(binsa)):
            box1a, box2a, box3a = draw_poisson_bands(c,b,l,residual)
            ax2a.set_yscale("log")
            ax2a.add_patch(box3a)
            ax2a.add_patch(box2a)
            ax2a.add_patch(box1a)
        residual2 = True
        for c,b,l, in zip(counts_MC, bins, pgh.get_bin_widths(bins)):
            box1b, box2b, box3b = draw_poisson_bands(c,b,l,residual2)
            ax2b.add_patch(box3b)
            ax2b.add_patch(box2b)
            ax2b.add_patch(box1b)

    plt.show()

    # plt.subplots_adjust(hspace=.0)

    if cuts == False:
        plt.savefig(dir+"/Spectra/"+detector+"/"+source+"/DataMC_"+MC_id+"_"+energy_filter+"_run"+str(run)+".pdf")
    else:
        plt.savefig(dir+"/Spectra/"+detector+"/"+source+"/DataMC_"+MC_id+"_"+energy_filter+"_run"+str(run)+"_cuts.eps")

    print("done")





def smallest_poisson_interval(cov, mu):
    #if (cov > 1 or cov < 0 or mu < 0):
    #    throw std::runtime_error("smallest_poisson_interval(): bad input");
    res = [mu,mu]
    #if (mu>10000): #gaussian pproximation
    #    res = [
    #    mu + norm.ppf((1-cov/2)*np.sqrt(mu)-0.5),
    #    mu - norm.ppf((1-cov/2)*np.sqrt(mu)+0.5)
    #    ]
    #else:
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




if __name__ == "__main__":
    main()
