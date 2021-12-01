import numpy as np
import pandas as pd
import json
import statistics as stat
import matplotlib.pyplot as plt

def StatisticalError (A, B, errA, errB):
   se=(errA/A)**2+(errB/B)**2
   return np.sqrt(se)


def variance(ratio,err, thresh=1e-6, iter_max=2):
    v=stat.variance(ratio)
    n=len(ratio)
    square_err=[error**2 for error in err]
    S=stat.mean(square_err)
    for i in range(1,iter_max):
        w=[1/(v+square_error) for square_error in square_err]
        w=w/sum(w)
        m=sum(ratio*w)
        v_0=max(0,sum((ratio-m)**2*w)*n/(n-1)-S)
        if (v<=v_0*(1+thresh) and v_0<=v*(1+thresh)):
            break
        v=v_0
    if (i>= iter_max):
        print("Exceeded maximum iteration")
    return m , v, i

def variance_milotti(ratio, err):
        n=len(ratio)
        m=sum(ratio)/n
        ratio_diff=[(r-m)**2 for r in ratio]
        v_0=max(0,sum(ratio_diff)/(n*(n-1)))
        return m , v_0, n

def variance_bevington(ratio,err):
    n=len(ratio)
    inv_err_square=[1/er**2 for er in err]
    num=[r/er**2 for r,er in zip(ratio,err)]
    v=max(0,sum(inv_err_square))
    m=max(0,sum(num))/v
    n_err=[((r-m)/er)**2 for r,er in zip(ratio,err)]
    v_err=max(0,sum(n_err))*n/(v*(n-1))
    return m , v_err, n

def bootstrap_test(n,mean,sigma,err):
    i=0
    mean_list=[]
    sigma_list=[]
    print(sigma)
    while i!=10000:
        #np.random.seed(0)
        y=np.random.normal(mean,sigma,n)
        epsilon=np.random.normal(0,err,n)
        x=y+epsilon
        mm,vv,_=variance(x,err)
        vv=np.sqrt(vv)
        mean_list.append(mm)
        sigma_list.append(vv)
        #print("err",err)
        #print(vv)
        i+=1
    return mean_list, sigma_list

def main():

    ratio_new_list=[]
    ratio_new_err_list=[]

    detectors_list = ["V07647B", "V07302A", "V07647A", "V07298B", "V07302B", "V04549B"]

    ParametersCal = "par_calibration.json"

    with open(ParametersCal) as json_file:
        Parameters = json.load(json_file)
        for detector in detectors_list:
            O_am1 = Parameters[detector]["am_HS1"]['O_Am241_data']
            O_am1_err = Parameters[detector]["am_HS1"]['O_Am241_data_err']
            a_am1 = Parameters[detector]["am_HS1"]['a']
            a_am1_err = Parameters[detector]["am_HS1"]['a_err']
            b_am1 = Parameters[detector]["am_HS1"]['b']
            b_am1_err = Parameters[detector]["am_HS1"]['b_err']

            O_am6 = Parameters[detector]["am_HS6"]['O_Am241_data']
            O_am6_err = Parameters[detector]["am_HS6"]['O_Am241_data_err']
            a_am6 = Parameters[detector]["am_HS6"]['a']
            a_am6_err = Parameters[detector]["am_HS6"]['a_err']
            b_am6 = Parameters[detector]["am_HS6"]['b']
            b_am6_err = Parameters[detector]["am_HS6"]['b_err']

            ratio = O_am6/O_am1
            ratio_err = StatisticalError(O_am6, O_am1, O_am6_err, O_am1_err)*ratio
            #num_err = np.sqrt((ratio*a_am1_err)**2 + (a_am1*ratio_err)**2)
            #ratio_new = a_am1 /(a_am6 /ratio)
            #ratio_new_err = StatisticalError(a_am1*ratio, a_am6, num_err, a_am6_err)
            num=a_am6*np.exp(b_am1-b_am6)
            e_err=np.sqrt(b_am1_err**2+b_am6_err**2)
            num_err=StatisticalError(a_am6, np.exp(b_am1-b_am6), a_am6_err, e_err)*num
            ratio_new = num/ratio
            ratio_new_err = StatisticalError(num, ratio, num_err, ratio_err)*ratio_new

            print("calibration_factor ", detector, " " , ratio_new, " +- ", ratio_new_err , 'ratio_err', ratio_err,'am6_err', a_am6_err)# I_6 ", a_am6, " R_6 ", O_am6, " R_1 ", O_am1)
            ratio_new_list.append(ratio_new)
            ratio_new_err_list.append(ratio_new_err)

    mean_ratio=stat.mean(ratio_new_list)
    n=len(ratio_new_list)
    mean_ratio_err=v=np.sqrt(stat.variance(ratio_new_list))*(n-1)/n
    #mean_ratio_err=(max(ratio_new_list)-min(ratio_new_list))/(2*np.sqrt(n))

    print ("mean ratio ", mean_ratio, " +- ", mean_ratio_err )
    mean, err, i = variance_bevington(ratio_new_list, ratio_new_err_list)
    sigma=np.sqrt(err/n)
    print(mean, " +- ", sigma , " rel_err ", sigma/mean*100, "%")

    mean_list, sigma_list=bootstrap_test(n, mean, sigma,ratio_new_err_list)

    fig, (ax1, ax2)= plt.subplots(1,2)
    ax1.hist(mean_list, bins=50)
    ax1.set_xlabel('Mean')
    ax1.axvline(mean, color='k', linestyle='dashed', linewidth=1)
    median_mean=stat.median(mean_list)
    ax1.axvline(median_mean, color='red', linestyle='dashed', linewidth=1)
    ax2.hist(sigma_list, bins=50)
    ax2.set_xlabel('Sigma')
    ax2.axvline(sigma, color='k', linestyle='dashed', linewidth=1)
    median_sigma=stat.median(sigma_list)
    print(median_sigma, sigma)
    ax2.axvline(median_sigma, color='red', linestyle='dashed', linewidth=1)
    plt.savefig('hist.png')


if __name__ == "__main__":
    main()
