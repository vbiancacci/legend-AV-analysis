import numpy as np
import pandas as pd
import json
import statistics as stat
import matplotlib.pyplot as plt

def PropagationError(O_am6, O_am1, a_am6, O_am6_err, O_am1_err, a_am6_err, b_am1_err, b_am6_err):
   se=(O_am6_err/O_am6)**2+ (O_am1_err/O_am1)**2 + (a_am6_err/a_am6)**2 + ( b_am1_err)**2 +( b_am6_err)**2
   return np.sqrt(se)

def PropagationErrorThree(a_am6, a_am6_correrr, b_am6_correrr):
    se=(a_am6_correrr/a_am6)**2+b_am6_correrr**2
    return np.sqrt(se)

def main():

    ratio_list=[]
    ratio_corr_list=[]
    ratio_uncorr_list=[]
    ratio_total_err_list=[]

    #detectors_list = ["V07647B", "V07302A", "V07647A", "V07298B", "V07302B"] #, "V04549B"]
    detectors_list = ["B00032B", "B00091B", "B00000B"]

    ParametersCal = "par_calibration_matrix.json"

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
            a_am6_uncorr_err = Parameters[detector]["am_HS6"]['a_uncorr_err']
            a_am6_corr_err = Parameters[detector]["am_HS6"]['a_corr_err']
            b_am6 = Parameters[detector]["am_HS6"]['b']
            b_am6_uncorr_err = Parameters[detector]["am_HS6"]['b_uncorr_err']
            b_am6_corr_err = Parameters[detector]["am_HS6"]['b_corr_err']

            ratio= O_am1/O_am6 * a_am6* np.exp(b_am1-b_am6)

            ratio_corr = PropagationErrorThree(a_am6, a_am6_corr_err, b_am6_corr_err ) * ratio
            ratio_uncorr = PropagationError(O_am6, O_am1, a_am6, O_am6_err, O_am1_err, a_am6_uncorr_err, b_am1_err, b_am6_uncorr_err) * ratio
            ratio_total_err = np.sqrt(ratio_corr**2+ratio_uncorr**2)

            print("calibration_factor ", detector, " " , ratio, " +- ", ratio_corr , 'ratio_corr +-', ratio_uncorr,'ratio_uncorr')

            ratio_list.append(ratio)
            ratio_total_err_list.append(ratio_total_err)
            ratio_corr_list.append(ratio_corr)
            ratio_uncorr_list.append(ratio_uncorr)

    weight=[1/r**2 for r in ratio_total_err_list]
    mean_ratio=np.average(ratio_list, weights=weight)

    ratio_array=np.array(ratio_list)
    ratio_corr_array=np.array(ratio_corr_list)
    ratio_uncorr_array=np.array(ratio_uncorr_list)

    #for BEGe
    covariance_matrix = np.array([
        [ratio_uncorr_array[0]**2, ratio_corr_array[0]*ratio_corr_array[1], ratio_corr_array[0]*ratio_corr_array[2] ],
        [ratio_corr_array[1]*ratio_corr_array[0], ratio_uncorr_array[1]**2, ratio_corr_array[1]*ratio_corr_array[2] ],
        [ratio_corr_array[2]*ratio_corr_array[0], ratio_corr_array[2]*ratio_corr_array[1], ratio_uncorr_array[2]**2 ],
        ])
    #for ICPC 
    #covariance_matrix = np.array([
    #    [ratio_uncorr_array[0]**2, ratio_corr_array[0]*ratio_corr_array[1], ratio_corr_array[0]*ratio_corr_array[2], ratio_corr_array[0]*ratio_corr_array[3], ratio_corr_array[0]*ratio_corr_array[4] ],
    #    [ratio_corr_array[1]*ratio_corr_array[0], ratio_uncorr_array[1]**2, ratio_corr_array[1]*ratio_corr_array[2], ratio_corr_array[1]*ratio_corr_array[3], ratio_corr_array[1]*ratio_corr_array[4] ],
    #    [ratio_corr_array[2]*ratio_corr_array[0], ratio_corr_array[2]*ratio_corr_array[1], ratio_uncorr_array[2]**2, ratio_corr_array[2]*ratio_corr_array[3], ratio_corr_array[2]*ratio_corr_array[4] ],
    #    [ratio_corr_array[3]*ratio_corr_array[0], ratio_corr_array[3]*ratio_corr_array[1], ratio_corr_array[3]*ratio_corr_array[2], ratio_uncorr_array[3]**2, ratio_corr_array[3]*ratio_corr_array[4] ],
    #    [ratio_corr_array[4]*ratio_corr_array[0], ratio_corr_array[4]*ratio_corr_array[1], ratio_corr_array[4]*ratio_corr_array[2], ratio_corr_array[4]*ratio_corr_array[3], ratio_uncorr_array[4]**2]
    #    ])
    '''
    covariance_matrix = np.array([
        [2.74, 1.15, 0.86, 1.31],
        [1.15, 1.67, 0.82, 1.32],
        [0.86, 0.82, 2.12, 1.05],
        [1.31,1.32, 1.05, 2.93]
    ])
    '''

    inverse_covariance_matrix = np.linalg.inv(covariance_matrix)

    uni_array=np.array([1,1,1]) #,1,1])
    #ratio_array=[9.5,11.9,11.1, 8.9]

    #alphasz
    a_num = np.matmul(inverse_covariance_matrix, uni_array)
    a_det = np.matmul(uni_array, a_num)
    alphas=a_num/a_det
    print(alphas)
    mean_ratio=np.sum([a*r for a,r in zip(alphas,ratio_array)] )
    variance_ratio = 0.
    #for i in range(4):
    #    for j in range(4):
    #        variance_ratio = covariance_matrix[i,j]*alphas[i]*alphas[j]

    #variance_ratio=np.sqrt(variance_ratio)
    variance_ratio=np.sum([[ covariance_matrix[i,j]*alphas[i]*alphas[j] for j in range(3)] for i in range(3) ])
    variance_ratio=np.sqrt(variance_ratio)

    print("mean+-variance ", mean_ratio, "+-", variance_ratio)

if __name__ == "__main__":
    main()
