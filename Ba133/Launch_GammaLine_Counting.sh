#Old bash script to launch Gamma line counting

export PATH=~/miniconda3/bin:$PATH

#=========SIM MODE===============
# detector=V07302A
# # FCCD_list=0.0 0.25 0.5 0.75 1.0 1.25 1.5 3.0 
# # DLF_list=1.0 
# smear="g"
# frac_FCCDbore=0.5
# TL_model=notl

# for FCCD in 2.13
#     do
#         for DLF in 1.0
#             do
#                 echo FCCD is $FCCD
#                 echo DLF is $DLF
#                 MC_id=${detector}-ba_HS4-top-0r-78z_${smear}_${TL_model}_FCCD${FCCD}mm_DLF${DLF}_fracFCCDbore${frac_FCCDbore}
#                 # echo $MC_id
#                 sim_path=/lfs/l1/legend/users/aalexander/legend-g4simple-simulation/legend/simulations/${detector}/ba_HS4/top_0r_78z/hdf5/AV_processed/${MC_id}.hdf5
#                 python GammaLine_Counting_Ba133.py --sim $detector $sim_path $MC_id
#           done
#     done



#===========DATA MODE:============= 
detector=V04199A
data_path=/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/gen/${detector}/tier2/ba_HS4_top_dlt/
# data_path=/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/gen/I02160B/tier2/ba_HS4_top_dlt/
energy_filter=cuspEmax_ctc
cuts=True
calibration=/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/genpar/dsp_ecal/${detector}.json
# calibration=/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/genpar/dsp_ecal/I02160B.json
run=1
python GammaLine_Counting_Ba133.py --data $detector $data_path $calibration $energy_filter $cuts $run
