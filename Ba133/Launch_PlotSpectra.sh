export PATH=~/miniconda3/bin:$PATH

detector=V07302A
FCCD=2.13
DLF=1.0 
smear=g
frac_FCCDbore=0.5
TL_model=notl
MC_id=${detector}-ba_HS4-top-0r-78z_${smear}_${TL_model}_FCCD${FCCD}mm_DLF${DLF}_fracFCCDbore${frac_FCCDbore}
sim_path=/lfs/l1/legend/users/aalexander/legend-g4simple-simulation/legend/simulations/${detector}/ba_HS4/top_0r_78z/hdf5/AV_processed/${MC_id}.hdf5
data_path=/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/gen/${detector}/tier2/ba_HS4_top_dlt/
data_path=/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/gen/I02160A/tier2/ba_HS4_top_dlt/
energy_filter=cuspEmax_ctc
calibration=/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/genpar/dsp_ecal/${detector}.json
# calibration=/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/genpar/dsp_ecal/I02160A.json
cuts=True
run=1

python PlotSpectra.py $detector $MC_id $sim_path $FCCD $DLF $data_path $calibration $energy_filter $cuts $run
