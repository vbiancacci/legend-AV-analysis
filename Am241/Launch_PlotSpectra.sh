#Old bash script to launch PlotSpectra.py

export PATH=~/miniconda3/bin:$PATH

detector=V07647B
FCCD=1.1
DLF=1.0 
smear=g
frac_FCCDbore=0.5
TL_model=notl
MC_id=${detector}-am_HS1-top-46r-3z_${smear}_${TL_model}_FCCD${FCCD}mm_DLF${DLF}_fracFCCDbore${frac_FCCDbore}
sim_path=/lfs/l1/legend/detector_char/enr/hades/simulations/legend-g4simple-simulation/simulations/V07647B/am_HS1/top_46r_3z/hdf5/AV_processed/${MC_id}.hdf5
data_path=/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/gen/${detector}/tier2/am_HS1_top_dlt/
energy_filter=cuspEmax_ctc
calibration=/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/genpar/dsp_ecal/${detector}.json
# calibration=/lfs/l1/legend/legend-prodenv/prod-usr/ggmarsh-test-v03/genpar/dsp_ecal/I02160A.json
cuts=True
run=1
ssource=am_HS1

python PlotSpectra.py $detector $MC_id $sim_path $FCCD $DLF $energy_filter $cuts $run $ssource
