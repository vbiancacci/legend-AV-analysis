#Old bash script to launch Calculate_FCCD.py

export PATH=~/miniconda3/bin:$PATH

detector=V07302B
MC_id=${detector}-ba_HS4-top-0r-78z
smear=g
TL_model=notl
frac_FCCDbore=0.5
energy_filter=trapEftp
cuts=True

python Calculate_FCCD.py $detector $MC_id $smear $TL_model $frac_FCCDbore $energy_filter $cuts
