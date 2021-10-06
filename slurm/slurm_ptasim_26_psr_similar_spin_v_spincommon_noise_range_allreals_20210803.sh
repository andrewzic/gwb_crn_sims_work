#!/bin/bash
#SBATCH --job-name=gwb_crn_sims_mc_spin_v_spincommon_noise_range_allreal
#SBATCH --output=/flush5/zic006/gwb_crn_sims/slurm_logs/gwb_crn_sims_mc_spin_v_spincommon_noise_range_20210803_%A_%a.log
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --mem=32G
#SBATCH --tmp=8G
#SBATCH --array=0-1210

# pyv="$(python -c 'import sys; print(sys.version_info[0])')"
# if [ "$pyv" == 2 ]
# then
#     echo "$pyv"
#     module load numpy/1.16.3-python-2.7.14
# fi


module load singularity

singularity exec /home/zic006/psr_gwb.sif which python3
singularity exec /home/zic006/psr_gwb.sif echo $TEMPO2
singularity exec /home/zic006/psr_gwb.sif echo $TEMPO2_CLOCK_DIR

echo $SLURM_ARRAY_TASK_ID

paramfiles=(/flush5/zic006/gwb_crn_sims/params/all_mc_array_spin_v_spincommon/params_all_mc_array_spin_v_spincommon_[0-9]*.dat)
echo "processing paramfile ${paramfiles[$SLURM_ARRAY_TASK_ID]}"


#doing the main run
singularity exec /home/zic006/psr_gwb.sif python3 /flush5/zic006/gwb_crn_sims/run_enterprise_simple.py --prfile ${paramfiles[$SLURM_ARRAY_TASK_ID]}

#doing the results run
singularity exec /home/zic006/psr_gwb.sif python3 -m enterprise_warp.results --result ${paramfiles[$SLURM_ARRAY_TASK_ID]} --info 1 -c 2 -p "gw" -f 1 -l 1 -m 1 -b 1 > `basename ${paramfiles[$SLURM_ARRAY_TASK_ID]} .dat`.result

#calculating optimal statistic
singularity exec /home/zic006/psr_gwb.sif python3 -m enterprise_warp.results --result ${paramfiles[$SLURM_ARRAY_TASK_ID]} -o 1 -g "hd,dipole,monopole" -N 10000 -M 1
