#!/bin/tcsh

set script_pwd=`readlink -f "$0"`
set scriptdir=`dirname $script_pwd`

filearr=( `ls "$scriptdir"/ptasim_input_files/spinnoise_100r/[0-9]*.inp` );
ind=$argv[1]
inpfile=${filearr[ind]}
cd ${scriptdir}/data/100r/
echo $inpfile
ptaSimulate "$inpfile"
set pref=`basename "$inp" .inp`
echo $pref
source regsamp_${pref}/scripts/runScripts_master
