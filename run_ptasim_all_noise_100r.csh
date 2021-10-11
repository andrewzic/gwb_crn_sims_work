#!/bin/tcsh

foreach inp ( `ls "$PWD"/ptasim_input_files/100r/[0-9]*.inp` );
cd data/100r/
echo $inp
ptaSimulate "$inp"
set ind=`basename "$inp" .inp`
echo $ind
source regsamp_${ind}/scripts/runScripts_master
cd ..
end
