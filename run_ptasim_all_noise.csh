#!/bin/tcsh

@ n = 0
foreach inp ( `ls ptasim_input_files/ptasim_all_similar_26_[0-4].inp` );
echo $inp
ptaSimulate $inp
source all_similar_regsamp_$n/scripts/runScripts_master
@ n = $n + 1
end
