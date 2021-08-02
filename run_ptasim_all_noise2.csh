#!/bin/tcsh

@ n = 10
foreach inp ( `ls ptasim_input_files/ptasim_all_similar_26_[01][01].inp` );
echo $inp
ptaSimulate $inp
source all_similar_regsamp_$n/scripts/runScripts_master
@ n = $n + 1
end
