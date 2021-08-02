#!/bin/csh


set psrlist=$1
foreach psr ( `cat ${psrlist}` )
echo $psr
#if ( $psr == "J1713+0747" ) then
set model=${psr}_input.model
echo $model
tempo2 -gr cholSpectra -f $psr.par $psr.tim -dcf $model
end
