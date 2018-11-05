#!/bin/bash
# this script submits whatever script(s) you're using to the batch
# typical usage (high memory requirements):
# qsub -q hep.q -o $PWD/submit.log -e $PWD/submit.err -l h_vmem=24G submit.sh

#inputs
CMD="python diphotonCategorisation.py -t /home/hep/jg4814/CMSSW_10_2_0/2016/trees -d /home/hep/jg4814/CMSSW_10_2_0/2016/frames/trainTotal.pkl --intLumi 35.9 --trainParams eta:2e-1 "
MYDIR=/home/hep/jg4814/CMSSW_10_2_0/src/Stage1categorisation/TwoStep
NAME=/home/hep/jg4814/CMSSW_10_2_0/src/Stage1categorisation/TwoStep/Jobs/diphotonCategorisation/2016/sub__eta_2e-1
RAND=$RANDOM

#execution
cd $MYDIR
eval `scramv1 runtime -sh`
cd $TMPDIR
mkdir -p scratch_$RAND
cd scratch_$RAND
cp -p $MYDIR/*.py .
echo "About to run the following command:"
echo $CMD
if ( $CMD ) then
  touch $NAME.done
  echo 'Success!'
else
  touch $NAME.fail
  echo 'Failure..'
fi
cd -
rm -r scratch_$RAND
