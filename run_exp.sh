#!/usr/bin/env bash

MAKE="make"

# -n: dry-run
while getopts "n" OPTION
do
  case $OPTION in
  n)
    MAKE="make -n"
  esac
done

#make run_exp EXP_DIR=experiments/v1.0.1_on_1.7.8 PRED_INFER=rule
#make run_exp EXP_DIR=experiments/v1.0.2_on_1.7.8_revise_sp PRED_INFER=rule FGC_VER=1.7.8-revise-sp USE_SE=sp
$MAKE FGC_VER=1.7.11-predict
$MAKE run_exp EXP_DIR=experiments/v1.0.2_on_1.7.11_predict PRED_INFER=rule FGC_VER=1.7.11-predict USE_SE=SHINT