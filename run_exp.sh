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
#$MAKE FGC_VER=1.7.11-predict
#$MAKE run_exp EXP_DIR=experiments/v1.0.2_on_1.7.11_predict PRED_INFER=rule FGC_VER=1.7.11-predict USE_SE=SHINT
#$MAKE FGC_VER=1.7.12
#$MAKE run_exp EXP_DIR=experiments/v2.0.2_on_1.7.12 PRED_INFER=rule FGC_VER=1.7.12 USE_SE=pred RUN_ON=raw
#$MAKE compare EXP_DIR=experiments/v2.0.2_on_1.7.12 COMPARED_EXP_DIR=experiments/v2.0.1_on_1.7.12
#$MAKE compare EXP_DIR=experiments/v2.0.2_on_1.7.12 COMPARED_EXP_DIR=experiments/v1.0.2_on_1.7.11_predict
#$MAKE FGC_VER=1.7.13
#$MAKE run_exp EXP_DIR=experiments/v2.0.2_on_1.7.13 PRED_INFER=rule FGC_VER=1.7.13 USE_SE=pred RUN_ON=raw
#$MAKE compare EXP_DIR=experiments/v2.0.2_on_1.7.13 COMPARED_EXP_DIR=experiments/v2.0.2_on_1.7.12
#$MAKE FGC_VER=1.7.13
#$MAKE run_exp EXP_DIR=experiments/v2.0.3_on_1.7.13 PRED_INFER=rule FGC_VER=1.7.13 USE_SE=pred RUN_ON=raw
#$MAKE compare EXP_DIR=experiments/v2.0.3_on_1.7.13 COMPARED_EXP_DIR=experiments/v2.0.2_on_1.7.13
$MAKE FGC_VER=1.7.13
$MAKE run_exp EXP_DIR=experiments/v2.0.3.1_on_1.7.13 PRED_INFER=rule FGC_VER=1.7.13 USE_SE=pred RUN_ON=raw
$MAKE compare EXP_DIR=experiments/v2.0.3.1_on_1.7.13 COMPARED_EXP_DIR=experiments/v2.0.3_on_1.7.13