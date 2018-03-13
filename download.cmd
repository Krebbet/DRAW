@echo off
SET BUCKET=gs://d-draw
REM #####################################################
SET TRIAL_NUM=39
SET READ_N=5
SET WRITE_N=5
SET ZSIZE=10
SET GLIMPSES=10
SET NAME=draw_r%READ_N%w%WRITE_N%z%ZSIZE%t%GLIMPSES%_%TRIAL_NUM%
SET DIR=C:\TensorFlow\draw-cloud\models\%NAME%_inference
mkdir %DIR%
@echo on
gsutil -m cp %BUCKET%/%NAME%/%NAME%* %DIR%