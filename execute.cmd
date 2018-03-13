SET BUCKET=gs://d-draw

echo Copy Model From Google Cloud
REM #####################################################
SET TRIAL_NUM=3
SET READ_N=5
SET WRITE_N=5
SET ZSIZE=10
SET GLIMPSES=10
SET NAME=draw_r%READ_N%w%WRITE_N%z%ZSIZE%t%GLIMPSES%_%TRIAL_NUM%
SET DIR=C:\TensorFlow\draw-cloud\models\%NAME%-inference

gsutil -m cp %BUCKET%/%NAME%/%NAME%* %DIR%
