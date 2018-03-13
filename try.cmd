@echo off
SET BUCKET=gs://d-draw

SET PROCESS_DIR=C:\TensorFlow\draw-cloud\process

SET OUT_DIR=C:\TensorFlow\draw-cloud\out
REM #####################################################
mkdir %DIR%
echo Copy Model From Google Cloud
@echo on
REM gsutil -m cp %BUCKET%/%NAME%/%NAME%* %DIR%
REM #####################################################
SET TRIAL_NUM=7
SET READ_N=5
SET WRITE_N=2
SET ZSIZE=10
SET GLIMPSES=20
SET NAME=draw_r%READ_N%w%WRITE_N%z%ZSIZE%t%GLIMPSES%_%TRIAL_NUM%
SET DIR=C:\TensorFlow\draw-cloud\models\%NAME%
SET GIF_DIR=%DIR%\gif

echo generate results!
SET CHECK_NUM=
python execute.py --job-dir %DIR% --model-name %NAME%%CHECK_NUM% ^
	--z-size %ZSIZE% ^
	--T %GLIMPSES% ^
	--write-n %WRITE_N% ^
	--read-n %READ_N% 


python %PROCESS_DIR%\animate_all.py --job-name %NAME%%CHECK_NUM% ^
	--z-size %ZSIZE% ^
	--T %GLIMPSES% ^
	--write-n %WRITE_N% ^
	--read-n %READ_N% 
	
echo Produce Read Filter Gif
python %PROCESS_DIR%\gifit.py --dir reads_frames --out %GIF_DIR%\read_filter%CHECK_NUM% --frames %GLIMPSES%
echo Produce Write Filter Gif
python %PROCESS_DIR%\gifit.py --dir writes_frames --out %GIF_DIR%\write_filter%CHECK_NUM% --frames %GLIMPSES%	


SET CHECK_NUM=-10000
python execute.py --job-dir %DIR% --model-name %NAME%%CHECK_NUM% ^
	--z-size %ZSIZE% ^
	--T %GLIMPSES% ^
	--write-n %WRITE_N% ^
	--read-n %READ_N% 


python %PROCESS_DIR%\animate_all.py --job-name %NAME%%CHECK_NUM% ^
	--z-size %ZSIZE% ^
	--T %GLIMPSES% ^
	--write-n %WRITE_N% ^
	--read-n %READ_N% 
	
echo Produce Read Filter Gif
python %PROCESS_DIR%\gifit.py --dir reads_frames --out %GIF_DIR%\read_filter%CHECK_NUM% --frames %GLIMPSES%
echo Produce Write Filter Gif
python %PROCESS_DIR%\gifit.py --dir writes_frames --out %GIF_DIR%\write_filter%CHECK_NUM% --frames %GLIMPSES%	


SET CHECK_NUM=-5000
python execute.py --job-dir %DIR% --model-name %NAME%%CHECK_NUM% ^
	--z-size %ZSIZE% ^
	--T %GLIMPSES% ^
	--write-n %WRITE_N% ^
	--read-n %READ_N% 


python %PROCESS_DIR%\animate_all.py --job-name %NAME%%CHECK_NUM% ^
	--z-size %ZSIZE% ^
	--T %GLIMPSES% ^
	--write-n %WRITE_N% ^
	--read-n %READ_N% 
	
echo Produce Read Filter Gif
python %PROCESS_DIR%\gifit.py --dir reads_frames --out %GIF_DIR%\read_filter%CHECK_NUM% --frames %GLIMPSES%
echo Produce Write Filter Gif
python %PROCESS_DIR%\gifit.py --dir writes_frames --out %GIF_DIR%\write_filter%CHECK_NUM% --frames %GLIMPSES%	



SET CHECK_NUM=-15000
python execute.py --job-dir %DIR% --model-name %NAME%%CHECK_NUM% ^
	--z-size %ZSIZE% ^
	--T %GLIMPSES% ^
	--write-n %WRITE_N% ^
	--read-n %READ_N% 


python %PROCESS_DIR%\animate_all.py --job-name %NAME%%CHECK_NUM% ^
	--z-size %ZSIZE% ^
	--T %GLIMPSES% ^
	--write-n %WRITE_N% ^
	--read-n %READ_N% 
	
echo Produce Read Filter Gif
python %PROCESS_DIR%\gifit.py --dir reads_frames --out %GIF_DIR%\read_filter%CHECK_NUM% --frames %GLIMPSES%
echo Produce Write Filter Gif
python %PROCESS_DIR%\gifit.py --dir writes_frames --out %GIF_DIR%\write_filter%CHECK_NUM% --frames %GLIMPSES%	
