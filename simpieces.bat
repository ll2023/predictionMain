@ECHO OFF
setlocal enabledelayedexpansion
del last-positions.csv
del final-report.csv

del *joint_maillog.csv
del *jointT_maillog.csv


python MainP.py sandprecent joinpred=ada
python MainP.py sandprecent joinpred=brf
python MainP.py sandprecent joinpred=gbf
python MainP.py sandprecent joinpred=eec
python MainP.py sandprecent joinpred=tsf

python MainP.py sandprecent joinlog=eec
python MainP.py sandprecent joinlog=brf
python MainP.py sandprecent joinlog=ada
python MainP.py sandprecent joinlog=gbf
python MainP.py sandprecent joinlog=tsf

python MainP.py sandprecent updatepos=yes1






