@ECHO OFF
setlocal enabledelayedexpansion
del last-positions.csv
del final-report.csv
del *sandprecent*.csv
del *join*.csv
rd /s /q sandprecent
del *.json
del *.txt
rem copy /Y slogs_after\sandp28.csv .

copy /Y *positions*.csv slogs_after
del slogs_after\sandprecent\*.csv
del slogs_after\*sandprecent*.csv


python spdownl-100.py hist 28 375
rename sandprecentHIST sandprecent

python splitpieces.py sandprecent 4

python split.py sandprecent

python split.py sandprecentpiece1
python split.py sandprecentpiece2
python split.py sandprecentpiece3
python split.py sandprecentpiece4
python split.py sandprecentpiece5

FOR /L %%A IN (1,1,10) DO (
  python MainP.py sandprecentpiece1_temp%%A 
  python MainP.py sandprecentpiece2_temp%%A 
  python MainP.py sandprecentpiece3_temp%%A 
  python MainP.py sandprecentpiece4_temp%%A 
  python MainP.py sandprecentpiece5_temp%%A 
)


call python MainP.py sandprecentpiece1 joinall=yes
call python MainP.py sandprecentpiece2 joinall=yes
call python MainP.py sandprecentpiece3 joinall=yes
call python MainP.py sandprecentpiece4 joinall=yes
call python MainP.py sandprecentpiece5 joinall=yes
call python MainP.py sandprecent joinlog=yes


FOR /L %%A IN (1,1,10) DO (

	echo %%A
	call python MainP.py sandprecent_temp%%A mergelog=yes
	IF EXIST last-positions.csv (
		call python MainP.py sandprecent updatepos=yes1
	)
)

FOR /L %%A IN (1,1,10) DO (
  rd /s /q sandprecentpiece1_temp%%A 
  rd /s /q  sandprecentpiece2_temp%%A 
  rd /s /q  sandprecentpiece3_temp%%A 
  rd /s /q  sandprecentpiece4_temp%%A 
  rd /s /q  sandprecentpiece5_temp%%A 
)

rd /s /q sandprecentpiece1
rd /s /q sandprecentpiece2
rd /s /q sandprecentpiece3
rd /s /q sandprecentpiece4
rd /s /q sandprecentpiece5

FOR /L %%A IN (1,1,10) DO (
   echo %%A
   rd /s /q sandprecent_temp%%A
)


copy /Y *sandprecent*.csv slogs_after
copy /Y sandprecent\*.csv slogs_after\sandprecent
del final-report.csv
del accum-positions.csv






