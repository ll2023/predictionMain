
del *.pt
rem del *.csv
del *.json
del *.txt

FOR /L %%A IN (60,1,200) DO (
  python MainP.py sandprecentpiece1_temp%%A
)


