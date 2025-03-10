import os
import sys
import pandas
from Configuration import Configuration

subd=''
testmode=False
if len(sys.argv)>2:
    testmode=True
    
Configuration.init()
fulldir = os.path.join(os.getcwd(),sys.argv[1])


if os.path.exists(fulldir)==False:
    sys.exit(0)
    
filelist = [ f for f in os.listdir(fulldir) ]

for t in os.listdir(os.path.join(os.getcwd(),subd)):
    if sys.argv[1]+"_temp" in t:
        partdir = os.path.join(os.getcwd(),t)
        if os.path.isdir(partdir)==True and os.path.exists(partdir)==True:
            filepart = [ f for f in os.listdir(partdir) ]
            for f in filepart:
                os.remove(os.path.join(partdir, f))
            os.rmdir(partdir)




samplelen=Configuration.seloffset

sample=0

dslen=max([len(pandas.read_csv(os.path.join(fulldir, fl),sep=' ').values) for fl in filelist])

while sample+samplelen<=dslen+1:
    
    partdir = os.path.join(os.getcwd(),subd,sys.argv[1]+"_temp"+str(sample+1))
    os.makedirs(partdir)
        
    for f in filelist:
        dataset = os.path.join(fulldir, f)
        reducedset = os.path.join(partdir, f)
    
        df = pandas.read_csv(dataset,sep=' ',header=None)
        
        if testmode:
            df_reduced  = df.iloc[0:(sample+samplelen)]
        else:
            df_reduced  = df.iloc[sample:(sample+samplelen)]
        
        df_reduced.to_csv(reducedset,sep=' ',index=False,header=False)
    sample=sample+1

f=open('chunk.txt','w')
f.write(str(sample))
f.close()

sys.exit(sample)
    
