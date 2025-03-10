import os
import sys
import shutil
PIECESIZE=34

from Configuration import Configuration
Configuration.init()


if len(sys.argv)>=2:
    PIECESIZE = int(sys.argv[2])
    
fulldir = os.path.join(os.getcwd(),sys.argv[1])

if os.path.exists(fulldir)==False:
    sys.exit(0)
    
filelist = [ f for f in os.listdir(fulldir) ]
for fl in filelist:
    fname = os.path.join(fulldir,fl)
    fh = open(fname)
    fsz = len(fh.readlines())
    fh.close()
    if fsz<max(Configuration.seqhists) + Configuration.validationHistory:
        print('Too short: ' + fl)
        filelist.remove(fl)
        os.remove(fname)


filelist = sorted(filelist)
print(filelist)
piece=1
piecestart=0
while piecestart<len(filelist):
    pieceend=min(len(filelist),piecestart+PIECESIZE)
    partd = os.path.join(os.getcwd(),sys.argv[1]+"piece"+str(piece))
    os.makedirs(partd)
    
    print('Piece ' + str(piece), end=' : ')
    
    for i in range(piecestart,pieceend):
        filename=filelist[i]
        print(filename,end=' ')
        srcf=os.path.join(fulldir,filename)
        dstf=os.path.join(partd,filename)
        shutil.copy(srcf,dstf)
    print('')
        
    piecestart=pieceend
    piece=piece+1
