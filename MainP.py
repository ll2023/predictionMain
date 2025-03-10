from numpy.random import seed

seed(7856)
#from tensorflow import set_random_seed
#set_random_seed(7856)

import os
import sys
import warnings 
import gc
import time
import atexit 
import shutil
from itertools import chain, combinations
#import h2o
#from h2o.backend import H2OLocalServer

from Configuration import Configuration

from dataman.DataManager import DataManager
from ReportManager import ReportManager
from Fuser import Fuser

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def cleaner():
    gc.collect()

def runall(model):
    """
    Run the entire model fusion process.
    
    Parameters:
    model (object): The model to be used for fusion.
    """
    fusion = Fuser(model)
    start1 = time.time()
    repMan = ReportManager(fusion)
    fusion.runseq()
    end1 = time.time() - start1
    print('\n     Processed in {0} seconds'.format(str(end1)))

    for alg in Configuration.ALGS:
        a = alg.split('-')[0]
        repMan.reportAggregationA(a)
    
    gc.collect()

def refreshModel():
    """
    Refresh the model.
    """
    fusion = Fuser(model)
    fusion.refresh()
    
    gc.collect()
    
    print('\n     Model refresh ')

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
atexit.register(cleaner)

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
os.environ["PYTHONHASHSEED"] = '0'
os.environ["TF_CUDNN_USE_AUTOTUNE"] ="0"

# DEVICE=torch.device('cuda')
# x=torch.zeros(10000).to(DEVICE)
# y=torch.ones(10000).to(DEVICE)
# z=x.add(y)


index = sys.argv[1]

joinlog=None
updatelog=False
mergelog=False
mergelog1=False
updatepos=0
refreshmodel=False
preponline=False
beforeclose=False
genlogs=False
placeorders=False
printpos=False
cancelorders=False
wsums=False
genraws=False
hist=False
invest=False
stateoffs=0
ordsend=False
mergerep=False
closepos=False
invert=False
joinall=False
joinpred=None
joinintegr=None

(a,b,c) = gc.get_threshold()
gc.set_threshold(3000,1,1)


for arg in sys.argv:
    if '=' in arg:
        command = arg.split('=')[0]
        param = arg.split('=')[1]
        acct=param
        
        if command=='updatepos':
            if 'yes1' in param:
                updatepos=1
            if 'yes2' in param:
                updatepos=2
        
        if command=='preponline':
            preponline=True
          
        if command=='joinlog':
            joinlog=param
        
        if command=='joinall':
            joinall=True
        
        if command=='joinpred':
            joinpred=param
        
        if command=='joinmerge':
            joinintegr=param
        
        
        
Configuration.init()

model=DataManager(index)

if joinlog is not None:
    
    model.joinMaillogs(joinlog)
    
    sys.exit()
# else:
#     model.joinMaillogs("") 
#     sys.exit()

if joinall==True:
    model.joinMaillogsM()
    sys.exit()

if joinpred is not None:
    model.joinMaillogsP(joinpred)
    sys.exit()
     
if updatepos>0:
    model.updatePositionFileGen()
    sys.exit()
    
if preponline==True:
    model.prepOnline()
    sys.exit()

    
runall(model)


