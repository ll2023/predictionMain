
from Configuration import Configuration

import traceback
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import KFold
from dataman.DataManager import DataManager
from Service.Utilities import Tools
import os
import shutil
import numpy
import numpy as np

# import torch
# from torchvision import datasets, transforms
# from torch.utils.data import Dataset, DataLoader
# from torch import nn

class Predictor(object):
    
    def __init__(self, name, datamanager):
    
        self.dataManager = datamanager
    
        self.name=name
        self.preprocessed={}
       
        self.histlen = max(Configuration.seqhists)
         
        self.grid=None
    
        if len(self.stickers)==0:
            self.stickers = numpy.array(datamanager.stickers)
    
        c=len(self.stickers)
        r = len(datamanager.datesList)

        #self.predictions = [[0.0 for i in range(r)] for j in range(c)]
        self.predictions = numpy.zeros((c,r),numpy.float32)
        
        #self.confidences = [[0.0 for i in range(r)] for j in range(c)]
        
        self.confidences = numpy.zeros((c,r),numpy.float32)
        self.repman=None
        self.curParamConfig=None
        self.dmContext = None
        self.PATH=None
        self.OPTPATH=None
        self.corrdf=None
    
    def skl2torch(self,model):
        pass
        
#         criterion = nn.CrossEntropyLoss()
#         model.set_criterion(criterion)
#             
#         model.set_optimizer('Adam', lr=0.1, weight_decay=0.005)
#         model.set_scheduler("OneCycleLR",max_lr=0.1,epochs=50,steps_per_epoch=100)
    
    def saveFirstState(self,prd,sticker):
        ds=self.dmContext.datasource.split('_')[0]
        cname=sticker+'_'+prd+'_refreshtrain_'+ds+'.txt'
        st=os.path.join(os.getcwd(),"freestate")+os.path.sep+self.PATH
        ost=os.path.join(os.getcwd(),"freestate")+os.path.sep+self.OPTPATH
        countr=os.path.join(os.getcwd(),"freestate")+os.path.sep+cname
                
        if os.path.exists(st)==False and os.path.exists(self.PATH)==True:
            shutil.copy2(self.PATH, os.path.join(os.getcwd(),"freestate"))
                
        if os.path.exists(ost)==False and os.path.exists(self.OPTPATH)==True:
            shutil.copy2(self.OPTPATH, os.path.join(os.getcwd(),"freestate"))
                
        if os.path.exists(countr)==False and os.path.exists(cname)==True:
            with open(cname,'wt') as f:
                f.write(str(Configuration.validationHistory))
            shutil.copy2(cname, os.path.join(os.getcwd(),"freestate"))
            
    def clearModel(self):
        pass
    
    def saveModel(self,relft):
        pass
        
    def restoreModel(self):
        return (None,False,None)
        
    
    def retraining(self,prd,sticker,timestamp,rtr=Configuration.RETRL):
        
        ds=self.dataManager.datasource.split('_')[0]
        tsi=self.dataManager.datesList.index(timestamp)
        
        if tsi<max(Configuration.seqhists):# + Configuration.validationHistory:
            return False
        
        
        PATH = prd+'_'+sticker+'.pt'
        FPATH = 'ft'+prd+'_'+sticker+'.pt'
        
        spath = os.path.join(os.getcwd(), PATH)
        sfpath = os.path.join(os.getcwd(), FPATH)
        
        cname=sticker+'_'+prd+'_refreshtrain_'+ds+'.txt'
        scname = os.path.join(os.getcwd(), cname)
        
        if os.path.exists(spath)==False:
            with open(scname,'wt') as f:
                f.write(str(Configuration.RETRL))
                return True
        
#         df=pandas.read_csv('correlation_'+prd+'_'+ds+'.csv',index_col=0)
#                         
#         v=df['Discr_Validate_'+sticker+'_'+prd].values[max(0,tsi-50):tsi]
#         sk=df['Skip_'+sticker+'_'+prd].values[max(0,tsi-50):tsi]
#         tp=0
#         ap=0
#         sr=0.0
#         for j in range(0,len(v)):
#             if sk[j]=='-':
#                 ap+=1
#                 tp=tp+v[j]
#         if ap>0:
#             sr=1.0*tp/ap
            
        #print('Success rate on ' + sticker + ' at '+ timestamp + ' is ' + str(round(sr,2)) + ' history ' + str(tsi))
          
#         else:
#             if sr>0.55:
#                 #print(prd + ' GOOD '+str(sr) +' NO RETRAIN on ' + sticker + ' ' + timestamp)
#                 return False
  
        remainsteps=-1

        try:
            with open(scname,'rt') as f:
                remainsteps_t = f.read()
                remainsteps = int(remainsteps_t)
                remainsteps=remainsteps-1
        except:
            pass
          
        #print(prd + ' BAD SR ' + str(sr) + ' ' + sticker + ' ' +timestamp + ' ' + str(tsi), end= '...')
        if remainsteps<0:
            self.clearModel()
            with open(scname,'wt') as f:
                f.write(str(Configuration.RETRL))
            #print('.. reboot 50 ')
        else:
            with open(scname,'wt') as f:
                f.write(str(remainsteps))
            #print('.. decrease to  '+ str(remainsteps))
              
        return remainsteps<0
           
        
    def setReportMan(self,rm):
        self.repman=rm
    
    def minConfidence(self,nm):
        if 'KNeighbors' in nm:
            return 0.5
        if 'SVC' in nm or 'SVR' in nm:
            return 0.52
        if 'Complement' in nm:
            return 0.5
        if 'QuadraticDiscriminant' in nm:
            return 0.55
        if 'seq' in nm:
            return 0.6
        if 'XGB' in nm:
            return 0.5
        if 'SGD' in nm:
            return 0.55
        return 0.6
    
    def adaptPredictor(self,engine):
        
        
#         def precision_metric(y_true, y_pred):
#             ml=min(len(y_pred),len(y_true))
#             val_growths=0
#             
#             for i in range(0,ml):
#                 pg=Tools.growth(y_pred[i])
#                 tg=Tools.growth(y_true.values[i])
#                 val_growths=val_growths+max(0,pg*tg)
#                 
#             return float(val_growths/ml)
        
        adaptEng = Configuration.adaptEngine[type(engine).__name__]
        try:
            if self.grid is None:
                self.curParamConfig = Configuration.paramGrids[type(engine).__name__]
            else:
                self.curParamConfig = self.grid
            #https://stackoverflow.com/questions/42011850/is-there-a-way-to-see-the-folds-for-cross-validation-in-gridsearchcv
            iterengine = GridSearchCV(engine, self.curParamConfig,cv=KFold(n_splits=10),scoring=adaptEng,verbose=0) #'neg_mean_squared_error'
        except:
            traceback.print_exc()
            return engine
            
        return iterengine
        
    
    def predict(self,sticker,timestamp):
        return (100,(0,0),0.0)
    
    def runAll(self,forday):
        
        self.preprocess(forday)
        
    
    def getSkip(self,timestamp,sticker):
        timestamp_index=-1
        try:
            timestamp_index = self.dataManager.dateindex[timestamp]
        except:
            pass
        
        if timestamp_index<0:
            return True
        
        if len(self.stickers)==1:
            sticker_index=0
        else:
            sticker_index = np.where(self.stickers==sticker)[0]
        
        pt = self.predictions[sticker_index][timestamp_index]
        if isinstance(pt,int) and pt==100:
            return True
        
        r = self.confidences[sticker_index][timestamp_index]
        return r<0 # Configuration.BERNSR
    
    def sticker(self):
        return self.stickers[0]

    
    def getConfidence(self,timestamp,sticker):

        timestamp_index=-1
        try:
            timestamp_index = self.dataManager.dateindex[timestamp]
        except:
            pass
        
        if timestamp_index<0:
            return 100
        
        if len(self.stickers)==1:
            sticker_index=0
        else:
            sticker_index = np.where(self.stickers==sticker)[0]
        
        r = float(self.confidences[sticker_index][timestamp_index])
        return r
    
    def getPrediction(self,timestamp,sticker):

        timestamp_index=-1
        try:
            timestamp_index = self.dataManager.dateindex[timestamp]
        except:
            pass
        
        if timestamp_index<0:
            return 100
        
        if len(self.stickers)==1:
            sticker_index=0
        else:
            sticker_index = np.where(self.stickers==sticker)[0]
        
        r = self.predictions[sticker_index][timestamp_index]
        return r
    
    def setPrediction(self,timestamp,sticker,prediction): 
        
        timestamp_index=-1
        try:
            timestamp_index = self.dataManager.dateindex[timestamp]
        except:
            pass
        
        if timestamp_index<0:
            return 
        
        
        if len(self.stickers)==1:
            sticker_index=0
        else:
            sticker_index = np.where(self.stickers==sticker)[0]
            
        self.predictions[sticker_index][timestamp_index] = prediction
    
    def setConfidence(self,timestamp,sticker,confidence): 
        
        timestamp_index=-1
        try:
            timestamp_index = self.dataManager.dateindex[timestamp]
        except:
            pass
        
        if timestamp_index<0:
            return 
        
        
        if len(self.stickers)==1:
            sticker_index=0
        else:
            sticker_index = np.where(self.stickers==sticker)[0]
            
        self.confidences[sticker_index][timestamp_index] = float(confidence)
    

    def preprocess(self,forday=None):
        
        sticker = self.sticker()
        if forday is not None:
            start_index=-1
            try:
                start_index = self.dataManager.dateindex[forday]
            except:
                pass
            if start_index<0:
                return
            if start_index<self.startOffset:
                return
            
        tsRange=[forday]
        if forday is None:
            tsRange=self.dataManager.datesList[self.startOffset:]
        
        h=self.histlen
    
        for sticker in self.stickers:
    
            
            for timestamp in tsRange:
                
#                 closes=self.dataManager.get(sticker,'close',timestamp,h)
#                 volumes = self.dataManager.get(sticker,'volume',timestamp,h)
#                 highs = self.dataManager.get(sticker,'high',timestamp,h)
#                 lows = self.dataManager.get(sticker,'low',timestamp,h)
#                 opens = self.dataManager.get(sticker,'open',timestamp,h)
                
                try: 
                    #print(sticker + ' ' + timestamp)
                    r = self.predict(sticker,timestamp)
                    
                    (decision,confidence,param)=r

                    dinv=decision
                    
                    self.setPrediction(timestamp, sticker,dinv)
                    #self.setPrediction(timestamp, sticker,decision)
                    self.setConfidence(timestamp, sticker, param)
                except:
                    decision=100
                    self.setPrediction(timestamp, sticker,decision)
                    traceback.print_exc()
                    pass
                    
#                     
#                 del closes
#                 del highs
#                 del lows
#                 del volumes
                
        
       
        self.preprocessed[forday]=True

            
                                
            
        
       

        
