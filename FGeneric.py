import os
import gc
import pandas
from Configuration import Configuration
from predictors.sticker_fusionPredictor import sticker_fusionPredictor
from Service.Utilities import Tools
import numpy
from dataman.DataManager import DataManager
import traceback
import itertools

class FGeneric(object):
    """
    FGeneric is a base class providing generic methods for data processing and model execution.
    """
    
    DISCR_PRED_COL='Discr_Predict_'
    DISCR_VAL_COL='Discr_Validate_'

#     TR1=[0.45,0.5,0.55] 
#     TR2=[0.45,0.5,0.55] 
    
    TR1=[0.01] #[0.25,0.3,0.35] 
    TR2=[0.01] #[0.25,0.3,0.35] 
    
#     TR1=[round(0.2*i,2) for i in range(1,7)]
#     TR2=[round(0.2*i,2) for i in range(1,7)]
    
    def __init__(self,dataManager):
        
        self.bestPredictors={}
        
        self.predictorList = []
        self.dataManager = dataManager
        self.stickers = []
        
        self.bestPredictPower={}
        self.bestPredictPowerA={}
        
        self.predDict={}
        
        self.repman = None
        self.splitpoints={}
        
        for alg in Configuration.ALGS:
            a=alg.split('-')[0]
            self.bestPredictPowerA[a]={}
            
        
            
        for st in dataManager.stickers:
            for alg in Configuration.ALGS:
                dirc=1
                if '-U' in alg:
                    dirc=-1
                alg0=alg.split('-')[0]          
                sp = sticker_fusionPredictor(dataManager,st,predTag=alg,dirct=dirc)
                self.predictorList.append(sp)
                pn = sp.name[8:]+'_'+alg
                self.predDict[pn] = sp
  
            self.stickers.append(st)
        
        #dsname = self.dataManager.datasource.split('_')[0]
        #self.crlogname = os.path.join(os.getcwd(),'correlation_'+dsname+'.csv')
        
        
        print('FGeneric constructed\n')

    def setReportMan(self,rm):
        """
        Set the report manager.
        
        Parameters:
        rm (ReportManager): The report manager to set.
        """
        self.repman=rm
        
        for sp in self.predictorList:
            sp.setReportMan(rm)
    
    def logCorrEntry(self,forday):
        """
        Log correlation entry for the given day.
        
        Parameters:
        forday (str): The day for which to log the correlation entry.
        """
        for alg in Configuration.ALGS:
            self.logCorrEntryA(forday, alg)
        
    
    
    def logCorrEntryA(self,forday,alg):
        """
        Log correlation entry for a specific algorithm.
        
        Parameters:
        forday (str): The day for which to log the correlation entry.
        alg (str): The algorithm identifier.
        """   
        corrcolumns=[]
        ALGS=[alg]
        LA = len(ALGS)
        TRAIN=False
        if self.dataManager.datesList.index(forday)<Configuration.TSET:
            TRAIN=True
        
        for st in self.stickers:
            
            pred_captions = [st+'_' + ph for ph in ALGS]
            
#             pred_captions1 = [st+'_' + ph+"-U" for ph in Configuration.ALGS]
#             pred_captions=pred_captions+pred_captions1
            
            corrcolumns = corrcolumns + ['Pred_'+cp for cp in pred_captions]
            corrcolumns = corrcolumns + ['Close-'+st]
            corrcolumns = corrcolumns + [FGeneric.DISCR_PRED_COL+cp for cp in pred_captions]
            corrcolumns = corrcolumns + [FGeneric.DISCR_VAL_COL+cp for cp in pred_captions]
            corrcolumns = corrcolumns + ['SR_'+cp for cp in pred_captions]
                 
            corrcolumns = corrcolumns + ['G1_'+cp for cp in pred_captions]
            corrcolumns = corrcolumns + ['G2_'+cp for cp in pred_captions]
            corrcolumns = corrcolumns + ['G3_'+cp for cp in pred_captions]
            
            
            corrcolumns = corrcolumns + ['Skip_'+cp for cp in pred_captions]
            corrcolumns = corrcolumns + ['TP_'+cp for cp in pred_captions]
            corrcolumns = corrcolumns + ['RP_'+cp for cp in pred_captions]
        
        dsname = self.dataManager.datasource.split('_')[0]
        
        fname = os.path.join(os.getcwd(),'correlation_'+alg+'_'+dsname+'.csv')
        if os.path.exists(fname)==False:  
            f=open(fname,'wt')
            f.write(','.join(['date']+corrcolumns)+'\n')
            f.close()
        
        df = pandas.read_csv(fname,names=corrcolumns,index_col=0,header=0)
            
        row=[]

        
        prevts = Tools.nextBusDay(forday,-1)#self.dataManager.prevTS(forday)
        prevts_1 = Tools.nextBusDay(prevts,-1) #self.dataManager.prevTS(prevts)
        ib=[pandas.Timestamp(t) for t in df.index.values]
        
        badnames=[]
        badchars=[]
        if len(df)>1 and pandas.Timestamp(prevts) in ib:
            ipr=ib.index(pandas.Timestamp(prevts))
            row_1 = df.iloc[ipr,:].values
            cv_base=0
            for i in range(0,len(self.stickers)):
                cprev = self.dataManager.getAt(self.stickers[i], 'close', prevts)
                cprev_1 = self.dataManager.getAt(self.stickers[i], 'close', prevts_1)
#                 cprev = self.dataManager.globalDatasource.getAt(self.stickers[i], 'close', prevts)
#                 cprev_1 = self.dataManager.globalDatasource.getAt(self.stickers[i], 'close', prevts_1)
                real_c = Tools.rel(cprev,cprev_1)
                
                cv_index = cv_base+LA
                
                row_1[cv_index] = cprev
                cv_index=cv_index+1
                
                for cvi in range(0,LA):
                    prediction_c = int(row_1[cv_index+cvi])
                
                    if numpy.fabs(prediction_c)<=1 and prediction_c*(real_c-1)>0:
                        val = 1
                    else:
                        val = 0
                    row_1[cv_index+cvi+LA]=val
                    
                cv_base=cv_base+10*LA+1 #10 for Gain=(g1,g2,g3), 12 for (g1..g5)
            
            df.loc[prevts] = row_1
                     
        
        for st in self.stickers:
            
            cprev = self.dataManager.getAt(st, 'close', prevts)
            #cprev = self.dataManager.globalDatasource.getAt(st, 'close', prevts)
            
            
            pred_captions = [st+'_' + ph for ph in ALGS]
    
#             pred_captions1 = [st+'_' + h+"-U" for ph in Configuration.ALGS]
#             pred_captions=pred_captions+pred_captions1
            
            preds = [self.predDict[pnc] for pnc in pred_captions]
            
            predict_raws=[p.getPrediction(forday,st) for p in preds] 
            predict_skips = [p.getSkip(forday,st) for p in preds]
            
            predictions = []
                
            for i in range(0,len(predict_raws)):
                p=predict_raws[i]
                
                if TRAIN:
                    predictions.append(0)
            
                    continue
                    
                if int(p)==100:
                    predictions.append(0)

                else:
                    predictions.append(p)
            
            row = row+[str(q) for q in predictions]
            row = row + [str(0.0)]
            
            preddeltas = Tools.delta(predictions,cprev)
        
            predsrs=[]
    
            predsigns = [100]*len(predictions)
            predskips = ['Y']*len(predictions)
            
            predtps = []
            predrps=[]
            
            for i in range(0,len(predictions)):
                x = predictions[i]
                sk = predict_skips[i]
                pcap = pred_captions[i]
                sr=0.0
                g=[0]*3 #Configuration.validationHistory   #(0,0,0)
                tp=0.0
                rp=0.0
                di=0
                try:
                    if forday in self.dataManager.globalDatasource.datesList:
                        di=self.dataManager.globalDatasource.datesList.index(forday)
                    else:
                        di=self.dataManager.datesList.index(forday)
                except:
                    pass
                
                
                if TRAIN:
                    predsigns[i]=0
                else:
                    predsigns[i] = Tools.growth(x)
                    
                    
                if di<Configuration.validationHistory+Configuration.TSET:
                    bern_pass=1
                else:
                    #g=[1]*5
                    bern_pass=1
                    sr=1.0
                    tp=1.0
                    rp=1.0
                    
                    if sk==False:
                        (g,sr,tp,rp,bern_pass) = self.bernoulliAccept(forday, pcap,predsigns[i])
                    else:
                        bern_pass=0
                    (g1,g2,g3)=g
                
                predsrs.append(sr)
                
                for e in g:
                    predsrs.append(e)
                
                predtps.append(tp)
                predrps.append(rp)
                
                if numpy.isnan(x) or numpy.isinf(x):
                    predsigns[i] = 100
                    continue
                
                if bern_pass>0:
                    predskips[i] = '-'
                    if bern_pass>1:
                        predskips[i] = '+'
                
                
            
            validations=[0]*len(preddeltas)
                
            row = row + [str(d) for d in predsigns]
            row = row + [str(d) for d in validations]
            row = row + [str(d) for d in predsrs]
            row = row + [d for d in predskips]
            row = row + [str(d) for d in predtps]
            row = row + [str(d) for d in predrps]
            
        if len(df)>0:
            df.loc[forday]=row
            df.to_csv(fname)
        else:
            f1=open(fname,'at')
            f1.write(','.join([forday]+row)+'\n')
            f1.close()
    

                

    def chooseBests(self,forday):
        """
        Choose the best options for the given day.
        
        Parameters:
        forday (str): The day for which to choose the best options.
        """
        pass
       
    
    def refresh(self):
        """
        Refresh the data and predictors.
        """
        splitpoints = self.dataManager.datesList[Configuration.TSET:]
    
        dsname = self.dataManager.datasource.split('_')[0]
        
        print('Refresh started\n')
        
#         for pindex,p in enumerate(splitpoints):
#             self.splitpoints[p] = pindex
            
        p=self.dataManager.datesList[-1]    
 
        for pred in self.predictorList:
            pred.refresh(p)
                
    
    def runseq(self):
        """
        Run the sequence of data processing and model execution steps.
        """
        
        dsname = self.dataManager.datasource.split('_')[0]
        
        print('Fuser started\n')
        
        splitpoints = self.dataManager.datesList[Configuration.TSET:]
        for pindex,p in enumerate(splitpoints):
            self.splitpoints[p] = pindex

         
        buffered=True
        for alg in Configuration.ALGS:
            sn = os.path.join(os.getcwd(),'correlation_'+alg+'_'+dsname+'.csv')
            buffered=buffered and os.path.exists(sn)
            
        
          
        if buffered==False:
            i=0
            while i<Configuration.TSET:
                i=i+1
                
            while i<len(self.dataManager.datesList)-1:
                
                p=self.dataManager.datesList[i]
                p_1=self.dataManager.datesList[i-1]
                
                
                for pred in self.predictorList:
                    pred.runAll(p)
                   
                    
                self.logCorrEntry(p)
                i=i+1
        
        p=self.dataManager.datesList[-1]    
        p_1=self.dataManager.datesList[-2]
        

        for pred in self.predictorList:
            
            pred.runAll(p)
        
        self.logCorrEntry(p)
        
        

        try:
            self.chooseBests(p)
        except:
            traceback.print_exc()
            pass
        
        gc.collect()
     
    def bernoulliAccept(self,forday,sticker_alg,taus):
        """
        Accept Bernoulli trials for the given day and sticker algorithm.
        
        Parameters:
        forday (str): The day for which to accept Bernoulli trials.
        sticker_alg (str): The sticker algorithm.
        taus (list): List of taus.
        
        Returns:
        tuple: A tuple containing the results of the Bernoulli trials.
        """
        pass



