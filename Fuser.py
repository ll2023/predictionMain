import os
import gc
import pandas as pd
from Configuration import Configuration
from Service.Utilities import Tools
import numpy as np
import FGeneric 
from dataman.DataManager import DataManager
import pywt
import talib
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Dict, List, Tuple, Optional


class Fuser(FGeneric.FGeneric):
    """Enhanced Fuser class with proper logging and type hints"""
    
    DISCR_PRED_COL='Discr_Predict_'
    DISCR_VAL_COL='Discr_Validate_'
    BBOLD = 0.55
    
    def __init__(self,dataManager):
        """
        Initialize the Fuser class with a data manager.
        
        Parameters:
        dataManager (DataManager): The data manager to use.
        """
        super().__init__(dataManager)
        self.cache = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Add file handler
        fh = logging.FileHandler('fuser.log')
        fh.setLevel(logging.INFO)
        self.logger.addHandler(fh)
    
    
    def genTADF(self, forday: str, st: str, pred: List[float], val: List[float]) -> Optional[pd.DataFrame]:
        """Generate TADF with enhanced error handling and logging"""
        try:
            cache_key = f"{forday}_{st}"
            if cache_key in self.cache:
                return self.cache[cache_key]
                
            # Parallel data fetching
            futures = []
            for data_type in ['open', 'high', 'low', 'close', 'volume']:
                future = self.executor.submit(
                    self.dataManager.globalDatasource.get,
                    st, data_type, forday, 100
                )
                futures.append(future)
            
            # Gather results
            data = [f.result() for f in futures]
            
            # Vectorized operations using numpy
            dfloc = pd.DataFrame(np.column_stack(data),
                               columns=['open', 'high', 'low', 'close', 'volume'])
            
            # Cache results
            self.cache[cache_key] = dfloc
            
            for col in dfloc.columns:
                x = dfloc[col].values
                ca, cd = pywt.dwt(x, "coif4")
                cat = pywt.threshold(ca, 1.5 * np.std(ca), mode="hard")
                cdt = pywt.threshold(cd, 1.5 * np.std(cd), mode="hard")
                dfloc[col] = pywt.idwt(cat, cdt, "coif4")
            
            openhist=dfloc['open']
            highhist=dfloc['high']
            lowhist=dfloc['low']
            closehist=dfloc['close']
            volhist=dfloc['volume']
            
            dfloc['3crows1'] = talib.CDL2CROWS(openhist,highhist,lowhist,closehist)
            dfloc['3crows2'] = talib.CDL3BLACKCROWS(openhist,highhist,lowhist,closehist)
            dfloc['3inside'] = talib.CDL3INSIDE(openhist,highhist,lowhist,closehist)
            dfloc['3linestrike'] = talib.CDL3LINESTRIKE(openhist,highhist,lowhist,closehist)
            dfloc['3outside'] = talib.CDL3OUTSIDE(openhist,highhist,lowhist,closehist)
            dfloc['3stsouth'] = talib.CDL3STARSINSOUTH(openhist,highhist,lowhist,closehist)
            dfloc['3wsoldiers'] = talib.CDL3WHITESOLDIERS(openhist,highhist,lowhist,closehist)
            dfloc['3abandonedbaby'] = talib.CDLABANDONEDBABY(openhist,highhist,lowhist,closehist)
            dfloc['3advblock'] = talib.CDLADVANCEBLOCK(openhist,highhist,lowhist,closehist)
            dfloc['3belthold'] = talib.CDLBELTHOLD(openhist,highhist,lowhist,closehist)
            dfloc['3CONCEALBABYSWALL'] = talib.CDLCONCEALBABYSWALL(openhist,highhist,lowhist,closehist)
            dfloc['3CNTATACK'] = talib.CDLCOUNTERATTACK(openhist,highhist,lowhist,closehist)
            dfloc['3DOJI'] = talib.CDLDOJI(openhist,highhist,lowhist,closehist)
            dfloc['3DOJIST'] = talib.CDLDOJISTAR(openhist,highhist,lowhist,closehist)
            dfloc['3ENGULF'] = talib.CDLENGULFING(openhist,highhist,lowhist,closehist)
            dfloc['3EVST'] = talib.CDLEVENINGSTAR(openhist,highhist,lowhist,closehist)
            dfloc['3HAMMER'] = talib.CDLHAMMER(openhist,highhist,lowhist,closehist)
            dfloc['3HARM'] = talib.CDLHARAMI(openhist,highhist,lowhist,closehist)
            dfloc['3HANGMAN'] = talib.CDLHANGINGMAN(openhist,highhist,lowhist,closehist)
            dfloc['3HOMINGPIGEON'] = talib.CDLHOMINGPIGEON(openhist,highhist,lowhist,closehist)
            dfloc['3DLINNECK'] = talib.CDLINNECK(openhist,highhist,lowhist,closehist)
            dfloc['3DKICK'] = talib.CDLKICKING(openhist,highhist,lowhist,closehist)
            dfloc['3DKICKLEN'] = talib.CDLKICKINGBYLENGTH(openhist,highhist,lowhist,closehist)
            dfloc['3DLONGLINE'] = talib.CDLLONGLINE(openhist,highhist,lowhist,closehist)
            dfloc['3DMARUB'] = talib.CDLMARUBOZU(openhist,highhist,lowhist,closehist)
            dfloc['3DPIERCE'] = talib.CDLPIERCING(openhist,highhist,lowhist,closehist)
            dfloc['3DHIK'] = talib.CDLHIKKAKE(openhist,highhist,lowhist,closehist)
            dfloc['3DHIKMOD'] = talib.CDLHIKKAKEMOD(openhist,highhist,lowhist,closehist)
            dfloc['3DHIKPIG'] = talib.CDLHOMINGPIGEON(openhist,highhist,lowhist,closehist)
            dfloc['3DID3CROWS'] = talib.CDLIDENTICAL3CROWS(openhist,highhist,lowhist,closehist)
            dfloc['3DTICKSAND'] = talib.CDLSTICKSANDWICH(openhist,highhist,lowhist,closehist)
            dfloc['3DTAKURI'] = talib.CDLTAKURI(openhist,highhist,lowhist,closehist)
            dfloc['3DTRISTAR'] = talib.CDLTRISTAR(openhist,highhist,lowhist,closehist)
            dfloc['3DTHRUST'] = talib.CDLTHRUSTING(openhist,highhist,lowhist,closehist)
            dfloc['3DSHORTLINE'] = talib.CDLSHORTLINE(openhist,highhist,lowhist,closehist)
            dfloc['3DSTALLED'] = talib.CDLSTALLEDPATTERN(openhist,highhist,lowhist,closehist)
            dfloc['3DTASUKI'] = talib.CDLTASUKIGAP(openhist,highhist,lowhist,closehist)
            dfloc['3DUNRIVER'] = talib.CDLUNIQUE3RIVER(openhist,highhist,lowhist,closehist)
            dfloc['3DUPSGAPCROW'] = talib.CDLUPSIDEGAP2CROWS(openhist,highhist,lowhist,closehist)
            dfloc['3DCDLXSIDEGAP3METHODS'] = talib.CDLXSIDEGAP3METHODS(openhist,highhist,lowhist,closehist)
            dfloc['3DKICKLEN'] = talib.CDLKICKINGBYLENGTH(openhist,highhist,lowhist,closehist)
            dfloc['3DINVHAM'] = talib.CDLINVERTEDHAMMER(openhist,highhist,lowhist,closehist)
            dfloc['3DHARCROSS'] = talib.CDLHARAMICROSS(openhist,highhist,lowhist,closehist)
            dfloc = dfloc.fillna(0)
            dfloc = dfloc.drop(columns=['open', 'high','low','close','volume'])
            
            
            if os.path.exists(st+'_candles.csv')==False:
                with open(st+'_candles.csv','wt') as chf:
                    chf.write('date,pred,val,' + ','.join(list(dfloc.columns))+'\n')
                    chf.close()
            else:
                with open(st+'_candles.csv','at') as chf:
                    p=[forday,str(pred),str(int(max(0,val)))]
                    p1=[str(e) for e in dfloc.iloc[-1,:].values]
                    chf.write(','.join(p+p1)+'\n')
                    chf.close()
        except Exception as e:
            self.logger.error(f"Error in genTADF: {e}", exc_info=True)
            return None

    def chooseBests(self, forday):
        """
        Choose the best options for the given day.
        
        Parameters:
        forday (str): The day for which to choose the best options.
        """
        try:
            self.bestPredictPower[forday] = []
            for alg in Configuration.ALGS:
                a = alg.split('-')[0]
                self.bestPredictPowerA[a][forday] = []
                
            prevday = Tools.nextBusDay(forday, -1)
            dsname = self.dataManager.datasource.split('_')[0]
            
            for st in self.stickers:
                optpred = 100
                optsr = -1000.0
                optg1 = -1000.0
                optg3 = -1000.0
                optpcap = '-'
                optswitch = False
                
                pred_captions = [st + '_' + ph for ph in Configuration.ALGS]
                predictions = [Fuser.DISCR_PRED_COL + cp for cp in pred_captions]
                
                for i in range(0, len(predictions)):
                    pcap = pred_captions[i]
                    alg = pcap.split('_')[1]
                    alg_1 = alg.split('-')[0] + '-' + alg.split('-')[1]
                    fname = os.path.join(os.getcwd(), 'correlation_' + alg_1 + '_' + dsname + '.csv')
                    df = pd.read_csv(fname, index_col=0, header=0)
                    
                    x = df.at[forday, predictions[i]]
                    if x == 0:
                        continue
            
                    optpred = x
                    optpcap = pcap
                            
                    a = optpcap.split('_')[1]
                    a_1 = a.split('-')[0]
                            
                    confidence = (1, 0)

                    try:
                        sr = df.at[prevday, 'SR_' + pcap]
                        g = df.at[prevday, 'G3_' + pcap]
                        param = (sr, g)
                    except:
                        param = (0, 0)
                    self.bestPredictPowerA[a_1][forday].append((st, optpred, pcap, param))
        except Exception as e:
            print(f"Error in chooseBests: {e}")

    def bernoulliAccept(self, forday, sticker_alg, taus_=None):
        """
        Accept Bernoulli trials for the given day and sticker algorithm.
        
        Parameters:
        forday (str): The day for which to accept Bernoulli trials.
        sticker_alg (str): The sticker algorithm.
        taus_ (list, optional): List of taus. Defaults to None.
        
        Returns:
        tuple: A tuple containing the results of the Bernoulli trials.
        """
 
       
        delta=0.2

        stick = sticker_alg.split('_')[0]
         
        alg = sticker_alg.split('_')[1]
        alg1=alg.split('-')[0]+'-'+alg.split('-')[1]
        dsname = self.dataManager.datasource.split('_')[0]
        sname = os.path.join(os.getcwd(),DataManager.SUBDIR,'correlation_'+alg1+'_'+dsname+'.csv')
        CVH=5
      
        
        df=None
         
        try:
            df = pd.read_csv(sname,header=0,index_col=0)
        except:
            #print('NO CORREL FILE ' + forday + ' ' + sticker_alg)
            return ((0,0,0),0,0,0,1)
        
        if len(df)<CVH:
            return ((0,0,0),0,0,0,1)
         
        pcol = df[Fuser.DISCR_PRED_COL+sticker_alg].values
        vcol = df[Fuser.DISCR_VAL_COL+sticker_alg].values
        g_col = self.dataManager.globalDatasource.get(stick, 'close', forday,len(pcol)) 
        gd_col = Tools.rels(g_col)
        G=[Tools.growth(g) for g in gd_col]
        if len(G)<CVH:
            return ((0,0,0),0,0,0,1)
         
        G = G[-CVH:]
        Gp = list(filter(lambda p:p>0,G))
        Gn = list(filter(lambda p:p<0,G))
        GPN=len(Gp)+len(Gn)
        
        try:
            realp_up = 1.0*len(Gp)/GPN
            realp_down = 1.0*len(Gn)/GPN
        except:
            realp_up=0.0
            realp_down=0.0
         
         
        mv = CVH
         
        cidx = len(df)
     
        mvl = max(0,cidx-mv)
        r = self.dataManager.getDeltas(forday,stick,mv)
        p_col = pcol[mvl:cidx]
        v_col = vcol[mvl:cidx]
        r_col = r
        src=0
        nonskipped=0
        sr=0.0
        g=0.0
        if len(p_col)>len(r_col):
            d=len(p_col)-len(r_col)
            p_col=p_col[d:]
        #print('R: ' + str(len(r_col)) + ' P: ' + str(len(p_col)))
        pc=pcol[cidx-CVH:cidx]
        for j in range(0,CVH):
             
            if np.fabs(pc[j])<=1 and pc[j]!=0:
                nonskipped = nonskipped+1
                if p_col[j]*r_col[j]>0.0:   #v_col[j]>0:
                    src=src+1
                 
        try:
            sr = round(float(src/nonskipped),3)
        except:
            return ((0,0,0),0,0,0,1)
        FD=6
        DG1=1 #5
        DG2=2  #10=2*DG1
        DG3=CVH #int(3*DG1)    #15=3*DG1
         
        r1=r[-DG1:]
        r2=r[-DG2:]
        r3=r[-DG3:]
 
        p1 = p_col[-DG1:]
        p2 = p_col[-DG2:]
        p3 = p_col[-DG3:]
         
        W1=[1 for i in range(0,DG1)]
        W2=[1 for i in range(0,DG2)]
        W3=[1 for i in range(0,DG3)]

         
        g1=sum([r1[i]*p1[i] for i in range(0,DG1)])
        g2=sum([r2[i]*p2[i] for i in range(0,DG2)])
        g3=sum([r3[i]*p3[i] for i in range(0,DG3)])
         
        g=(g1,g2,g3)
         
        tp_up=0
        tp_down=0
        p_up=0
        p_down=0
         
        for i in range(0,len(p_col)):
            if np.abs(p_col[i])>1 or p_col[i]==0:
                continue
             
            if p_col[i]>0:
                p_up=p_up+1
                if v_col[i]>0:
                    tp_up=tp_up+1
            if p_col[i]<0:
                p_down=p_down+1
                if v_col[i]>0:
                    tp_down=tp_down+1
                 
        fract_tp1 = 0.0
        fract_tp0 = 0.0
        if p_up>0:
            fract_tp1 = float(tp_up/p_up)
        if p_down>0:
            fract_tp0 = float(tp_down/p_down)
         

        RP=0.0
        TP=0.0
        if realp_up>=0.55:
            RP=realp_up
            TP=fract_tp1
                 
        if realp_down>=0.55:
            RP=realp_down
            TP=fract_tp0
                 
        return (g,sr,TP,RP,1)




