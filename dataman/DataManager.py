
import os
import datetime
import time
from Configuration import Configuration
import numpy
from Service.Utilities import Tools
from pandas.tseries.offsets import BDay
import pandas
from dateutil import parser
import traceback
import glob
import json
import threading
import shutil
import random

from datetime import datetime,timedelta,date
import talib
#import ezibpy

# from dydx3 import Client
# from dydx3.constants import API_HOST_GOERLI
# from dydx3.constants import MARKET_BTC_USD
# from dydx3.constants import NETWORK_ID_GOERLI
# from dydx3.constants import ORDER_SIDE_BUY,ORDER_SIDE_SELL
# from dydx3.constants import ORDER_STATUS_OPEN
# from dydx3.constants import ORDER_TYPE_LIMIT,ORDER_TYPE_MARKET
# from dydx3.constants import TIME_IN_FORCE_GTT,TIME_IN_FORCE_FOK,TIME_IN_FORCE_IOC
# from dydx import invest,neutralizePosition
# from web3 import Web3

# from ib.opt import Connection, message
# from ib.ext.Contract import Contract
# from ib.ext.Order import Order
# from ib.ext.CommissionReport import CommissionReport
# from ib.ext.TickType import TickType as tt




WALLET_ADDRESS = '0x27A66f3889890c24ad6bA59fbE6A955239784c29'
WEB_PROVIDER_URL = 'https://goerli.gateway.tenderly.co'
INVESTMENT=10
CURRENCIES=['USD','EUR','GBP','JPY','CAD','AUD']
MINVAL=10
MIN_DOSAGE = {
    '1INCH-USD' : 1.0,
    'BTC-USD': 0.001,
    'ETH-USD': 0.01,
    'LINK-USD': 1.0,
    'AAVE-USD': 0.1,
    'UNI-USD': 1.0,
    'SUSHI-USD': 1.,
    'SOL-USD': 1.0,
    'YFI-USD': 0.001,
    'ONEINCH-USD': 1.0,
    'AVAX-USD': 1.0,
    'SNX-USD': 1.0,
    'CRV-USD': 10.0,
    'UMA-USD': 1.0,
    'DOT-USD': 1.0,
    'DOGE-USD': 100.0,
    'MATIC-USD': 10.0,
    'MKR-USD': 0.01,
    'FIL-USD': 1.0,
    'ADA-USD': 10.0,
    'ATOM-USD': 1.0,
    'COMP-USD': 0.1,
    'BCH-USD': 0.1,
    'LTC-USD': 0.1,
    'EOS-USD': 10,
    'ALGO-USD': 10,
    'ZRX-USD': 10.0,
    'XMR-USD': 0.1,
    'ZEC-USD': 0.1,
    'ENJ-USD': 10.0,
    'ETC_USD': 1.0,
    'XLM-USD': 100.0,
    'TRX-USD': 100.0,
    'XTZ-USD': 10.0,
    'ICP-USD': 1.0,
    'RUNE-USD': 10.0,
    'LUNA-USD': 0.1,
    'NEAR-USD': 1.0,
    'CELO-USD': 10.0,
}


class DataManager():
    
    SUBDIR=''
    TREND=1
    ORDERS=False
    
    #TRENDSUFFIX={1:'-D',-1:'-U'}
    
    wdays={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
    
    @staticmethod
    def copyState():
        curdname = os.getcwd().split(os.sep)[-1]
        simdir = os.path.join(os.getcwd(),'..',curdname+'-simul')
        
        state_timestamp = None
        fresh_timestamp = None
        refreshed=False
        
        try:
            if os.path.exists(os.path.join(os.getcwd(),'ready.mak')):
                with open(os.path.join(os.getcwd(),'ready.mak')) as f:
                    t=f.read().replace("\n", "")
                    state_timestamp = parser.parse(t,dayfirst=True)
            
            if os.path.exists(os.path.join(simdir,'ready.mak')):
                with open(os.path.join(simdir,'ready.mak')) as f1:
                    t1=f1.read().replace("\n", "")
                    fresh_timestamp = parser.parse(t1,dayfirst=True)
            
            if fresh_timestamp is not None and state_timestamp is None:
                refreshed=True
            else:
                refreshed = fresh_timestamp>state_timestamp
        except:
            pass
        
        if refreshed==False:
            return
        
        if os.path.exists('sandprecent')==False:
            os.mkdir('sandprecent')
            
        for st in os.listdir(os.path.join(simdir,'sandprecent')):
            try:
                os.remove(os.path.join(os.getcwd(),'sandprecent',st))
            except:
                pass
            shutil.copy(os.path.join(simdir,'sandprecent',st),os.path.join(os.getcwd(),'sandprecent'))
        
        for csvf in glob.glob(os.path.join(simdir,'*.csv')):
            if 'final-report' in csvf:
                continue
#             try:
#                 os.remove(os.path.join(os.getcwd(),csvf))
#             except:
#                 pass
            shutil.copy(csvf,os.getcwd())
        
        for csvf in glob.glob(os.path.join(simdir,'*.pt')):
    #         if 'last-positions' i'final-report' in csvf:
    #             continue
#             try:
#                 os.remove(os.path.join(os.getcwd(),csvf))
#             except:
#                 pass
            shutil.copy(csvf,os.getcwd())
        
        for csvf in glob.glob(os.path.join(simdir,'*.txt')):
    #         if 'last-positions' in csvf or 'final-report' in csvf:
    #             continue
#             try:
#                 os.remove(os.path.join(os.getcwd(),csvf))
#             except:
#                 pass
            shutil.copy(csvf,os.getcwd())
            
        try:
            os.remove(os.path.join(os.getcwd(),'ready.mak'))
        except:
            pass
        shutil.copy(os.path.join(simdir,'ready.mak'),os.getcwd())
        
        
     
    def __init__(self, dsource):
        
        #self.clearPosition()
        #self.placeCloseMarker()
#         if '_' in dsource:
#             DataManager.copyState()
        
        self.longstickers={}
        self.shortstickers={}
        
        self.maillogs=False
        
        self.stickers=None
        self.rawData={}
        
        self.datasource=dsource
        self.globalDatasource=None
        self.ezConn=None
        
        self.dframes={}
        self.dfindices={}
        self.shares={}

        self.extension=''
        self.alldates=None
        self.datesList=None
        self.tsindex = None
        self.dateindex={}
        self.verifiedOrders={}
        
#         self.longstickers={}
#         self.shortstickers={}
        
        dsname = self.datasource.split('_')[0]
#         if dsname!='sandprecent':
#             self.dsName=dsname
#         else:
#             self.dsName=self.datasource
            
        self.collectData()
        self.prepDFrames()
        self.dydxclient=None
        self.dydxposition_id=None
        self.dydxposition=None
        
        
        if '_temp' in self.datasource:
            dsdir = self.datasource.split('_')[0]
            if self.globalDatasource is None:
                self.globalDatasource = DataManager(dsdir) 
        
    def vaultProcess(self,longs,shorts,timestamp):
        MDL=10
        vault={}
        if os.path.exists('vault.json'):
            with open('vault.json', 'rt') as file:
                vault = json.load(file)
        else:
            for c1 in CURRENCIES:
                vault[c1]=100000
            
        bl_l=set([])
        bl_s=set([])
        for l in shorts:
            x=l[0:3]
            y=l[3:]
            pr=self.getAt(l, 'close', timestamp)
            if x in vault.keys() and vault[x]>MINVAL:
                vault[x]=vault[x]-1
                if y in vault.keys():
                    vault[y]=vault[y]+pr*MDL
                else:
                    vault[y]=pr*MDL
            else:
                bl_s.add(l)
                    
        for l in longs:
            x=l[0:3]
            y=l[3:]
            pr=self.getAt(l, 'close', timestamp)
            if y in vault.keys() and vault[y]>MINVAL:
                vault[y]=vault[y]-pr
                if x in vault.keys():
                    vault[x]=vault[x]+1*MDL
                else:
                    vault[x]=1
            else:
                bl_l.add(l)
                    
        longs = longs.difference(bl_l)
        shorts = shorts.difference(bl_s)
            
        with open('vault.json', 'wt') as f:
            json.dump(vault, f)
            
        return (longs,shorts)
                    
            
  

    
    
    def prevTS(self,ts,DST=1):
        try:
            tsi = self.datesList.index(ts)
        except:
            return self.datesList[0]
        
        if tsi==0:
            return self.datesList[0]
        
        return self.datesList[max(0,tsi-DST)]
        
    
    def prepDFrame(self,sticker):

        df = self.rawData[sticker] #100*self.rawData[sticker].pct_change(fill_method='ffill')
        ld = pandas.Timestamp(self.datesList[-1])
        #df.iloc[0] = [0]*len(Configuration.characteristics)
        df.loc[ld] = [0]*len(Configuration.characteristics)
        
        df.fillna(0,inplace=True)
        
        return df
    
    def prepDFrames(self):
            
        for sticker in self.stickers:
            df = self.prepDFrame(sticker)
            if df is not None:
                self.dframes[sticker] = df
                self.dfindices[sticker] = df.index.astype(int)//10**9
        
    def collectStickers(self):
        
        sts=[]
        dr = os.path.join(os.getcwd(),self.datasource)
        #print(dr)
        for file in os.listdir(dr):
            if '.' not in file:
                continue
            try:
                
                stickername = os.path.splitext(file)[0]
                #print(stickername)
#             if len(self.extension)==0:
#                 self.extension=os.path.splitext(file)[1]
            
                sts.append(stickername)
            except:
                pass
            #self.stickers.append(stickername)
        self.stickers = numpy.array(sts)
        
    
    @staticmethod
    def dateparse(d):    
        dp=d.replace('"',"")
        return parser.parse(dp)
                        
    def collectData(self):
        self.collectStickers()
        ns=['date','open','high','low','close','volume']
         
         
        for sticker in self.stickers:
            dfr=None
            tname = os.path.join(os.getcwd(), self.datasource, sticker + '.csv')
            if os.path.exists(tname)==False:
                continue
                            
            try:
                dfr = pandas.read_csv(tname,sep='\s+', parse_dates=True,date_parser=DataManager.dateparse,header=None,names=ns,index_col=0)    
                    
            except:
                traceback.print_exc()
                
                 
            self.rawData[sticker]=dfr
            iv = list(dfr.index.unique())
            
            
            iv1 = iv
                
            if self.alldates is None:
                self.alldates = iv1
            else:
                for ts in iv1:
                    if ts not in self.alldates:
                        self.alldates.append(ts)

                    
        #self.stickers = numpy.array(self.rawData.keys())
        self.alldates = sorted(self.alldates)
        
        if Configuration.HOURLY:
            nextbd = Tools.nextBusHour(self.alldates[-1])
        else:   
            nextbd = Tools.nextBusDay(self.alldates[-1])
        self.alldates.append(pandas.Timestamp(nextbd))
            
        self.datesList = [td.strftime(Configuration.timestamp_format) for td in self.alldates]    
        
        for i in range(0,len(self.datesList)):
            d = self.datesList[i]
            self.dateindex[d] = i
        
        self.tsindex = pandas.DatetimeIndex(self.alldates)  
    
    
    
    
    
    def get(self,sticker,column,timestamp,depth):
        
        if column not in Configuration.characteristics+['date']:
            return []
        
        dm=self

        
        df = dm.dframes[sticker]
        arr = dm.dfindices[sticker]
        tsn = pandas.Timestamp(timestamp)
        l=int(tsn.timestamp())
        
#         try:
#             idx = arr.index(tsn) #numpy.max(numpy.searchsorted(arr, l, side='left')-1, 0)+1
#             idx_start = max(0,idx-depth)
#             
#             return df[column].iloc[idx_start:idx]
#         except:
#             return []
        
        try:
            idx = numpy.max(numpy.searchsorted(arr, l, side='left')-1, 0)+1
            #idx = df.index.get_loc(tsn)
            idx_start = max(0,idx-depth)
             
            return df[column].iloc[idx_start:idx]
        except:
            return []
        

    def getAt(self,sticker,column,timestamp):
        
        dm=self
    
        ts = pandas.Timestamp(timestamp)
        if sticker not in dm.dframes:
            return 0.0
        df = dm.dframes[sticker]
        arr = dm.dfindices[sticker]
        l=int(ts.timestamp())
        res=None
        
#         try:
#             idx = arr.index(tsn) #numpy.searchsorted(arr, l, side='left')
#             row = df.iloc[idx,:]
#             res = row[column]
#             if res is not None:
#                 return res  
#         except:
#             return 0.0
        
        try:
            idx = numpy.searchsorted(arr, l, side='left')
            row = df.iloc[idx,:]
            res = row[column]
            if res is not None:
                return res  
        except:
            return 0.0
    
    def getDelta(self,sticker,column,timestamp):
        tsi=0
        
        try:
            tsi = self.dateindex[timestamp] #self.datesList.index(startts)
            tsi1 = max(0,tsi-1)
            if tsi1==tsi:
                return 0.0  
        except:
            return 0.0
        

        timestamp_1 = self.datesList[tsi1]
        try:
        
            td = self.getAt(sticker, column, timestamp)
            ys = self.getAt(sticker, column, timestamp_1)
            if td<0 or ys<0 or numpy.fabs(td)<0.00001 or numpy.fabs(ys)<0.00001:
                return 0.0
            return 100.0*float((td-ys)/ys)
        except:
            return 0.0
        
        
    
    def getDeltas(self,startts,sticker,D):
        
        deltas=[]
        starttsi=0
        
        try:
            starttsi = self.dateindex[startts] #self.datesList.index(startts)
        except:
            return deltas
        
        starttsi1 = max(0,starttsi-D)
        if starttsi1==starttsi:
            return []
        
        tsline = self.datesList[starttsi1:starttsi]
        
        for tds in tsline:
            d=self.getDelta(sticker, 'close', tds)
            deltas.append(d)
        return deltas
    
        #return numpy.log(td/ys)
             
    
    def updatePositionFileGen(self):
        
        actlogs = glob.glob("*_jointT_maillog.csv")
        outf = 'final-report.csv'
        colnames=['Date','SelectedTickers','SelectedDeltas',
                  'AvgDelta','AvgDeltaL','AvgDeltaS',
                  'Chosen','PosChosen','LongChosen',
                  'SuccessRows','SuccessTickers',
                  'RawGain','AccumulatedGain','AccumulatedGainL','AccumulatedGainS']
        poslog1 = os.path.join(os.getcwd(),outf)
        if os.path.exists(poslog1)==True:
            os.remove(poslog1)
        
        ofl = open(poslog1,'at')
        ofl.write(','.join(colnames)+'\n')
        ofl.close()
        timestamps=set([])
        for actlog in actlogs:
            if self.datasource in actlog and '_jointT_maillog' in actlog:
                print('Adding ' + actlog)
                try:
                    df = pandas.read_csv(actlog,index_col=0,header=None)
                    if len(df)<1:
                        continue
                    df.fillna('',inplace=True)
                    df = df[~df.index.duplicated(keep='first')]
        
                    timestamps = timestamps.union(set(list(df.index)))
                except:
                    pass
        
        delta_l=[]
        delta_s=[]
        gain=[]
        gain_l=[]
        gain_s=[]
        dailygain=[]
        dailygain_l=[]
        dailygain_s=[]
        rawgain=[]
        
        dailyrate=0.0
        stickrate=0.0
        
        accumGain=0.0
        accumGain_l=0.0
        accumGain_s=0.0
        posdays=0
        nonskippeddays=0
        postickers=0
        alltickers=0
        
        timestampsL = sorted(list(timestamps))
        
        for rwn in range(0,len(timestampsL)):
            actual_timestamp = timestampsL[rwn]
            templongs=set([])
            tempshorts = set([])
            
            lprofitS=''
            sprofitS=''
            postoday=0
            avgl=0
            avgs=0
            avg=0
            
            for inpf in actlogs:
                poslog=os.path.join(os.getcwd(),inpf)
                posdf = pandas.read_csv(poslog,index_col=0,header=None)
                try:
                    (ll,sl)=self.genJoinEntry(df, rwn)
                except:
                    print('Missing timestamp when merging at ' + actual_timestamp)
                    ll=None
                    sl=None
                    traceback.print_exc()
                if ll is None or sl is None:
                    continue
                if len(templongs)==0:
                    templongs=set(ll)
                else:
                    templongs=templongs.union(set(ll))

                if len(tempshorts)==0:
                    tempshorts=set(sl)
                else:
                    tempshorts=tempshorts.union(set(ll)) 
        
            
            for l in templongs:
                delta = self.getDelta(l,'close',actual_timestamp)
                lprofitS = lprofitS+l+':'+str(round(delta,2))+'_'
                if delta>0:
                    postoday=postoday+1
                avgl=avgl+delta
                
                   
            for s in tempshorts:    
                delta =-self.getDelta(s,'close',actual_timestamp)
                sprofitS = sprofitS+s+':'+str(round(delta,2))+'_'
                if delta>0:
                    postoday=postoday+1
                avgs=avgs+delta
                
            
            if len(templongs)+len(tempshorts)>0:
                avg=(avgl+avgs)/(len(tempshorts)+len(templongs))
            if len(templongs)>0:
                avgl=avgl/len(templongs)
            if len(tempshorts)>0:
                avgs=avgs/len(tempshorts)
            
            
            
            deltas=lprofitS+"_"+sprofitS
            ctl = templongs.union(tempshorts)
            
            ct = 'L:'+'_'.join(templongs)+'_S:'+'_'.join(tempshorts)
            
            nrow=[actual_timestamp,ct,deltas]
            nrow = nrow + [str(round(avg,2)),str(round(avgl,2)),str(round(avgs,2))]
            nrow = nrow + [str(len(templongs)+len(tempshorts)),str(postoday),str(len(templongs))]
            
            try:
                dailyrate=1.0*posdays/nonskippeddays
            except:
                dailyrate=0.0
            
            try:
                stickrate=1.0*postickers/alltickers
            except:
                stickrate=0.0
                
            print(actual_timestamp + ' '  + ct + ' ' + str(round(dailyrate,2))+ ' : ' + str(round(stickrate,2)))
            
            nrow=nrow+[str(round(dailyrate,2)),str(round(stickrate,2)),str(round(sum(rawgain),2))]
            nrow=nrow+[str(round(accumGain,2)),str(round(accumGain_l,2)),str(round(accumGain_s,2))]
            
            ofl = open(poslog1,'at')
            ofl.write(','.join(nrow)+'\n')
            ofl.close()
            
                      
            rawdeltas = [self.getDelta(sticker,'close',actual_timestamp) for sticker in self.stickers]
            rgnan=[]
            for v in rawdeltas:
                try:
                    if numpy.fabs(v)>50:
                        continue
                    if numpy.isnan(v)==False and numpy.isinf(v)==False:
                        rgnan.append(v)
                except:
                    pass
            rgd = numpy.mean(rgnan)
            rawgain.append(rgd)
            accumGain=accumGain+avg
            accumGain_l=accumGain_l+avgl
            accumGain_s=accumGain_s+avgs
            if avg>0:
                posdays=posdays+1
            
            alltickers=alltickers+len(templongs)+len(tempshorts)
            if len(templongs)+len(tempshorts)>0:
                nonskippeddays=nonskippeddays+1
                
            postickers=postickers+postoday
            
        print('Report built')
            
            
            
    def genJoinEntry(self,posdf,rn):   
        prev_timestamp=''
        actual_timestamp = posdf.index.values[rn]
        verifiedDeltas={}
        chosen_tickers=posdf.iloc[rn,0]
        verifiedDeltas[actual_timestamp]={}
            
        il = chosen_tickers.find("L:")
        iss=chosen_tickers.find("S:")
                
        ll=chosen_tickers[2:iss].split(" ")
        sl=chosen_tickers[iss+2:].split(" ")
        ll=list(filter(lambda e:len(e)>0,ll))
        sl=list(filter(lambda e:len(e)>0,sl))
                
        suspicious=set([])
        if rn>=1:
            prev_timestamp = posdf.index.values[rn-1]
            chosen_tickers_1=posdf.iloc[rn-1,0]
            verifiedDeltas[prev_timestamp]={}
            
            #il = chosen_tickers_1.find("L:")
            iss=chosen_tickers_1.find("S:")
                
            ll1=chosen_tickers_1[2:iss].split(" ")
            sl1=chosen_tickers_1[iss+2:].split(" ")
            ll1=list(filter(lambda e:len(e)>0,ll1))
            sl1=list(filter(lambda e:len(e)>0,sl1))
                
            for l1 in ll1:
                delta = self.getDelta(l1,'close',prev_timestamp)
                verifiedDeltas[prev_timestamp][l1]=delta
                 
            for s1 in sl1:    
                delta =-self.getDelta(s1,'close',prev_timestamp)
                verifiedDeltas[prev_timestamp][s1]=delta

            prevkeys = verifiedDeltas[prev_timestamp].keys()
            prevAvg=sum([verifiedDeltas[prev_timestamp][k] for k in prevkeys])
            if len(prevkeys)>0:
                prevAvg=prevAvg/len(prevkeys)
                
            for l in ll:
                if l in prevkeys and verifiedDeltas[prev_timestamp][l]<0:
                    suspicious.add(l)
            
            verifiedNegatives=0.0
            for k in prevkeys:  
                if verifiedDeltas[prev_timestamp][k]<0:
                    verifiedNegatives=verifiedNegatives+1
                    
            if len(prevkeys)>0:
                verifiedNegatives=verifiedNegatives/len(prevkeys)
            
            suspFract=0.0
            if len(ll)>0:
                suspFract=1.0*len(suspicious)/len(ll)

            
            nowAvg=sum([self.getDelta(k,'close',actual_timestamp) for k in ll])
            
            sfn = open('statneg.csv','at')
            idd='0'
            if prevAvg<0 and nowAvg<0:
                idd='1'
                
            sfn.write(actual_timestamp+','+str(round(prevAvg,2))+','+str(round(nowAvg,2))+','+str(round(suspFract,2))+','+str(round(verifiedNegatives,2))+','+str(len(prevkeys)) + ','+str(len(ll))+','+idd+'\n')
            sfn.close()
                
            if prevAvg<0 and (suspFract>0 or prevAvg<-1.5):
                ll=[]
                sl=[]
        return (ll,sl)
        
            
    
    
                
                
    def updatePositionFile(self,RED=1):
 
        dsname = self.datasource.split('_')[0]
        inpf = 'sandprecent_jointT_maillog.csv'
        
        outf = 'final-report.csv'
        colnames=['Date','SelectedTickers','SelectedDeltas',
                  'AvgDelta','AvgDeltaL','AvgDeltaS',
                  'Chosen','PosChosen','LongChosen',
                  'SuccessRows','SuccessTickers',
                  'RawGain','AccumulatedGain','AccumulatedGainL','AccumulatedGainS']

        
        poslog1 = os.path.join(os.getcwd(),outf)
        if os.path.exists(poslog1)==True:
            os.remove(poslog1)
        
        ofl = open(poslog1,'at')
        ofl.write(','.join(colnames)+'\n')
        ofl.close()
         
 
        poslog=os.path.join(os.getcwd(),inpf)
        posdf = pandas.read_csv(poslog,index_col=0,header=None)
        chosen_tickers = posdf.iloc[:,0].values
        
         
        delta_l=[]
        delta_s=[]
        gain=[]
        gain_l=[]
        gain_s=[]
        dailygain=[]
        dailygain_l=[]
        dailygain_s=[]
        rawgain=[]
        
        dailyrate=0.0
        stickrate=0.0
        
        accumGain=0.0
        accumGain_l=0.0
        accumGain_s=0.0
        posdays=0
        nonskippeddays=0
        postickers=0
        alltickers=0
        verifiedDeltas={}
        
        for rn in range(0,len(posdf)):
            readyst=posdf.iloc[0:rn,0].values
            readytimestamps=posdf.iloc[0:rn,0].index.values
            
            prev_timestamp=''
            actual_timestamp = posdf.index.values[rn]
            
            chosen_tickers=posdf.iloc[rn,0]
            verifiedDeltas[actual_timestamp]={}
            
            il = chosen_tickers.find("L:")
            iss=chosen_tickers.find("S:")
                
            ll=chosen_tickers[2:iss].split(" ")
            sl=chosen_tickers[iss+2:].split(" ")
            ll=list(filter(lambda e:len(e)>0,ll))
            sl=list(filter(lambda e:len(e)>0,sl))
            
            
            
            
            
            lprofitS=''
            sprofitS=''
            postoday=0
            avgl=0
            avgs=0
            avg=0
                
            maybeskip=False
            suspicious=set([])
            if rn>1:
                prev_timestamp = posdf.index.values[rn-1]
                chosen_tickers_1=posdf.iloc[rn-1,0]
                verifiedDeltas[prev_timestamp]={}
            
                #il = chosen_tickers_1.find("L:")
                iss=chosen_tickers_1.find("S:")
                
                ll1=chosen_tickers_1[2:iss].split(" ")
                sl1=chosen_tickers_1[iss+2:].split(" ")
                ll1=list(filter(lambda e:len(e)>0,ll1))
                sl1=list(filter(lambda e:len(e)>0,sl1))
                
                for l1 in ll1:
                    delta = self.getDelta(l1,'close',prev_timestamp)
                    verifiedDeltas[prev_timestamp][l1]=delta
                 
                for s1 in sl1:    
                    delta =-self.getDelta(s1,'close',prev_timestamp)
                    verifiedDeltas[prev_timestamp][s1]=delta

                prevkeys = verifiedDeltas[prev_timestamp].keys()
                prevAvg=sum([verifiedDeltas[prev_timestamp][k] for k in prevkeys])
                if len(prevkeys)>0:
                    prevAvg=prevAvg/len(prevkeys)
                
                verifiedNegatives=0.0
                for l in ll:
                    if verifiedDeltas[prev_timestamp][l]<0:
                        verifiedNegatives=verifiedNegatives+1
                    if l in prevkeys:
                        suspicious.add(l)
                        
                if len(ll)>0:
                    verifiedNegatives=1.0*verifiedNegatives/len(ll)
                    
                
                
                nowAvg=sum([self.getDelta(k,'close',actual_timestamp) for k in ll])
                with open('statneg.csv','at') as sfn:
                    
                    idd='0'
                    susp=''
                    if prevAvg<0: # and verifiedNegatives>0.4:  #len(suspicious)>0:
                        idd='1'
                        susp='_'.join(suspicious)
                        
                        ll=[]
                        sl=[]
                        print('----------------------  miss at ' + actual_timestamp)
                    sfn.write(actual_timestamp+','+str(round(prevAvg,2))+','+str(round(nowAvg,2))+','+idd+','+susp+'\n')
                    sfn.close()
                    
                    
                
            for l in ll:
                delta = self.getDelta(l,'close',actual_timestamp)
                lprofitS = lprofitS+l+':'+str(round(delta,2))+'_'
                if delta>0:
                    postoday=postoday+1
                avgl=avgl+delta
                
                   
            for s in sl:    
                delta =-self.getDelta(s,'close',actual_timestamp)
                sprofitS = sprofitS+s+':'+str(round(delta,2))+'_'
                if delta>0:
                    postoday=postoday+1
                avgs=avgs+delta
                
            
            if len(ll)+len(sl)>0:
                avg=(avgl+avgs)/(len(ll)+len(sl))
            if len(ll)>0:
                avgl=avgl/len(ll)
            if len(sl)>0:
                avgs=avgs/len(sl)
            
            
            
            deltas=lprofitS+"_"+sprofitS
        
            nrow=[actual_timestamp,chosen_tickers,deltas]
            nrow = nrow + [str(round(avg,2)),str(round(avgl,2)),str(round(avgs,2))]
            nrow = nrow + [str(len(ll)+len(sl)),str(postoday),str(len(ll))]
            
            try:
                dailyrate=1.0*posdays/nonskippeddays
            except:
                dailyrate=0.0
            
            try:
                stickrate=1.0*postickers/alltickers
            except:
                stickrate=0.0
                
            print(actual_timestamp + ' '  + chosen_tickers + ' ' + str(round(dailyrate,2))+ ' : ' + str(round(stickrate,2)))
            
            nrow=nrow+[str(round(dailyrate,2)),str(round(stickrate,2)),str(round(sum(rawgain),2))]
            nrow=nrow+[str(round(accumGain,2)),str(round(accumGain_l,2)),str(round(accumGain_s,2))]
            
            ofl = open(poslog1,'at')
            ofl.write(','.join(nrow)+'\n')
            ofl.close()
            
                      
            rawdeltas = [self.getDelta(sticker,'close',actual_timestamp) for sticker in self.stickers]
            rgnan=[]
            for v in rawdeltas:
                try:
                    if numpy.fabs(v)>50:
                        continue
                    if numpy.isnan(v)==False and numpy.isinf(v)==False:
                        rgnan.append(v)
                except:
                    pass
            rgd = numpy.mean(rgnan)
            rawgain.append(rgd)
            accumGain=accumGain+avg
            accumGain_l=accumGain_l+avgl
            accumGain_s=accumGain_s+avgs
            if avg>0:
                posdays=posdays+1
            
            alltickers=alltickers+len(ll)+len(sl)
            if len(ll)+len(sl)>0:
                nonskippeddays=nonskippeddays+1
                
            postickers=postickers+postoday
            
            
            
        print('Report built')
    

    
        
    def prepOnline(self):    
        dsname = self.datasource.split('_')[0]
        if self.globalDatasource is None:
            dsdir=os.path.join(os.getcwd(), DataManager.SUBDIR,dsname)
            self.globalDatasource = DataManager(dsdir)
                
        actlogs = glob.glob(os.path.join(os.getcwd(), DataManager.SUBDIR,"correlation*.csv"))
        sticks_cont={}
        for actlog in actlogs:
            sticks_cont[actlog] = set([])
        
        for actlog in actlogs:
            df = pandas.read_csv(actlog,index_col=0)
            for st in self.globalDatasource.stickers:
            
                for ts in df.index.values:
                    try:
                        v = df.at[ts,'Close-'+st] 
                    except:
                        v=None
                    if v is None:
                        continue
                    
                    sticks_cont[actlog].add(st)

        
        for k in sticks_cont:
            prf=k.find("-D")
            [n,ex]=k[prf+3:].split('.')
            print(n,end=':')
            l=sticks_cont[k]
            lk=sorted(l)
            print('_'.join(lk))
            partd = os.path.join(os.getcwd(),DataManager.SUBDIR,n)
            try:
                os.makedirs(partd)
                for filename in lk:
                    srcf=os.path.join(DataManager.SUBDIR,dsname,filename+'.'+ex)
                    dstf=os.path.join(partd,filename+'.'+ex)
                    shutil.copy(srcf,dstf)
            except:
                pass

        
                

    
    def joinMaillogsIntegr(self,pred):
           
        actlogs = glob.glob("*maillog.csv")
        actlogJ=self.datasource+'_jointT_maillog.csv'
        actlogA=self.datasource+'_'+pred+'_joint_maillog.csv'
           
        if os.path.exists(actlogJ)==False or os.path.exists(actlogA)==False:
            return
        
        actdfs=[]
        timestamps=set([])
        actlogs=[actlogJ,actlogA]
               
        for actlog in actlogs:
            print('Adding ' + actlog)
            try:
                df = pandas.read_csv(actlog,index_col=0,header=None)
                if len(df)<1:
                    continue
                df.fillna('',inplace=True)
                df = df[~df.index.duplicated(keep='first')]
       
                actdfs.append(df)
                timestamps = timestamps.union(set(list(df.index)))
            except:
                pass
               
           
        timestamps=sorted(timestamps)
       
        stickers={}
        detls={}
           
        for ts in timestamps:
            sts=[]
            detll=[]
                   
            for df in actdfs:
                av=df.index.values
                ids=numpy.where(av==ts) 
                if len(ids)>0:
                    try:
                        sts.append(df.iloc[ids[0][0],0]) 
                        detl = df.iloc[ids[0][0],1]   
                        detll.append(detl)
                    except:
                        pass
                else:
                    sts.append('')
                       
            stickers[ts]=sts
            detls[ts] = detll
            
            
        os.remove(actlogJ)
        for ts in timestamps:
            chosenlists=[st.replace(':',' ').split(' ') for st in stickers[ts]]
            templongs=set([])
            tempshorts=set([])
               
            for chosenlist in chosenlists:
                if 'L' in chosenlist or 'S' in chosenlists:
                    li = chosenlist.index('L')
                    si = chosenlist.index('S')
           
                    longs =  chosenlist[li+1:si] 
                    shorts = chosenlist[si+1:]
                    
                    if len(templongs)==0:
                        templongs=set(longs)
                    else:
                        templongs=templongs.union(set(longs))
                        
                    if len(tempshorts)==0:
                        tempshorts=set(shorts)
                    else:
                        tempshorts=tempshorts.union(set(shorts))
                   
       
            longss=set(filter(lambda li:len(li)>0,templongs))
            shortss=set(filter(lambda li:len(li)>0,tempshorts))
            d=list(filter(lambda li:len(li)>0,detls[ts]))
             
#             if len(longss)>=8: >8: 0.65-tsf-ada, >=8: 0.6-tsf-ada - 
#                 print(ts + ' '+ str(len(shortss)) + ' SKIP ')
#                 longss=set([])
#                 d=[]

#             if len(longss)>=7: #0.6 - brf-ada
#                 print(ts + ' '+ str(len(shortss)) + ' SKIP ')
#                 longss=set([])
#                 d=[]

#             if len(longss)>=3: #0.6 - brf-tsf
#                 print(ts + ' '+ str(len(shortss)) + ' SKIP ')
#                 longss=set([])
#                 d=[]

#             if len(longss)>=5: #rus
#                 print(ts + ' '+ str(len(shortss)) + ' SKIP ')
#                 longss=set([])
#                 d=[]
             
                    
            lastMessage = 'L:{0} S:{1}'.format(' '.join(longss), ' '.join(shortss))
            lastExtMessage = '#'.join(d)
               
            with open(actlogJ, 'a') as fa:
                addt='{0},{1},{2}\n'.format(ts,lastMessage,lastExtMessage)
                fa.write(addt)
                fa.close()
                
    def joinMaillogs(self,predl):
            
        actlogs = glob.glob("*maillog.csv")
        actlogJ=self.datasource+predl+'_jointT_maillog.csv'
        actlogI=self.datasource+'_investments_maillog.csv'
        SUMINVEST=50000
            
        if os.path.exists(actlogJ):
            os.remove(actlogJ)
         
        
        actdfs=[]
        timestamps=set([])
                
        for actlog in actlogs:
            if self.datasource in actlog and 'joint_maillog' in actlog and predl+'_' in actlog:
                print('Adding ' + actlog + ' with ' + predl)
                try:
                    df = pandas.read_csv(actlog,index_col=0,header=None)
                    if len(df)<1:
                        continue
                    df.fillna('',inplace=True)
                    df = df[~df.index.duplicated(keep='first')]
        
                    actdfs.append(df)
                    timestamps = timestamps.union(set(list(df.index)))
                except:
                    pass
                
            
        timestamps=sorted(timestamps)
        
        stickers={}
        detls={}
            
        for ts in timestamps:
            sts=[]
            detll=[]
                    
            for df in actdfs:
                av=df.index.values
                ids=numpy.where(av==ts) 
                if len(ids)>0:
                    try:
                        sts.append(df.iloc[ids[0][0],0]) 
                        detl = df.iloc[ids[0][0],1]   
                        detll.append(detl)
                    except:
                        pass
                else:
                    sts.append('')
                        
            stickers[ts]=sts
            detls[ts] = detll
                
        for ts in timestamps:
            chosenlists=[st.replace(':',' ').split(' ') for st in stickers[ts]]
            templongs=set([])
            tempshorts=set([])
                
            for chosenlist in chosenlists:
                if 'L' in chosenlist or 'S' in chosenlists:
                    li = chosenlist.index('L')
                    si = chosenlist.index('S')
            
                    longs =  chosenlist[li+1:si] 
                    shorts = chosenlist[si+1:]
                     
                    if len(templongs)==0:
                        templongs=set(longs)
                    else:
                        templongs=templongs.union(set(longs))

                    if len(tempshorts)==0:
                        tempshorts=set(shorts)
                    else:
                        tempshorts=tempshorts.union(set(shorts))         
        
            longss=set(filter(lambda li:len(li)>0,templongs))
            shortss=set(filter(lambda li:len(li)>0,tempshorts))
            
#             contradictions1 = longss.intersection(shortss)
#             contradictions2 = shortss.intersection(longss)
#               
#             if len(contradictions1)>0 or len(contradictions2)>0:
#                 print('Inconsistency  at ' + ts, end=' ')
#                 print(contradictions1)
#                 print(contradictions2)
#                 
#             # longs >=5 switch
#             longss = longss.difference(contradictions1)
#             longss = longss.difference(contradictions2)
#             shortss = shortss.difference(contradictions1)
#             shortss = shortss.difference(contradictions2)
            
            shortss=set([])
            
#             longss=set([])
#             else:
#                 shortss=longss.copy()
            
            d=list(filter(lambda li:len(li)>0,detls[ts]))
             
            # ALEX: Invert
            lastMessage = 'L:{0} S:{1}'.format(' '.join(longss), ' '.join(shortss))
            lastExtMessage = '#'.join(d)
                
            with open(actlogJ, 'a') as fa:
                addt='{0},{1},{2}\n'.format(ts,lastMessage,lastExtMessage)
                fa.write(addt)
                fa.close()

    def joinMaillogsV(self):
             
        actlogs = glob.glob("*maillog.csv")
        actlogJ=self.datasource+'_jointT_maillog.csv'
        actlogI=self.datasource+'_investments_maillog.csv'
             
        if os.path.exists(actlogJ):
            os.remove(actlogJ)
          
         
        actdfs=[]
        timestamps=set([])
                 
        for actlog in actlogs:
            if self.datasource in actlog and 'joint_maillog' in actlog:
                print('Adding for vote ' + actlog)
                try:
                    df = pandas.read_csv(actlog,index_col=0,header=None)
                    if len(df)<1:
                        continue
                    df.fillna('',inplace=True)
                    df = df[~df.index.duplicated(keep='first')]
         
                    actdfs.append(df)
                    timestamps = timestamps.union(set(list(df.index)))
                except:
                    pass
                 
             
        timestamps=sorted(timestamps)
         
        stickers={}
        detls={}
             
        for ts in timestamps:
            sts=[]
            detll=[]
                     
            for df in actdfs:
                av=df.index.values
                ids=numpy.where(av==ts) 
                if len(ids)>0:
                    try:
                        sts.append(df.iloc[ids[0][0],0]) 
                        detl = df.iloc[ids[0][0],1]   
                        detll.append(detl)
                    except:
                        pass
                else:
                    sts.append('')
                         
            stickers[ts]=sts
            detls[ts] = detll
                 
        for ts in timestamps:
            chosenlists=[st.replace(':',' ').split(' ') for st in stickers[ts]]
            templongs=set([])
            tempshorts=set([])
             
            votesL={}
            votesS={}
             
            for chosenlist in chosenlists:
                 
                if 'L' in chosenlist or 'S' in chosenlists:
                    li = chosenlist.index('L')
                    si = chosenlist.index('S')
             
                    longs =  chosenlist[li+1:si] 
                    shorts = chosenlist[si+1:]
                      
                    for s in longs:
                        if s not in votesL:
                            votesL[s]=1
                    else:
                        votesL[s]=votesL[s]+1
                     
                    for s in shorts:
                        if s not in votesS:
                            votesS[s]=1
                        else:
                            votesS[s]=votesS[s]+1
     
            enLongs=set([])
            enShorts=set([])
                     
            for s in votesL:
                if len(s)>0 and votesL[s]>int(0.5*len(actdfs)):
                    enLongs.add(s)
#                 else:
#                     print(ts + ' ' + 'not enough ' + s + ' ' + str(int(votesL[s])))
                             
            for s in votesS:
                if len(s)>0 and votesS[s]>int(0.5*len(actdfs)):
                    enShorts.add(s)
#                 else:
#                     print(ts + ' ' + 'not enough ' + s + ' ' + str(int(votesS[s])))
                     
            if len(templongs)==0:
                templongs=enLongs
            else:
                templongs=templongs.union(enLongs)
                      
  
            if len(tempshorts)==0:
                tempshorts=enShorts
            else:
                tempshorts=tempshorts.union(enShorts)  
                        
            longss=set(filter(lambda li:len(li)>0,templongs))
            shortss=set(filter(lambda li:len(li)>0,tempshorts))
             
            contradictions1 = longss.intersection(shortss)
            contradictions2 = shortss.intersection(longss)
               
            if len(contradictions1)>0 or len(contradictions2)>0:
                print('Inconsistency  at ' + ts, end=' ')
                print(contradictions1)
                print(contradictions2)
                   
            longss = longss.difference(contradictions1)
            longss = longss.difference(contradictions2)
                          
            shortss = shortss.difference(contradictions1)
            shortss = shortss.difference(contradictions2)
 
            shortss=set([])
            d=list(filter(lambda li:len(li)>0,detls[ts]))
    
             
                      
            lastMessage = 'L:{0} S:{1}'.format(' '.join(longss), ' '.join(shortss))
            lastExtMessage = '#'.join(d)
                 
            with open(actlogJ, 'a') as fa:
                addt='{0},{1},{2}\n'.format(ts,lastMessage,lastExtMessage)
                fa.write(addt)
                fa.close()


    def joinMaillogsP(self,pred):
            
        actlogs = sorted(glob.glob("*_maillog.csv"))
        actlogJ=self.datasource+'_'+pred+'_joint_maillog.csv'
        timestamps=set([])
        if os.path.exists(actlogJ):
            os.remove(actlogJ)
        
        actdfs={}
        actdfsS=[]
   
        for actlog in actlogs:
            
            if self.datasource not in actlog:
                continue 
            if 'joint_maillog' in actlog:
                continue
            if '_' not in actlog:
                continue
            
            if actlog.split('_')[1]==pred:
                
                #print('Adding ' + actlog)
                
                try:
                    df = pandas.read_csv(actlog,index_col=0,header=None)
                    if len(df)<1:
                        continue
                    df.fillna('',inplace=True)
                    df = df[~df.index.duplicated(keep='first')]
        
                    actdfs[actlog]=df         
                    timestamps = timestamps.union(set(list(df.index)))
                except:
                    pass
                
        timestamps=sorted(timestamps)
        
        stickers={}
        detls={}
        stickersS={}
            
        for ts in timestamps:
            sts=[]
            
            detll=[]
                    
            for actl in actdfs.keys():
                df=actdfs[actl]
                av=df.index.values
                ids=numpy.where(av==ts) 
                if len(ids)>0:
                    try:
                        sts.append(df.iloc[ids[0][0],0])         
                        detl = df.iloc[ids[0][0],1]   
                        detll.append(detl)
                        
                    except:
                        pass
                else:
                    sts.append('')
            
            
            detls[ts] = detll
            stickers[ts]=sts  
                
        for ts in timestamps:
     
            chosenlists=[st.replace(':',' ').split(' ') for st in stickers[ts]]
            
            tlongs=[]
            tshorts=[]
            tshortsS=[]
            tlongsS=[]
            actls = list(actdfs.keys())
            for ic in range(0,len(chosenlists)):
                chosenlist = chosenlists[ic]
                if 'L' in chosenlist or 'S' in chosenlists:
                    li = chosenlist.index('L')
                    si = chosenlist.index('S')
            
                    longs =  set(chosenlist[li+1:si])
                    shorts = set(chosenlist[si+1:])
                    
                    longs=set(filter(lambda e:len(e)>0,longs))
                    shorts=set(filter(lambda e:len(e)>0,shorts)) 
                    
#                     longs=set(filter(lambda e:'USD'==e[:3],longs))
#                     shorts=set(filter(lambda e:'USD'==e[:3],shorts)) 
    
                    tlongs.append(longs)
                    tshorts.append(shorts)
           
            templongs=set([])
            tempshorts=set([])
            
            if len(tlongs)>0:
                templongs=set(tlongs[0]).union(*tlongs)
                
            if len(tshorts)>0:
                tempshorts=set(tshorts[0]).union(*tshorts)

             
            longss=set(filter(lambda li:len(li)>0,templongs))
            shortss=set(filter(lambda li:len(li)>0,tempshorts)) #ada,tsf - bad ones, eec+brf -shorts, gbf only longs
            #shortss=set([]) 

                
            # 1 CF:eec 0.58,gbf 0.56, {'eec':3,'gbf':1} ada':-3 brf':-8
            
            #2 {'ada':-5,'gbf':1}
            # 3 {'brf':5,'eec':4} 'gbf':-11,
            limpredsL={} #{'eec':5,'brf':6}
            for k in limpredsL.keys():
                if pred in k and len(longss)>0:
                    if limpredsL[k]>0 and len(longss)<=limpredsL[k]:
                        #print(pred +  ' ' + ts + ' ' + str(len(longss))+ ' items - LONGSKIP')
                        longss=set([])
                     
                    if limpredsL[k]<0 and len(longss)>=-limpredsL[k]:
                        #print(pred +  ' ' + ts + ' ' + str(len(longss))+ ' items - LONGSKIP')
                        longss=set([])
                       
                    break
             
            limpredsS={} #{'eecS':2,'tsfS':3}
            for k in limpredsS.keys():
                if pred in k and len(shortss)>0:
                    if limpredsS[k]>0 and len(shortss)<=limpredsS[k]:
                        #print(pred +  ' ' + ts + ' ' + str(len(longss))+ ' items - SHORTskip')
                        shortss=set([])
                    
                    if limpredsS[k]<0 and len(shortss)>=-limpredsS[k]:
                        #print(pred +  ' ' + ts + ' ' + str(len(longss))+ ' items - SHORTskip')
                        shortss=set([])    
                    break
                      
            blacklist=['GOOGL','SMMT']
            for bf in blacklist:
                
                if bf in longss:
                    longss.remove(bf)
                
                if bf in shortss:
                    shortss.remove(bf)
            


            lastMessage = 'L:{0} S:{1}'.format(' '.join(longss), ' '.join(shortss))
            #lastExtMessage = '#'.join(d)
                
            with open(actlogJ, 'a') as fa:
                addt='{0},{1}\n'.format(ts,lastMessage) #,lastExtMessage)
                fa.write(addt)
                fa.close()
        
    
