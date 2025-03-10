from Configuration import Configuration
from predictors import Predictor
import numpy
import pandas
from Service.Utilities import Tools
import traceback
import os
import json
#import xgboost
from sklearn.preprocessing import StandardScaler,MinMaxScaler

import talib
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller
from scipy.stats import linregress
from scipy.ndimage import shift
from sklearn.svm import SVC
from scipy.signal import find_peaks
from scipy import signal
from sklearn.feature_selection import RFECV,RFE,SequentialFeatureSelector,SelectFromModel
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier,BaggingClassifier

from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV,KFold
import ta
from imblearn.ensemble import RUSBoostClassifier,EasyEnsembleClassifier #,BalancedRandomForestClassifier


class featurePredictorS(Predictor.Predictor):
    PK=0.05
    TRSIZE=100
    SHW=30
    
    class customPredictors:
        
        def extrCCPClassifier(ccpalpha,X=None,Y=None,GR=0.5,K=0.1):
            crit='log_loss'
            lr=0.5
            H=Configuration.seloffset
            VL=1-featurePredictorS.SHW/H
            
            
            TSZ=featurePredictorS.TRSIZE
            param_grid = [0.8] #[0.5,0.65,0.8]
            cla=ExtraTreesClassifier(random_state=58645,ccp_alpha=ccpalpha,n_estimators=featurePredictorS.TRSIZE,bootstrap=False,criterion=crit,max_features=param_grid[0])
            if X is not None and Y is not None:
                
                lopt=lr
                aopt=0.0
                V = int(VL*len(X))
                X_train, X_val, y_train, y_val= X[:V], X[V:], Y[:V],Y[V:]

                for l in param_grid:   
                    estimator=ExtraTreesClassifier(random_state=58645,ccp_alpha=ccpalpha,
                                                   n_estimators=featurePredictorS.TRSIZE,
                                                   bootstrap=False,criterion=crit,max_features=l)
                    estimator.fit(X_train,y_train)
                    y_pred = estimator.predict(X_val)
                    ac=accuracy_score(y_val,y_pred)
                    if ac>aopt:
                        aopt=ac
                        lopt=l
                        
                cla=ExtraTreesClassifier(random_state=58645,ccp_alpha=ccpalpha,
                                         n_estimators=featurePredictorS.TRSIZE,
                                         bootstrap=False,criterion=crit,max_features=lopt)
            return cla
        
        def rffCCPClassifier(ccpalpha,X=None,Y=None,GR=0.5,K=0.1):
            crit='log_loss'
            lr=0.5
            H=Configuration.seloffset
            VL=1-featurePredictorS.SHW/H
            
            TSZ=featurePredictorS.TRSIZE
            param_grid = [0.8]
            cla=RandomForestClassifier(max_features=param_grid[0],random_state=58645,verbose=False,n_estimators=featurePredictorS.TRSIZE,ccp_alpha=ccpalpha,bootstrap=False,criterion=crit)
            if X is not None and Y is not None:
                
                lopt=lr
                aopt=0.0
                V = int(VL*len(X))
                X_train, X_val, y_train, y_val= X[:V], X[V:], Y[:V],Y[V:]

                for l in param_grid:   
                    estimator=RandomForestClassifier(max_features=l,random_state=58645,verbose=False,
                                                     n_estimators=featurePredictorS.TRSIZE,ccp_alpha=ccpalpha,
                                                     bootstrap=False,criterion=crit)
                    estimator.fit(X_train,y_train)
                    y_pred = estimator.predict(X_val)
                    ac=accuracy_score(y_val,y_pred)
                    if ac>aopt:
                        aopt=ac
                        lopt=l
                        
                cla=RandomForestClassifier(max_features=lopt,random_state=58645,verbose=False,
                                           n_estimators=featurePredictorS.TRSIZE,ccp_alpha=ccpalpha,
                                           bootstrap=False,criterion=crit)

            return cla
        
        def bgCCPClassifier(ccpalpha,X=None,Y=None,GR=0.5,K=0.1):
            crit='log_loss'
            lopt=0.65
            H=Configuration.seloffset
            VL=1-featurePredictorS.SHW/H
            TSZ=featurePredictorS.TRSIZE
            
            dt = DecisionTreeClassifier(criterion=crit,ccp_alpha=ccpalpha,random_state=37536,max_features=lopt)
        
            param_grid = [0.8]
            if X is not None and Y is not None:
                
                aopt=0.0
                V = int(VL*len(X))
                X_train, X_val, y_train, y_val= X[:V], X[V:], Y[:V],Y[V:]

                for l in param_grid:  
                    dt = DecisionTreeClassifier(criterion=crit,ccp_alpha=ccpalpha,random_state=37536,max_features=l)
                    estimator=BaggingClassifier(base_estimator=dt,bootstrap=False,n_estimators=TSZ,random_state=3574,max_features=l)
                    estimator.fit(X_train,y_train)
                    y_pred = estimator.predict(X_val)
                    ac=accuracy_score(y_val,y_pred)
                    if ac>aopt:
                        aopt=ac
                        lopt=l
                        
            cla=BaggingClassifier(base_estimator=dt,bootstrap=False,n_estimators=TSZ,random_state=3574,max_features=lopt)
            return cla
        
        def adaCCPClassifier(ccpalpha,X=None,Y=None,GR=0.5,K=0.1):
            crit='log_loss'
            adalr=0.0005
            lropt=adalr
            TSZ=featurePredictorS.TRSIZE
            H=Configuration.seloffset
            
            dtopt = DecisionTreeClassifier(criterion=crit,ccp_alpha=ccpalpha,random_state=37536,max_features=0.3)
            

            VL=1-featurePredictorS.SHW/H
            param_grid = [0.0005]
            if X is not None and Y is not None:
                
                aopt=0.0
                #param_grid = [0.5*lr,lr] #[0.45*lr,0.55*lr,0.65*lr]
                
                V = int(VL*len(X))
                X_train, X_val, y_train, y_val= X[:V], X[V:], Y[:V],Y[V:]
              
                for l in param_grid:   
                    dt = DecisionTreeClassifier(criterion=crit,ccp_alpha=ccpalpha,random_state=37536,max_features=0.3)
                    estimator=AdaBoostClassifier(base_estimator=dt,n_estimators=TSZ,random_state=3574,learning_rate=l)
                    estimator.fit(X_train,y_train) #,sample_weight=W)
                    y_pred = estimator.predict(X_val)
                    ac=accuracy_score(y_val,y_pred)
                    if ac>aopt:
                        aopt=ac
                        dtopt=dt
                        lropt=l

            cla=AdaBoostClassifier(base_estimator=dtopt,n_estimators=TSZ,random_state=3574,learning_rate=lropt)
            return cla
        
        
        def gbCCPClassifier(ccpalpha,X=None,Y=None,GR=0.5,K=0.1):
            crit='log_loss'
            lr=0.0005
            lropt=lr
            
            TSZ=featurePredictorS.TRSIZE
            H=Configuration.seloffset
            
            dtopt = GradientBoostingClassifier(n_estimators=TSZ,ccp_alpha=ccpalpha,random_state=65433,learning_rate=lropt,max_features=0.3)
            
            VL=1-featurePredictorS.SHW/H
            param_grid = [0.0005,0.001]
            if X is not None and Y is not None:
                
                aopt=0.0
                #param_grid = [0.5*lr,lr] #[0.45*lr,0.55*lr,0.65*lr]
                
                V = int(VL*len(X))
                X_train, X_val, y_train, y_val= X[:V], X[V:], Y[:V],Y[V:]
              
                for l in param_grid:   
                    estimator=GradientBoostingClassifier(n_estimators=TSZ,ccp_alpha=ccpalpha,random_state=65433,learning_rate=l,max_features=0.3)
                    estimator.fit(X_train,y_train) #,sample_weight=W)
                    y_pred = estimator.predict(X_val)
                    ac=accuracy_score(y_val,y_pred)
                    if ac>aopt:
                        aopt=ac
                        dtopt=estimator
                        lropt=l

            return dtopt
            #cla=GradientBoostingClassifier(n_estimators=TSZ,random_state=3574,learning_rate=lropt,,max_features=0.3)
            
        
        def rusCCPClassifier(ccpalpha,X=None,Y=None,GR=0.5,K=0.1):
            crit='log_loss'
            adalr=0.0005
            
            dtopt = DecisionTreeClassifier(criterion=crit,ccp_alpha=ccpalpha,random_state=37536,max_features=0.5)
            #dt=ExtraTreesClassifier(random_state=58645,ccp_alpha=ccpalpha,n_estimators=featurePredictorS.TRSIZE,bootstrap=False,criterion=crit)
            
            TSZ=featurePredictorS.TRSIZE
            H=Configuration.seloffset
            
            VL=1-featurePredictorS.SHW/H
            param_grid = [0.001,0.0006]
            if X is not None and Y is not None:
                
                aopt=0.0
                #param_grid = [0.5*lr,lr] #[0.45*lr,0.55*lr,0.65*lr]
                
                V = int(VL*len(X))
                X_train, X_val, y_train, y_val= X[:V], X[V:], Y[:V],Y[V:]
              
                for l in param_grid:   
                    dt = DecisionTreeClassifier(criterion=crit,ccp_alpha=ccpalpha,random_state=37536,max_features=0.3)
                    estimator=RUSBoostClassifier(estimator=dt,n_estimators=TSZ,random_state=3574,learning_rate=l)
                    estimator.fit(X_train,y_train) #,sample_weight=W)
                    y_pred = estimator.predict(X_val)
                    ac=accuracy_score(y_val,y_pred)
                    if ac>aopt:
                        aopt=ac
                        dtopt=dt

            cla=RUSBoostClassifier(base_estimator=dtopt,n_estimators=TSZ,random_state=3574,learning_rate=adalr)
            return cla
    
        def ensCCPClassifier(ccpalpha,X=None,Y=None,GR=0.5,K=0.1):
            crit='log_loss'
            adalr=0.0005
            
            dtopt = DecisionTreeClassifier(criterion=crit,ccp_alpha=ccpalpha,random_state=37536,max_features=0.5)
            
            TSZ=featurePredictorS.TRSIZE
            H=Configuration.seloffset
            
            VL=1-featurePredictorS.SHW/H
            param_grid = [0.001,0.0006]
            if X is not None and Y is not None:
                
                aopt=0.0
                #param_grid = [0.5*lr,lr] #[0.45*lr,0.55*lr,0.65*lr]
                
                V = int(VL*len(X))
                X_train, X_val, y_train, y_val= X[:V], X[V:], Y[:V],Y[V:]
              
                for l in param_grid:   
                    
                    dt = DecisionTreeClassifier(criterion=crit,ccp_alpha=ccpalpha,random_state=37536,max_features=0.3)
                    estbase=AdaBoostClassifier(base_estimator=dt,n_estimators=TSZ,random_state=3574,learning_rate=l)
                    estimator=EasyEnsembleClassifier(n_estimators=10,estimator=estbase, warm_start=False, sampling_strategy=0.35, 
                                                     replacement=False, n_jobs=None, random_state=75545, verbose=0)

        
                    estimator.fit(X_train,y_train) #,sample_weight=W)
                    y_pred = estimator.predict(X_val)
                    ac=accuracy_score(y_val,y_pred)
                    if ac>aopt:
                        aopt=ac
                        dtopt=dt

            cla=RUSBoostClassifier(base_estimator=dtopt,n_estimators=TSZ,random_state=3574,learning_rate=adalr)
            return cla
        
  
    class customIndicators:
        def sma(data, window): #simple moving average
            sma = data.rolling(window = window).mean()
            return sma
        
        def rdp(self,data,window):
            r=[0]*window
            for i in range(window,len(data)):
                c=100*(data[i]-data[i-window])/data[i-window]
                r.append(c)
            return r
        
        def mint(data,window):
            r=[0]*window
            for i in range(window,len(data)):
                c=min(data[i-window:i])
                r.append(c)
            return r
        
        def maxt(data,window):
            r=[0]*window
            for i in range(window,len(data)):
                c=max(data[i-window:i])
                r.append(c)
            return r
        
        def median(dataL,dataH,window):
            r=[0]*window
            for i in range(window,len(dataL)):
                c=0.5*(dataL[i-window]+dataH[i-window])
                r.append(c)
            return r
        
        def sigline(data,window1,window2):
            s1 = featurePredictorS.customIndicators.sma(data,window1)
            s2 = featurePredictorS.customIndicators.sma(data,window1)
            r=[]
            for i in range(0,len(data)):
                if s2[i]==0:
                    r.append(0)
                else:
                    c=s2[i]+(s1[i]-s2[i])/(10*s2[i])
                    r.append(c)
            return r
        
        def vwap(dataL,dataH,dataC,dataV):
            v = dataV.values
            c = dataC.values
            l = dataL.values
            h = dataH.values
            
            tp = (l + c + h)/3
            vwp = (tp * v).cumsum() / v.cumsum()
            return vwp
            
        def bb(data, sma, window): #boilinger bands
            std = data.rolling(window = window).std()
            upper_bb = sma + std * 2
            lower_bb = sma - std * 2
    
            return upper_bb, lower_bb
    
        def sarcross(close,sar): #sar crossover or under
            value = close/sar
            under = numpy.where(value<1,1,0)
            over = numpy.where(value>1,1,0)
            return under, over
            
    
        def emacross(close,ema): #exponential moving average crossover or under
            value = close/ema
            under = numpy.where(value<1,1,0)
            over  = numpy.where(value>1,1,0)
            return under, over
        
        def normalize(ts):
            return (ts - ts.min()) / (ts.max() - ts.min())
        
        def signnorm(r):
            #if type(r[0])==type(1) or type(r[0])==type(0.1):
            return [int(numpy.sign(e)) for e in r]
        
        def kc(highhist, lowhist, closehist, kc_lookback, atr_lookback):
            multiplier=2
            tr1 = pandas.DataFrame(highhist - lowhist)
            tr2 = pandas.DataFrame(numpy.abs(highhist - closehist.shift()))
            tr3 = pandas.DataFrame(numpy.abs(lowhist - closehist.shift()))
            frames = [tr1, tr2, tr3]
            tr = pandas.concat(frames, axis = 1, join = 'inner').max(axis = 1)
            atr = tr.ewm(alpha = 1/atr_lookback).mean()
    
            kc_middle = closehist.ewm(kc_lookback).mean()
            kc_upper = closehist.ewm(kc_lookback).mean() + multiplier * atr
            kc_lower = closehist.ewm(kc_lookback).mean() - multiplier * atr
    
            return kc_middle, kc_upper, kc_lower
        
        def stoch(h,l,c):
            n=14
            low_14 = l.transform(lambda x: x.rolling(window = n).min())
            high_14 = h.transform(lambda x: x.rolling(window = n).max())
            st = 100 * ((c-low_14)/(high_14-low_14))
            
            return st
    
        def macd(highhist,lowhist,closehist,volhist):
            
            def split_dataframe(df, chunk_size = 200): 
                chunks = list()
                num_chunks = len(df) // chunk_size + 1
                for i in range(num_chunks):
                    chunks.append(df[i*chunk_size:(i+1)*chunk_size])
                return chunks
            
            def fibonacci_retracement_levels(df):
                start = df['Low'].min()
                end = df['High'].max()
                
                levels = [0, 23.6, 38.2, 50, 61.8]
                retracement_values = []
                for level in levels:
                    retracement = start + (level / 100) * (end - start)
                    retracement_values.append(retracement)

                return retracement_values
            
            def fibonacci_retracement_vectors(df):
                
                levels = fibonacci_retracement_levels(df)
                fibvectors=[[]]*len(levels)
                for i in range(0,len(levels)):
                    level=levels[i]
                    fibvectors[i]=[level]*len(df)
                return fibvectors
              
            
            df = pandas.DataFrame({'Settle': closehist,'High':highhist,'Low':lowhist,'Volume':volhist})    
            
            macd, macdsignal, macdhist = talib.MACD(df['Settle'].values, fastperiod=12, slowperiod=26, signalperiod=9)
            
            adx = talib.ADX(highhist,lowhist,closehist,timeperiod=14)
            adxr = talib.ADXR(highhist,lowhist,closehist,timeperiod=14)
            rsi=talib.RSI(closehist,timeperiod=14)
            obv = talib.OBV(closehist,volhist)
            
            bbu = talib.BBANDS(closehist, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0]
            bbm = talib.BBANDS(closehist, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1]
            bbl = talib.BBANDS(closehist, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2]
            
            ad = talib.AD(highhist,lowhist,closehist,volhist)
            roc = talib.ROCP(closehist)
            atr = talib.ATR(highhist,lowhist,closehist, timeperiod=14)
            htdc = talib.HT_DCPERIOD(closehist)
            
            willr = talib.WILLR(highhist,lowhist,closehist,timeperiod=14)
            
            df['MACD'] = macd
            df['MACDsignal'] = macdsignal
            
            sma5 = talib.SMA(closehist,timeperiod=5).fillna(0).values
            sma8 = talib.SMA(closehist,timeperiod=8).fillna(0).values
            sma13 = talib.SMA(closehist,timeperiod=13).fillna(0).values
            sma50 = talib.SMA(closehist,timeperiod=50).fillna(0).values
            sma200 = talib.SMA(closehist,timeperiod=200).fillna(0).values
            
            mACDSIGgtMACD = numpy.where(df['MACDsignal'] > df['MACD'], 1, 0)
            longsignal=numpy.zeros(len(df))
            shortsignal=numpy.zeros(len(df))
            
            for ii in range(1,len(df)):
                if sma50[ii]>sma200[ii] and sma50[ii-1]<sma200[ii-1]:
                    longsignal[ii]=1
                if sma5[ii]>sma13[ii] and sma5[ii-1]<=sma13[ii-1] and sma8[ii]>sma13[ii]:
                    shortsignal[ii]=1
                                                                           
                
                
            
            
            #longsignal1=numpy.where(df['sma50']>df['sma200']) and numpy.df['sma50'].shift(1)<df['sma200'].shift(1),1,0)
            #shortsignal1=numpy.where(df['sma5']>df['sma13'] and df['sma5'].shift(1)<=df['sma13'].shift(1) and df['sma8']>df['sma13'],1,0)
            
            dfchunks = split_dataframe(df)
            fibvectors = [[]]*5
            
            for df in dfchunks:
                if len(df)>0:
                    fvs = fibonacci_retracement_vectors(df)
                    for j in range(0,len(fvs)):
                        fibvectors[j]=fibvectors[j]+fvs[j]
 
            return [macd,macdsignal,mACDSIGgtMACD,roc,adx,adxr,rsi,bbu,bbm,bbl,atr,htdc] #+fibvectors
            #return [macd,macdsignal,willr,adx,adxr,rsi,bbu,bbm,bbl,atr,htdc] #+[obv]
        
        #+[df['dcl'].values,df['dcm'].values,df['dcu'].values]
        
#         def kc(highhist, lowhist, closehist, kc_lookback, atr_lookback):
#             multiplier=2
#             tr1 = pandas.DataFrame(highhist - lowhist)
#             tr2 = pandas.DataFrame(numpy.abs(highhist - closehist.shift()))
#             tr3 = pandas.DataFrame(numpy.abs(lowhist - closehist.shift()))
#             frames = [tr1, tr2, tr3]
#             tr = pandas.concat(frames, axis = 1, join = 'inner').max(axis = 1)
#             atr = tr.ewm(alpha = 1/atr_lookback).mean()
#     
#             kc_middle = closehist.ewm(kc_lookback).mean()
#             kc_upper = closehist.ewm(kc_lookback).mean() + multiplier * atr
#             kc_lower = closehist.ewm(kc_lookback).mean() - multiplier * atr
#     
#             return kc_middle, kc_upper, kc_lower
        
#         def macd(highhist,lowhist,closehist):
#             df = pandas.DataFrame({'Settle': closehist,'High':highhist,'Low':lowhist})
#             
#             macd, macdsignal, macdhist = talib.MACD(df['Settle'].values, fastperiod=12, slowperiod=26, signalperiod=9)
#             df['MACD'] = macd
#             df['MACDsignal'] = macdsignal
#             mACDSIGgtMACD = numpy.where(df['MACDsignal'] > df['MACD'], 1, 0)
#             
#             df['EMA5'] = talib.EMA(df['Settle'].values, timeperiod=5)
#             df['EMA10'] = talib.EMA(df['Settle'].values, timeperiod=10)
#             df['EMA20'] = talib.EMA(df['Settle'].values, timeperiod=20)
#             df['EMA30'] = talib.EMA(df['Settle'].values, timeperiod=30)
#             df['EMA40'] = talib.EMA(df['Settle'].values, timeperiod=40)
#             
#             df['SMA5'] = talib.SMA(df['Settle'].values, timeperiod=5)
#             df['SMA10'] = talib.SMA(df['Settle'].values, timeperiod=10)
#             df['SMA20'] = talib.SMA(df['Settle'].values, timeperiod=20)
#             df['SMA30'] = talib.SMA(df['Settle'].values, timeperiod=30)
#             df['SMA40'] = talib.SMA(df['Settle'].values, timeperiod=40)
#             
#         
#             emcross5 = numpy.where(df['Settle'] > df['EMA5'], 1, 0)
#             emcross10 = numpy.where(df['Settle'] > df['EMA10'], 1, 0)
#             emcross20 = numpy.where(df['Settle'] > df['EMA20'], 1, 0)
#             emcross30 = numpy.where(df['Settle'] > df['EMA30'], 1, 0)
#             
#             emscross5 = numpy.where(df['Settle'] > df['SMA5'], 1, 0)
#             emscross10 = numpy.where(df['Settle'] > df['SMA10'], 1, 0)
#             emscross20 = numpy.where(df['Settle'] > df['SMA20'], 1, 0)
#             emscross30 = numpy.where(df['Settle'] > df['SMA30'], 1, 0)
#             
#             em5 = numpy.where(df['EMA5'] > df['EMA10'], 1, 0)
#             em1030 = numpy.where(df['EMA10'] > df['EMA30'], 1, 0)
#             em1020 = numpy.where(df['EMA10'] > df['EMA20'], 1, 0)
#             em1040 = numpy.where(df['EMA10'] > df['EMA40'], 1, 0)
#             
#             ems5 = numpy.where(df['SMA5'] > df['SMA10'], 1, 0)
#             ems1030 = numpy.where(df['SMA10'] > df['SMA30'], 1, 0)
#             ems1020 = numpy.where(df['SMA10'] > df['SMA20'], 1, 0)
#             ems1040 = numpy.where(df['SMA10'] > df['SMA40'], 1, 0)
#             
#             #em2040 = numpy.where(df['EMA20'] > df['EMA40'], 1, -1)
#             #em2030 = numpy.where(df['EMA20'] > df['EMA30'], 1, -1)
#             
#             return [macd,macdsignal,mACDSIGgtMACD]+[em5,ems1020,ems1030,ems1040,emscross5,emscross10]
            
#             data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
#             data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
#             data['MACD'] = data['EMA12'] - data['EMA26']
#             data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
#             
#             L=0
#             last_row = data.iloc[L+1]
#             second_last_row = data.iloc[L]
#             macds=[0]
#             while L<len(data)-1:
# 
#                 if second_last_row['MACD'] > second_last_row['Signal_Line'] and last_row['MACD'] < last_row['Signal_Line']:
#                     print('Cross Below Signal Line')
#                     macds.append(-1)
#                 elif second_last_row['MACD'] < second_last_row['Signal_Line'] and last_row['MACD'] > last_row['Signal_Line']:
#                     print('Cross Above Signal Line')
#                     macds.append(1)
#                 else:
#                     #print('No Crossover')
#                     macds.append(0)
#                 
#                 L=L+1
            #return macds
            
    
    
    def __init__(self, dataManager,name,stick,context,candle=False):
        self.stickers = [stick]
        super().__init__(name,dataManager)
        self.engine = None
        self.startOffset=Configuration.TSET
        self.dmContext = context
        numpy.random.seed(13)
        
        self.TT=0.75 #0.8
        self.TTL=0.8
        self.TTS=0.8
        
        self.PIVOT=0.5
        self.GENPIVOT=0.5
        
        self.LTAU=0.75 #0.75
        self.RTAU=0.75
        self.SHWINDOW=featurePredictorS.SHW
        self.PATH=None
        self.FPATH=None
        self.candles=candle
        self.treecrit='log_loss'
        self.bootstrap=False
        self.featurerandom=True
        self.oob=False
        self.LG=[1] #[0,0.25,0.5,0.75,1]
        self.VALIDPRUNE=1.0-1.0*self.SHWINDOW/Configuration.seloffset #0.94
        self.PK=featurePredictorS.PK
           
        self.STATIONARY=1.05
        self.BERN=False
        
        if self.STATIONARY>1:
            self.BERN=True
            
    
    
        self.KS=[0.8]
        self.MINK=0.0
        self.MAXK=50
        self.KSTEP=2
        self.ISTEP=0.0015
        
        self.LGS=[0.6,0.4]
        self.LGSTEP=0.1
        self.MINLG=0.0
        
        self.MAXFT='sqrt'
        self.NUMFEATURES=1.1
            
        self.LIMESTS=10
        self.gblr=0.0001
        self.LRs=1
        
        os.environ['PYTHONHASHSEED']=str(1)

    def runAll(self,forday=None):
        
        self.preprocess(forday)
    
    def formFeatures(self,sticker,timestamp,noisecoeff,tcoeff):

        aggrwin=[5,10,20,50,100,150,200] 
 
             
        L=max(Configuration.seqhists)
         
        v=False
        taggr=not self.candles
        t3=self.candles
   
        closehist=self.dataManager.get(sticker,'close',timestamp,L)
        volhist = self.dataManager.get(sticker,'volume',timestamp,L)
        highhist = self.dataManager.get(sticker,'high',timestamp,L)
        lowhist = self.dataManager.get(sticker,'low',timestamp,L)
        openhist = self.dataManager.get(sticker,'open',timestamp,L)
          
        if len(closehist)==0:
            return None  
         
         
        H=Configuration.seloffset
        VL=1-featurePredictorS.SHW/H
         
        #-----------------------------------------------------#
        O_S=[3.0] #[2.9,2.65,2.2,1.9]
        TLS=[0] #[2.9,2.65,2.2,1.9]
         
        for oi in range(0,len(O_S)):
            O=O_S[oi]
            tls=TLS[oi]
             
            V = int(tls*len(closehist))
            closehist_=closehist[V:]
            lowhist_=lowhist[V:]
            highhist_=highhist[V:]
             
            nuaC=numpy.array(closehist_.values)
         
            m = numpy.mean(nuaC, axis=0)
            sd = numpy.std(nuaC, axis=0)
         
            out1 = [x for x in nuaC if (x>m+O*sd)]
            out2 = [x for x in nuaC if (x<m-O*sd)]
          
            if len(out1)>0 or len(out2)>0:
                return None
         
         
        lbs = [0]+[max(0,int(numpy.sign(closehist[ii + 1] - closehist[ii]))) for ii in range(len(closehist) - 1)]
         
         
        dataframe=pandas.DataFrame()
        dataframe['Open'] = openhist
        dataframe['High'] = highhist
        dataframe['Low'] = lowhist
        dataframe['Close'] = closehist
        dataframe['Volume'] = volhist
 
        dfloc=dataframe
        dweek=[]
        months=[]
        quarters=[]
         
        OFFS=0
        for t in list(dfloc.index):
            try:
                quarters.append(OFFS+t.quarter)
            except:
                quarters.append(0)
             
            try:
                dweek.append(OFFS+t.day_of_week)
            except:
                dweek.append(0)
                 
            try:
                months.append(OFFS+t.month)
            except:
                months.append(0)
             
 
        vwp=featurePredictorS.customIndicators.vwap(lowhist, highhist,closehist,volhist)
        if t3:
            #dfloc['3DAYWEEK'] = dweek
            #dfloc['3MONTH'] = months
            #dfloc['3QUARTER'] = quarters
            #cvwp=[int(numpy.sign(closehist[i]-vwp[i])) for i in range(0,len(vwp))]
            #dfloc['3VWP'] = cvwp
            
            cdls=talib.get_function_groups()['Pattern Recognition']
            for cdl in cdls:
                cdl_func = getattr(talib, cdl)
                p=cdl_func(openhist,highhist,lowhist,closehist)
                psgn = [int(numpy.sign(x)) for x in p]
                dfloc['3'+cdl] = psgn
  
        else:
            pass
            #dfloc['DAYWEEK'] = dweek
            #dfloc['MONTH'] = months
            #dfloc['QUARTER'] = quarters
            #dfloc['VWP']=vwp

        if taggr:
            macds = featurePredictorS.customIndicators.macd(highhist,lowhist,closehist,volhist)
            #stch = featurePredictorS.customIndicators.stoch(highhist,lowhist,closehist)
            #dfloc['STOCH'] = stch
            
            for mi in range(0,len(macds)):
                
                dfloc['MACD'+str(mi+1)] = macds[mi]
            for window1 in aggrwin:
                  
                dfloc['sma'+str(window1)] = featurePredictorS.customIndicators.sma(dfloc['Close'],window1)
                if window1<0:
                    dfloc['smav'+str(window1)] = featurePredictorS.customIndicators.sma(dfloc['Volume'],window1)
#                 dfloc['ema'+str(window1)] = talib.EMA(dfloc['Close'], timeperiod=window1)
#                   
                #dfloc['rsi'+str(window1)] =talib.RSI(dfloc['Close'],window1)
#                 dfloc['bblower'+str(window1)] = featurePredictorS.customIndicators.bb(dfloc['Close'],dfloc['sma'+str(window1)], window1)[1]
#                 dfloc['bbupper'+str(window1)] = featurePredictorS.customIndicators.bb(dfloc['Close'],dfloc['sma'+str(window1)], window1)[0]
#                 dfloc['bbupperratio'+str(window1)] = dfloc['Close']/dfloc['bbupper'+str(window1)]
#                 dfloc['bblowerratio'+str(window1)] = dfloc['Close']/dfloc['bblower'+str(window1)]
#                   
#                 dfloc['arnd'+str(window1)],dfloc['arnu'+str(window1)] = talib.AROON(dfloc['High'],dfloc['Low'],timeperiod=window1)
#            
                #dfloc['willr'+str(window1)] = talib.WILLR(dfloc['High'],dfloc['Low'],dfloc['Close'],timeperiod=window1)
                #dfloc['cci'+str(window1)] = talib.CCI(dfloc['High'],dfloc['Low'],dfloc['Close'], timeperiod=window1)
                #dfloc['mom'+str(window1)] = talib.MOM(dfloc['Close'], timeperiod=window1)
                #dfloc['roc'+str(window1)] = talib.ROC(dfloc['Close'], timeperiod=window1)
#                 dfloc['bias'+str(window1)] = (dfloc['Close']-dfloc['sma'+str(window1)])/window1
#                 dfloc['ATR'+str(window1)] = talib.ATR(dfloc['High'],dfloc['Low'],dfloc['Close'], timeperiod = window1)
#                 dfloc['med'+str(window1)] = featurePredictorS.customIndicators.median(dfloc['Low'],dfloc['High'],window=window1)
#                 dfloc['dpo'+str(window1)] = ta.trend.DPOIndicator(closehist, window1,True).dpo()  
                     
                                                              
        closehist = dfloc['Close'].values
        labs = [0]+[max(0,int(numpy.sign(closehist[ii + 1] - closehist[ii]))) for ii in range(len(closehist) - 1)]
        dfloc=dfloc.fillna(0)
         
        if taggr==True:   
            LMW = max(aggrwin)
            dfloc=dfloc[LMW:]
            labs = labs[LMW:]
 
        
        #dfloc = dfloc.drop(columns=['Open','High','Low','Close','Volume'],axis=1)
        dfloc = dfloc.drop(columns=['Open','High','Low','Close','Volume'],axis=1)
        #dfloc = dfloc[['Open','High','Low','Close','MACD1','MACD2']]
         
        for c in dfloc.columns:
            v = dfloc[c].values
            if '3' in c:
                 
                vc=featurePredictorS.customIndicators.signnorm(v)
                dfloc[c]=dfloc[c].astype('category',copy=False)
            else:
                vc=v # featurePredictorS.customIndicators.normalize(v)
            dfloc[c]=vc
         
        
        
        scale = MinMaxScaler(feature_range=(-1,1))
        d = scale.fit_transform(dfloc)
        dataframeFinal = pandas.DataFrame(d, columns=dfloc.columns)
        dataframeFinal.set_index(dfloc.index,inplace=True)
        
        selection=PCA(n_components=0.85, svd_solver='full',random_state=12339)          
        X_pc = selection.fit_transform(dataframeFinal,labs)
        n_pcs= selection.components_.shape[0] 
        most_important = [numpy.abs(selection.components_[i]).argmax() for i in range(n_pcs)]
        selX = list(set([dataframeFinal.columns[most_important[i]] for i in range(n_pcs)]))
        dataframeFinal=dataframeFinal[selX] 
         
        return (dataframeFinal,labs)
    

    # https://github.com/fizahkhalid/forex_factory_calendar_news_scraper/blob/main/scraper.py
#     def formFeatures(self,sticker,timestamp,noisecoeff,tcoeff):   #[5,10,15,20,25,30,35,40,45,50,55,60][5,10,15,20],[5,10,20],[10,20,30]
#          
#          
#         def lowpassfilter(signal, thresh, wavelet):
#             thresh = thresh*numpy.nanmax(signal)
#             coeff = pywt.wavedec(signal, wavelet, mode="per" )
#             coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
#             reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
#             return reconstructed_signal
#         
#         if Configuration.HOURLY:
#             aggrwin=[6,12,18,24,30,36,42,48,54,60,66,72] 
#         else:
#             aggrwin=[10,20,30,40,60,80,100,140,180,220] #,220,260,300]
# 
#             
#         L=max(Configuration.seqhists)
#         
#         v=False
#         taggr=not self.candles
#         t3=self.candles
#   
#         closehist=self.dataManager.get(sticker,'close',timestamp,L)
#         volhist = self.dataManager.get(sticker,'volume',timestamp,L)
#         highhist = self.dataManager.get(sticker,'high',timestamp,L)
#         lowhist = self.dataManager.get(sticker,'low',timestamp,L)
#         openhist = self.dataManager.get(sticker,'open',timestamp,L)
#          
#         if len(closehist)==0:
#             return None  
#         
#         
#         H=Configuration.seloffset
#         VL=1-featurePredictorS.SHW/H
#         
#         #-----------------------------------------------------#
#         O_S=[2.9,2.65,2.2,1.9] #[2.9,2.1,1.9]
#         TLS=[0,0.3,0.5,0.75] #[0,0.5,0.7]
#         
#         for oi in range(0,len(O_S)):
#             O=O_S[oi]
#             tls=TLS[oi]
#             
#             V = int(tls*len(closehist))
#             closehist_=closehist[V:]
#             lowhist_=lowhist[V:]
#             highhist_=highhist[V:]
#             
#             nuaC=numpy.array(closehist_.values)
#         
#             m = numpy.mean(nuaC, axis=0)
#             sd = numpy.std(nuaC, axis=0)
#         
#             out1 = [x for x in nuaC if (x>m+O*sd)]
#             out2 = [x for x in nuaC if (x<m-O*sd)]
#          
#             if len(out1)>0 or len(out2)>0:
#                 return None
#         
#         
#         lbs = [0]+[max(0,int(numpy.sign(closehist[ii + 1] - closehist[ii]))) for ii in range(len(closehist) - 1)]
#         
#         
#         dataframe=pandas.DataFrame()
#         dataframe['Open'] = openhist
#         dataframe['High'] = highhist
#         dataframe['Low'] = lowhist
#         dataframe['Close'] = closehist
#         dataframe['Volume'] = volhist
# 
#         dfloc=dataframe
#         dweek=[]
#         months=[]
#         quarters=[]
#         
#         OFFS=0
#         for t in list(dfloc.index):
#             try:
#                 quarters.append(OFFS+t.quarter)
#             except:
#                 quarters.append(0)
#             
#             try:
#                 dweek.append(OFFS+t.day_of_week)
#             except:
#                 dweek.append(0)
#                 
#             try:
#                 months.append(OFFS+t.month)
#             except:
#                 months.append(0)
#             
# 
#         vwp=featurePredictorS.customIndicators.vwap(lowhist, highhist,closehist,volhist)
#         if t3:
#             dfloc['3DAYWEEK'] = dweek
#             dfloc['3MONTH'] = months
#             dfloc['3QUARTER'] = quarters
#             cvwp=[int(numpy.sign(closehist[i]-vwp[i])) for i in range(0,len(vwp))]
#             dfloc['3VWP'] = cvwp
# 
#             
#         else:
#             dfloc['DAYWEEK'] = dweek
#             dfloc['MONTH'] = months
#             dfloc['QUARTER'] = quarters
#         
#         if t3:
# 
#             cdls=talib.get_function_groups()['Pattern Recognition']
#             for cdl in cdls:
#                 cdl_func = getattr(talib, cdl)
#                 p=cdl_func(openhist,highhist,lowhist,closehist)
#                 psgn = [int(numpy.sign(x)) for x in p]
#                 dfloc['3'+cdl] = psgn
#         
#         
#         #dfloc['momentum_kama'] = ta.momentum.kama(dfloc['Close'])
#         #dfloc['trend_psar_up'] = ta.trend.psar_up(dfloc['High'], dfloc['Low'], dfloc['Close'])
#         #dfloc['volume_vwap'] = ta.volume.VolumeWeightedAveragePrice(dfloc['High'], dfloc['Low'], dfloc['Close'], dfloc['Volume']).volume_weighted_average_price()
#         
#         #dfloc['ichimoku_bb'] = ta.trend.ichimoku_b(dfloc['High'], dfloc['Low'])
#         #dfloc['ichimoku_base'] = ta.trend.ichimoku_base_line(dfloc['High'], dfloc['Low'])
#         #dfloc['ichimoku_conv'] = ta.trend.ichimoku_a(dfloc['High'], dfloc['Low'])
#         
#         #dfloc['volatility_dcm'] = ta.volatility.DonchianChannel(dfloc['High'], dfloc['Low'], dfloc['Close']).donchian_channel_mband()
#         #dfloc['volatility_dch'] = ta.volatility.DonchianChannel(dfloc['High'], dfloc['Low'], dfloc['Close']).donchian_channel_hband()
#         
#         #dfloc['volatility_bbl'] = ta.volatility.BollingerBands(dfloc['Close']).bollinger_lband()
#         #dfloc['volatility_bbm'] = ta.volatility.BollingerBands(dfloc['Close']).bollinger_mavg()
# 
#         
#         
# #         dfloc['stochk'] = ta.momentum.stoch(dfloc['High'],dfloc['Low'],dfloc['Close'],window=5,smooth_window=3)
# #         dfloc['ATR'] = talib.ATR(dfloc['High'],dfloc['Low'],dfloc['Close'], timeperiod = 10)
# #         dfloc['med'] = featurePredictorS.customIndicators.median(dfloc['Low'],dfloc['High'],window=10)
# #         dfloc['sigline'] = featurePredictorS.customIndicators.sigline(dfloc['Close'],10,20)
# #         dfloc['mint'] = featurePredictorS.customIndicators.mint(dfloc['Low'],10)
# #         dfloc['maxt'] = featurePredictorS.customIndicators.maxt(dfloc['High'],10)
# #         dfloc['ultosc'] = talib.ULTOSC(dfloc['High'],dfloc['Low'],dfloc['Close'],10,20,30)
#          
#         if taggr:
#             
#             dfloc['VWP']=vwp
#             macds = featurePredictorS.customIndicators.macd(highhist,lowhist,closehist,volhist)
#             
#             for mi in range(0,len(macds)):
#                 dfloc['MACD'+str(mi+1)] = macds[mi]
#             for window1 in aggrwin:
#                  
#                 dfloc['sma'+str(window1)] = featurePredictorS.customIndicators.sma(dfloc['Close'],window1)
#                 dfloc['ema'+str(window1)] = talib.EMA(dfloc['Close'], timeperiod=window1)
#                  
#                 dfloc['rsi'+str(window1)] =talib.RSI(dfloc['Close'],window1)
#                 dfloc['bblower'+str(window1)] = featurePredictorS.customIndicators.bb(dfloc['Close'],dfloc['sma'+str(window1)], window1)[1]
#                 dfloc['bbupper'+str(window1)] = featurePredictorS.customIndicators.bb(dfloc['Close'],dfloc['sma'+str(window1)], window1)[0]
#                 dfloc['bbupperratio'+str(window1)] = dfloc['Close']/dfloc['bbupper'+str(window1)]
#                 dfloc['bblowerratio'+str(window1)] = dfloc['Close']/dfloc['bblower'+str(window1)]
#                  
#                 dfloc['arnd'+str(window1)],dfloc['arnu'+str(window1)] = talib.AROON(dfloc['High'],dfloc['Low'],timeperiod=window1)
#           
#                 dfloc['willr'+str(window1)] = talib.WILLR(dfloc['High'],dfloc['Low'],dfloc['Close'],timeperiod=window1)
#                 dfloc['cci'+str(window1)] = talib.CCI(dfloc['High'],dfloc['Low'],dfloc['Close'], timeperiod=window1)
#                 dfloc['mom'+str(window1)] = talib.MOM(dfloc['Close'], timeperiod=window1)
#                 dfloc['roc'+str(window1)] = talib.ROC(dfloc['Close'], timeperiod=window1)
#                 dfloc['bias'+str(window1)] = (dfloc['Close']-dfloc['sma'+str(window1)])/window1
#                 dfloc['ATR'+str(window1)] = talib.ATR(dfloc['High'],dfloc['Low'],dfloc['Close'], timeperiod = window1)
#                 dfloc['med'+str(window1)] = featurePredictorS.customIndicators.median(dfloc['Low'],dfloc['High'],window=window1)
#                 dfloc['dpo'+str(window1)] = ta.trend.DPOIndicator(closehist, window1,True).dpo()
#                 
#                 #dfloc['adi'+str(window1)] = ta.volume.EaseOfMovementIndicator(highhist,lowhist,volhist,window1,True).ease_of_movement()
#                 
# #                 if window1>=100:
# #                     DCL=ta.volatility.DonchianChannel(dfloc['High'], dfloc['Low'], dfloc['Close'],window=window1,offset=0,fillna=True)
# #                     dfloc['volatility_dcl'+str(window1)] = DCL.donchian_channel_lband()
# #                     dfloc['volatility_dcc'+str(window1)] = DCL.donchian_channel_mband()
# #                     dfloc['volatility_dch'+str(window1)] = DCL.donchian_channel_hband()
# #                     kcm,kcu,kcl=featurePredictorS.customIndicators.kc(dfloc['High'], dfloc['Low'], dfloc['Close'], window1, window1/2)
# #                     dfloc['volatility_kcl'+str(window1)]=kcl
# #                     dfloc['volatility_kcu'+str(window1)]=kcu
# #                     dfloc['volatility_kcm'+str(window1)]=kcm
#                 
# #                     KL = ta.volatility.KeltnerChannel(dfloc['High'], dfloc['Low'], dfloc['Close'],window=window1,fillna=True)
# #                     dfloc['volatility_kcl'+str(window1)] = KL.keltner_channel_lband()
# #                     dfloc['volatility_kcc'+str(window1)] = KL.keltner_channel_mband()
# #                     dfloc['volatility_kch'+str(window1)] = KL.keltner_channel_hband()
# #                     dfloc['volatility_kcp'+str(window1)] = KL.keltner_channel_pband()
# #                     dfloc['volatility_kcw'+str(window1)] = KL.keltner_channel_wband()
#                 
# #                     DCL=ta.volatility.DonchianChannel(dfloc['High'], dfloc['Low'], dfloc['Close'],window=window1,offset=0, fillna=True)
# #                     dfloc['volatility_dcl'+str(window1)] = DCL.donchian_channel_lband()
# #                     dfloc['volatility_dcc'+str(window1)] = DCL.donchian_channel_mband()
# #                     dfloc['volatility_dch'+str(window1)] = DCL.donchian_channel_hband()
# #                     dfloc['volatility_dcp'+str(window1)] = DCL.donchian_channel_pband()
# #                     dfloc['volatility_dcw'+str(window1)] = DCL.donchian_channel_wband()
#              
# #             for w1 in [10,20,40,80]:
# #                 for w2 in [10,20,40,80]:
# #                     if w2==2*w1:
# #                         dfloc['oscp'+str(w1)+'_'+str(w2)]=(dfloc['sma'+str(w1)]-dfloc['sma'+str(w2)])/dfloc['sma'+str(w1)]
# #                         dfloc['macd'+str(w1)+'_'+str(w2)]=dfloc['ema'+str(w1)]-dfloc['ema'+str(w2)]
# #         
#                     
#                                                              
#         closehist = dfloc['Close'].values
#         labs = [0]+[max(0,int(numpy.sign(closehist[ii + 1] - closehist[ii]))) for ii in range(len(closehist) - 1)]
#         dfloc=dfloc.fillna(0)
#         
#         if taggr==True:   
#             LMW = max(aggrwin)
#             dfloc=dfloc[LMW:]
#             labs = labs[LMW:]
# 
#         
#         dfloc = dfloc.drop(columns=['Open','High','Low','Close','Volume'],axis=1)
#         
#         for c in dfloc.columns:
#             v = dfloc[c].values
#             if '3' in c:
#                 
#                 vc=featurePredictorS.customIndicators.signnorm(v)
#                 dfloc[c]=dfloc[c].astype('category',copy=False)
#             else:
#                 vc=featurePredictorS.customIndicators.normalize(v)
#             dfloc[c]=vc
#         
#         
#         selection=PCA(n_components=int(len(dfloc.columns)*0.4), svd_solver='full',random_state=12339)          
#         X_pc = selection.fit_transform(dfloc,labs)
#         n_pcs= selection.components_.shape[0] 
#         most_important = [numpy.abs(selection.components_[i]).argmax() for i in range(n_pcs)]
#         selX = list(set([dfloc.columns[most_important[i]] for i in range(n_pcs)]))
#         
#         dataframeFinal=dfloc[selX] 
#         
#         return (dataframeFinal,labs)

        
    def refresh(self,sticker,timestamp):
        pass
            
    
    def objective_1(self,df,sticker,timestamp,lr,gr,K):
        
        l=len(df)-1
        X = df.iloc[:,0:-1]
        y = df.iloc[:,-1]
        
        X_train_1, X_test_1, y_train_1, y_test_1= X[:l], X[l:], y[:l],y[l:]  
        select_X_train = X_train_1         
        V = len(X)-self.SHWINDOW
        #int(self.VALIDPRUNE*len(X))
        X_train, X_val, y_train, y_val= X[:V], X[V:], y[:V],y[V:] 
        
        if 'rus' in self.name:
            model= featurePredictorS.customPredictors.rusCCPClassifier(lr,X_train,y_train)
        else:
            if 'xgb' in self.name:
                model=HistGradientBoostingClassifier(max_iter=featurePredictorS.TRSIZE,learning_rate=lr,max_depth=ht,random_state=7643,l2_regularization=2.0,early_stopping=False,n_iter_no_change=1000,verbose=0)
                #model=xgboost.XGBClassifier(max_depth=ht,verbosity=0,silent=0,n_estimators=1000,learning_rate=lr,objective='binary:logistic',reg_lambda=rlambda,reg_alpha=0,seed=165228,tree_method='exact')
            else:
                if 'ngb' in self.name:
                    #bs=DecisionTreeClassifier(criterion="log_loss",min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_depth=2,splitter="best",random_state=None)
                    bs=DecisionTreeRegressor(criterion="friedman_mse",min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_depth=ht,splitter="best",random_state=45343)
                    model = NGBClassifier(Dist=k_categorical(2),Score=LogScore,Base=bs,natural_gradient=True,n_estimators=featurePredictorS.TRSIZE,random_state=58645,verbose=False,learning_rate=lr)
                else:
                    if 'log' in self.name:
                        model = LogitBoost(n_estimators=featurePredictorS.TRSIZE,learning_rate=lr,random_state=7643,bootstrap=False)
                    else:
                        if 'tsf' in self.name: 
                            model = featurePredictorS.customPredictors.extrCCPClassifier(lr,X_train,y_train)
                        else:
                            if 'svc' in self.name:
                                model = SVC(kernel='rbf',C=lr)
                            else:
                                if 'tsb' in self.name:
                                    model = featurePredictorS.customPredictors.ensCCPClassifier(lr,X_train,y_train)
                                    #model = TimeSeriesForest(n_estimators=featurePredictorS.TRSIZE,random_state=65433,ccp_alpha=lr,bootstrap=False,oob_score=False)
                                else:
                                    if 'brf' in self.name:
                                        model = featurePredictorS.customPredictors.bgCCPClassifier(lr,X_train,y_train)
                                    else:
                                        if 'gbf' in self.name:
                                            model = featurePredictorS.customPredictors.gbCCPClassifier(lr,X_train,y_train)
                                        else:
                                            if 'ada' in self.name:
                                                model = featurePredictorS.customPredictors.adaCCPClassifier(lr,X_train,y_train)
                                            else:
                                                if 'brf' in self.name:
                                                    model = TSBF(n_estimators=featurePredictorS.TRSIZE,random_state=65433,ccp_alpha=lr,bootstrap=False)
                                                else:
                                                    model = featurePredictorS.customPredictors.rffCCPClassifier(lr,X_train,y_train)
                                                    #model=RandomForestClassifier(max_features=self.MAXFT,random_state=58645,verbose=False,n_estimators=featurePredictorS.TRSIZE,ccp_alpha=lr,bootstrap=self.bootstrap,criterion=self.treecrit)
                                            
            
        model.fit(X_train,y_train) 
        y_pred = model.predict(X_val)
   
        SLIPPAGE=0.0
        
        vv=[int(v) for v in y_val]
        gains=[]
        gainpos=[]
        gainneg=[]
        hits=[]
        rawups=[]
        sr_ups=[]
        sr_downs=[]
        for i in range(0,len(vv)):
            
            t=X_val.index[i].strftime(Configuration.timestamp_format)
            delta=self.dataManager.getDelta(sticker,'close', t)
            delta = delta-SLIPPAGE
            
            p=y_pred[i]
            pt=y_val[i]
            if numpy.fabs(p)>1:
                continue
            if p==0:
                p=-1
                
            g=delta*p
            gains.append(g)
            
            if p>0:
                gainpos.append(g)
            else:
                gainneg.append(g)
            
            hits.append(int(max(0,numpy.sign(g))))
            rawups.append(int(max(0,numpy.sign(pt))))
            
            if pt>0:
                sr_ups.append(int(max(0,numpy.sign(g))))
            else:
                sr_downs.append(int(max(0,numpy.sign(g))))
        
        T=[] #5 ALEX
        G=[]
        PREC=[]
    
        QRL=int(self.SHWINDOW/3)
        QRS=QRL
        if len(hits)>0:
            #PREC.append(1.0*sum(hits)/len(vv)) 
            G.append(1.0*sum(hits)/len(vv)) #g[0]
        else:
            #PREC.append(0) 
            G.append(0) 
            
        
        hqr=hits[-QRL:]
        g_small = gains[-QRL:]
        
        if len(hqr)>0:
            PREC.append(1.0*sum(hqr)/QRL) #sr[0]
            G.append(sum(g_small)/QRL)
        else:
            PREC.append(0)
            G.append(0)
        
        if len(gainpos)>0:
            G.append(sum(gainpos[-QRL:])/QRL)
        else:
            G.append(0)
            
        if len(gainneg)>0:
            G.append(sum(gainneg[-QRS:])/QRS)
        else:
            G.append(0)
        
        
        srup=0.0
        srdown=0.0
        try:
            srup=1.0*sum(sr_ups)/len(sr_ups)
        except:
            pass
        
        try:
            srdown=1.0*sum(sr_downs)/len(sr_downs)
        except:
            pass
        
        
        PREC.append(srup) 
        PREC.append(srdown) 
        PREC.append(1.0*sum(rawups)/len(vv)) 
        
        return (G,PREC)              
                
    
    def adapt_dt_prune(self,df):
        
        l=len(df)-1
        X = df.iloc[:,0:-1]
        y = df.iloc[:,-1]
        
        X_train_1, X_test_1, y_train_1, y_test_1= X[:l], X[l:], y[:l],y[l:]  
        V = len(X)-self.SHWINDOW        
        #V = int(self.VALIDPRUNE*len(X))
        X_train, X_val, y_train, y_val= X[:V], X[V:], y[:V],y[V:] 
        
        if 'tsf' in self.name:
            tree = featurePredictorS.customPredictors.extrCCPClassifier(0.0,X_train,y_train)
            #tree = ExtraTreesClassifier(n_estimators=featurePredictorS.TRSIZE,random_state=4000,criterion=self.treecrit)
        else:
            if 'tsb' in self.name:
                tree=featurePredictorS.customPredictors.ensCCPClassifier(0.0,X_train,y_train)
                #tree = TimeSeriesForest(n_estimators=featurePredictorS.TRSIZE,random_state=48745)
            else:
                if 'brf' in self.name:
                    tree=featurePredictorS.customPredictors.bgCCPClassifier(0.0,X_train,y_train)
                else:
                    if 'gbf' in self.name:
                        tree = featurePredictorS.customPredictors.gbCCPClassifier(0.0,X_train,y_train)
                    else:
                        if 'ada' in self.name:
                            tree=featurePredictorS.customPredictors.adaCCPClassifier(0.0,X_train,y_train)
                        else:
                            if 'rus' in self.name:
                                tree=featurePredictorS.customPredictors.rusCCPClassifier(0.0,X_train,y_train)
                            else: 
                                tree=featurePredictorS.customPredictors.rffCCPClassifier(0.0,X_train,y_train)
                                #tree=RandomForestClassifier(n_estimators=featurePredictorS.TRSIZE,random_state=4000,bootstrap=False,max_features=self.MAXFT,criterion=self.treecrit)
                                
        
        
        tree.fit(X_train,y_train)
        y_train_pred = tree.predict(X_train)
        y_val_pred = tree.predict(X_val)
        
        alphas=set([])
        for t in tree.estimators_:
            if isinstance(t, (list, tuple, numpy.ndarray))==True:
                t_=t[0]
            else:
                t_=t
            path = t_.cost_complexity_pruning_path(X_train,y_train)
            a = [round(al,2) for al in path['ccp_alphas']]
            alphas = alphas.union(set(a))
        
        max_accuracy_val=0.0
        max_alpha=0
        accdict={}
        for i in alphas:
            if 'tsf' in self.name:
                tr = featurePredictorS.customPredictors.extrCCPClassifier(i)
                #tr = ExtraTreesClassifier(n_estimators=featurePredictorS.TRSIZE,ccp_alpha=i,criterion=self.treecrit)
            else:
                if 'tsb' in self.name:
                    tr=featurePredictorS.customPredictors.ensCCPClassifier(i)
                    #tr = TimeSeriesForest(n_estimators=featurePredictorS.TRSIZE,ccp_alpha=i)
                else:
                    if 'brf' in self.name:
                        tr=featurePredictorS.customPredictors.bgCCPClassifier(i)
                    else:
                        if 'gbf' in self.name:
                            tr=featurePredictorS.customPredictors.gbCCPClassifier(i)
                        else:
                            if 'ada' in self.name:
                                tr=featurePredictorS.customPredictors.adaCCPClassifier(i)
                            else:
                                if 'rus' in self.name:
                                    tr=featurePredictorS.customPredictors.rusCCPClassifier(i)
                                else:
                                    tr=featurePredictorS.customPredictors.rffCCPClassifier(i)
                                    #tr=RandomForestClassifier(n_estimators=featurePredictorS.TRSIZE,ccp_alpha=i,bootstrap=False,max_features=self.MAXFT,criterion=self.treecrit)

            tr.fit(X_train,y_train)
            y_val_pred = tr.predict(X_val)
            a=accuracy_score(y_val,y_val_pred)
            accdict[i]=a
            if a>max_accuracy_val:
                max_alpha=i
                max_accuracy_val = a
        
        sorted_acc = {key: value for key, value in sorted(accdict.items(), key=lambda item: item[1], reverse=True)}
        alphas=list(sorted_acc.keys())
        lrs=self.LRs
        return alphas[0:lrs]
        
    
    
    def adapt_objective_1(self,df,sticker,timestamp,LR):
        #LR=  #,0.001,0.0001,0.0005]
        Lambdas=[2]
        optlr=0.0
        opta=None
        optlmbd=0.5
        optcoef=0.0
        optsrS=0.0
        optsr=None
        optncoef=0.0
        optgr=0.5
        optk=0.5
        optlg=0.5
        KS=self.KS
        LGS=self.LGS

        
        KS=[0.5]  
        LGS=[1] 
        for k in KS:
            for lr in LR:
                for gr in LGS:
                    (a,sr)=self.objective_1(df,sticker,timestamp,lr,gr,k)
                    gf = a[0] #sr[0] 
                                #gf = self.LG*sr[0]+(1-self.LG)*sr[1] #a[0] 
                                #gf = 0.25*sr[0]+0.25*sr[1]+0.5*sr[2]
                                #(a_middle,sr_middle)=self.objective_1(sticker,timestamp_middle,lr,ht,nc,tc) #ALEX

                    if gf>optsrS:
                    #gf>optsrS and sr[0]>0.5:
                        opta=a #a_middle+a
                        optlr=lr
                        optsrS=gf
                        #optsrS=gf
                        optsr=sr
                        optgr=gr
                        optk=k
            
        return (opta,optsr,optlr,optgr,optk)

    
    def predict(self,sticker,timestamp):
        
        prd0=self.name
        
        self.PATH = prd0+'_'+sticker+'.pt'
        self.OPTPATH = 'opt'+prd0+'_'+sticker+'.pt'
        self.FPATH = 'ft'+prd0+'_'+sticker+'.pt'
        
        NCF=[0.8] 
        HT=[2]
        FP='optparam_'+self.name+'_'+sticker+'.txt'
        
        dfl=self.formFeatures(sticker,timestamp,0.3,0.018) #optcoef=0.018
        
        if dfl is None:
            return (100,(0,0),0.0)
        (df,label)=dfl
        TCF=[0.018] #0.018
            
        l=len(df)-1
        df['label']=label
        X = df.iloc[:,0:-1]
        y = df.iloc[:,-1]
        
        X_train_1, X_test_1, y_train_1, y_test_1= X[:l], X[l:], y[:l],y[l:]  
        select_X_train = X_train_1 
        
        if 'brf' in self.name or 'eec' in self.name or 'tsf' in self.name or 'ada' in self.name or 'tsb' in self.name or 'gbf' in self.name:
            LRS=self.adapt_dt_prune(df)
            opt=self.adapt_objective_1(df,sticker,timestamp,LRS)
            (opta,optsr,optlr,optgr,optk)=opt
        else:
            opta=1.0
            optsr=1.0
            optlr=0.001
            optgr=1.0
            optk=0.5
        
                 
        model = None
        model_leader = None
        optaN=None
        optcr=0.01
        try:
          
            
            if 'rus' in self.name:
                model_leader = featurePredictorS.customPredictors.rusCCPClassifier(optlr,X_train_1,y_train_1,optgr,optk)
                #RUSBoostClassifier(n_estimators=1000,learning_rate=optlr,random_state=874437)
            else:
                if 'xgb' in self.name:
                    model_leader=HistGradientBoostingClassifier(max_iter=featurePredictorS.TRSIZE,learning_rate=optlr,max_depth=optht,random_state=7643,l2_regularization=2.0,early_stopping=False,n_iter_no_change=1000,verbose=0)
                else:
                    #model_leader = xgboost.XGBClassifier(max_depth=optht,verbosity=0,silent=0,n_estimators=1000,learning_rate=optlr,objective='binary:logistic',reg_lambda=optlmbd,reg_alpha=0,seed=165228,tree_method='exact')
                    if 'ngb' in self.name:
                        #bs=DecisionTreeClassifier(criterion="log_loss",min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_depth=2,splitter="best",random_state=None)
                        bs=DecisionTreeRegressor(criterion="friedman_mse",min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.0,max_depth=optht,splitter="best",random_state=74433)
                        model_leader = NGBClassifier(Dist=k_categorical(2),Score=LogScore,Base=bs,natural_gradient=True,n_estimators=featurePredictorS.TRSIZE,random_state=58645,verbose=False,learning_rate=optlr)
                    else:
                        if 'log' in self.name:
                            model_leader = LogitBoost(n_estimators=featurePredictorS.TRSIZE,learning_rate=optlr,random_state=7643,bootstrap=True)
                        else:
                            if 'eec' in self.name:
                                model_leader=featurePredictorS.customPredictors.rffCCPClassifier(optlr,X_train_1,y_train_1,optgr,optk)
                            else:
                                if 'tsf' in self.name:
                                    #model_leader = ExtraTreesClassifier(random_state=58645,ccp_alpha=optlr,n_estimators=featurePredictorS.TRSIZE,bootstrap=self.bootstrap,criterion=self.treecrit)
                                    model_leader = featurePredictorS.customPredictors.extrCCPClassifier(optlr,X_train_1,y_train_1,optgr,optk)
                                else:
                                    if 'svc' in self.name:
                                        model_leader = SVC(kernel='rbf',C=optlr)
                                    else:
                                        if 'tsb' in self.name:
                                            model_leader=featurePredictorS.customPredictors.ensCCPClassifier(optlr,X_train,y_train)
                                            #model_leader = TimeSeriesForest(n_estimators=featurePredictorS.TRSIZE,random_state=65433,ccp_alpha=optlr,bootstrap=False)
                                        else:
                                            if 'brf' in self.name:
                                                model_leader = featurePredictorS.customPredictors.bgCCPClassifier(optlr,X_train_1,y_train_1,optgr,optk)
                                            else:
                                                if 'gbf' in self.name:
                                                    model_leader = featurePredictorS.customPredictors.gbCCPClassifier(optlr,X_train_1,y_train_1,optgr,optk)
                                                else:
                                                    model_leader = featurePredictorS.customPredictors.adaCCPClassifier(optlr,X_train_1,y_train_1,optgr,optk)
                                            #model_leader = TSBF(n_estimators=1000,random_state=65433,ccp_alpha=optlr,bootstrap=False)
            
        except:
            traceback.print_exc()
            print('OOOPPP ' + sticker + ' ' + timestamp)
            return (100,(0,0),0.0)
        
        
        try:
             
            if model_leader is None:
                return (100,(0,0),0.0)
            
        
            if optsr is None or opta is None:
                return (100,(0,0),0.0)
         
            #print('Let us predict')
            X = df.iloc[:,0:-1]
            y = df.iloc[:,-1]
            X_test_1=X[l:]        
             
            
            L=len(X_train_1)
              
            raw_up=optsr[-1]
            sr_d=optsr[-2]
            sr_u=optsr[-3]
            sr = optsr[-4]
            longsr=opta[0]

            
#             try:
#                 fk=open('ltau_'+sticker+'.txt','rt')
#                 optlt=fk.read()
#                 fk.close()
#             except:
#                 optlt=self.LTAU
#             
#             try:
#                 fk=open('rtau_'+sticker+'.txt','rt')
#                 optrt=fk.read()
#                 fk.close()
#             except:
#                 optrt=self.RTAU
            
    
            SRM=0.75
            if 'gbf' in self.name: #'eecS' in self.name or 'tsfS' in self.name: 0.9
                SRM=0.6
#             if 'brf' in self.name or 'ada' in self.name or 'eec' in self.name:
#                 SRM=0.65
            
            crit = longsr>SRM #and raw_up>0.6
#             crit_up=False
#             crit_down=False
#             if self.BERN==True:
#                 crit_up = raw_up>=self.PIVOT and sr_u>optlt*sr
#                 crit_down = 1-raw_up>self.PIVOT and sr_d>optrt*sr
#                 crit = crit and (crit_up or crit_down)
            
            if crit==False:
                return (100,(0,0),0.0)

            model_leader.fit(X_train_1,y_train_1)
            
#             w = model_leader.predict(X_test_1)
#             p = max(0,w[0])

            w = model_leader.predict_proba(X_test_1)  
            W=w[0]  
            p = numpy.argmax(W)
            
            CF=0.65 #0.58
            if 'gbf' in self.name:
                CF=0.56
            
#             if 'brf' in self.name or 'ada' in self.name or 'eec' in self.name:
#                 CF=0.65
            
            if 'eecS' in self.name or 'tsfS' in self.name:
                CF=1.0
                
            if W[p]<CF:
                return (100,(0,0),0.0)
                
            del model_leader
            model_leader = None
            
            if os.path.exists('successrates.csv')==False:
                with open('successrates.csv','wt') as fl:
                    fl.write('Date,Ticker,Predictor,Prediction,SuccRate,SuccProbability,RawUp,Sign\n')
                    fl.close()
            else:
                rl = self.dataManager.globalDatasource.getDelta(sticker, 'close', timestamp)
                if p<=0:
                    p=-1
                g=int(numpy.sign(p*rl))
                with open('successrates.csv','at') as fl:
                    fl.write(timestamp+','+sticker+','+self.name+','+str(p)+','+str(longsr)+','+str(round(W[p],2))+','+str(round(raw_up,2))+','+str(g)+'\n')
                    fl.close()
   
        
            return (p+0.51,(1,0),1.0)
        except:
            traceback.print_exc()
            return (100,(0,0),0.0)


# alpha L1 - Lasso shrinks the less important feature coefficient to zero. So, this works well for feature selection in case we have a huge number of features.
# lambda L2 - higher is underfit, lower is overfit
# #l2 is lambda, l1 is alpha
# https://www.linkedin.com/pulse/time-series-classification-model-based-transformer-gokmen/
# https://medium.com/cmotions/hyperparameter-tuning-for-hyperaccurate-xgboost-model-d6e6b8650a11
# https://www.capitalone.com/tech/machine-learning/how-to-control-your-xgboost-model/
