import datetime
import numpy
import calendar
from itertools import chain, combinations
#from Service.econometry import max_dd
#from Service.huffman import HuffmanCoding
import pandas
import random
#import requests
import os
from Configuration import Configuration
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import pandas_market_calendars as mcal
import pprint
from sklearn import preprocessing 
from datetime import datetime,timedelta



class Tools(object):
    
    #ONE_DAY = datetime.timedelta(days=1)
    #nyse_holidays = mcal.get_calendar('NYSE').holidays()
    RESERVED = ['2021-04-02']
    STATEOFFS=0
    date_format='%d-%b-%y'
    DEFVALS={'MINSR':0.8,'SIGMA':2.9}

    daynames=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    #year='2021'
    #nyseholidays={'newyear':'01-01','newyear1':'01-02','luther':'01-18','washington':'02-15','goodfriday':'04-02','memorial':'05-31','juneteenth:':'06-20','independence':'07-04','labor':'09-06','thanksgiving':'11-25','christmas':'12-24'}
    
    year='2023'
    nyseholidays={'luther':'01-17','washington':'02-21','goodfriday':'04-15','memorial':'05-30','juneteenth:':'06-20','independence':'07-04',
                  'labor':'09-05','thanksgiving':'11-24','christmas':'12-24'}
    
    
    lastTimestamps={}
    
    SCOPES = ['https://mail.google.com/']
    TO='shlomidolev@gmail.com'
    #BCC=['shlomidolev@gmail.com','binunalex@gmail.com']
    BCC=['assaf.lansky@gmail.com','yona.hollander@gmail.com','elev63@gmail.com','Chen.munitz@gmail.com','shlomidolev@gmail.com','binunalex@gmail.com','hagar.dolev@gmail.com','yoraid@gmail.com']
    
    logMode=False
    cause=None
    
    col_for_x = ['mix_mv_avg','5_price_diff','mv_avg_diff', 'avg_quantity','quantity_price','ct_rising','aux_flag', 'aux_flagD','price']

    WSIZE = 50
    VOLWINDOW=6
    DEVICE=None
    
#     @staticmethod
#     def adaptGPU():
#         gpuname=tf.test.gpu_device_name()
#         if len(gpuname)>1 and ':' in gpuname and 'gpu' in gpuname.lower():
#             cmps = gpuname.split(':')
#             config = tf.ConfigProto(device_count = {'GPU': int(cmps[2])})
#             print('Running on GPU ' + gpuname)
#         else:
#             config = tf.ConfigProto()
#         return (gpuname,config)
    
    
    @staticmethod
    def getIniValue(k,tp='f'):
        if os.path.exists('initvalues.ini')==False:
            try:
                return Tools.DEFVALS[k]
            except:
                return None
        else:
            f=open('initvalues.ini','rt')
            rl=f.readlines()
            f.close()
            for l in rl:
                if '=' not in l:
                    continue
                l=l.replace('\n','')
                [ky,v]=l.split('=')
                if k==ky:
                    if tp=='f':
                        return float(v)
                    else:
                        return v
                    
            try:
                return Tools.DEFVALS[k]
            except:
                return None
            
        
    @staticmethod
    def get_last_ts(filename='sandprecent_joint_maillog.csv'):
        try:
            with open(filename,'rt') as f:
                lns=f.readlines()
                if len(lns)<1:
                    return None
                comps1=lns[-1].split(',')
                if len(comps1)<=1:
                    comps1=lns[-2].split(',')
                ts1=pandas.Timestamp(comps1[0])
                return ts1
        except:
            return None
    
#     @staticmethod
#     def get_last_ts(filename='sandprecent_joint_maillog.csv'):
#         try:
#             with open(filename,'rt') as f:
#                 lns=f.readlines()
#                 if len(lns)<1:
#                     return None
#                 
#                 ts1=None
#                 ts2=None
#                 comps1=lns[-1].split(',')
#                 ts1=pandas.Timestamp(comps1[0])
#                 if len(lns)>=2:
#                     comps2=lns[-2].split(',')
#                     ts2=pandas.Timestamp(comps2[0])
#                     return (ts1,ts2)
#                 else:
#                     return (ts1,None)
#         except:
#             return None
            
    
    @staticmethod
    def get_rsi(close, lookback):
        ret = close.diff()
        up = []
        down = []
        for i in range(len(ret)):
            if ret[i] < 0:
                up.append(0)
                down.append(ret[i])
            else:
                up.append(ret[i])
                down.append(0)
        up_series = pandas.Series(up)
        down_series = pandas.Series(down).abs()
        up_ewm = up_series.ewm(com = lookback - 1, adjust = False).mean()
        down_ewm = down_series.ewm(com = lookback - 1, adjust = False).mean()
        rs = up_ewm/down_ewm
        rsi = 100 - (100 / (1 + rs))
        rsi_df = pandas.DataFrame(rsi).rename(columns = {0:'rsi'}).set_index(close.index)
        rsi_df = rsi_df.dropna()
        return rsi_df[3:]
    
    @staticmethod
    def sma(data, lookback):
        sm = data.rolling(lookback).mean()
        return sm
    
    @staticmethod
    def get_bb(data, lookback):
        std = data.rolling(lookback).std()
        upper_bb = Tools.sma(data, lookback) + std * 2
        lower_bb = Tools.sma(data, lookback) - std * 2
        middle_bb = Tools.sma(data, lookback)
        return upper_bb, middle_bb, lower_bb
    
    @staticmethod
    def create_feature_label(df):
    
        dfprice = df['Close']
        df['price'] = dfprice
        c=df['Close'].values
        y=[-1]
        for i in range(0,len(df)-1):
            if c[i+1]>c[i]:
                y.append(1)
            else:
                y.append(-1)
            
        dfquantity = df['Volume']
        
        df["mv_avg_3"] = dfprice.rolling(3,min_periods=1).mean()
        df["mv_avg_6"] = dfprice.rolling(6,min_periods=1).mean()
        df["mv_avg_12"] = dfprice.rolling(12,min_periods=1).mean()
        df["mv_avg_24"] = dfprice.rolling(24,min_periods=1).mean()
        df["mix_mv_avg"] = (df.mv_avg_3 + df.mv_avg_6 + df.mv_avg_12 + df.mv_avg_24)/4
        df['mv_avg_diff'] = df.mv_avg_3-df.mv_avg_6
            
        df['avg_quantity'] = dfquantity.rolling(5, min_periods = 1).mean()
        df['quantity_price'] = dfquantity / dfprice
        
        df['price_diff'] = dfprice.diff()
        df['5_price_diff'] = dfprice.diff(periods = 4)
           
        df['pos'] = [1 if x>0 else 0 for x in df['price_diff'].shift(-1)]
        df['ct_rising'] = df.pos.rolling(10, min_periods = 1).sum()
        
        df['aux_flag'] = dfprice.pct_change().rolling(Tools.VOLWINDOW).mean()
        df['aux_flagD']= dfprice.pct_change().rolling(Tools.VOLWINDOW).std()
        
        
        df['price_diff'] = df['price_diff'].bfill()
        df['5_price_diff'] = df['5_price_diff'].bfill()
        
        df['label'] = y
        #df["label"] = [1 if x > threshold else -1 if x < -threshold else 0 for x in df["perc"]] 
           
        return df
    
    @staticmethod
    def reshape(df,winsize=50):
        df_as_array=numpy.array(df)
        temp = numpy.array([numpy.arange(i-winsize,i) for i in range(winsize,df.shape[0])])
        new_df = df_as_array[temp[0:len(temp)]]
        new_df2 = new_df.reshape(len(temp),len(Tools.col_for_x)*winsize)
        return new_df2   
     
    @staticmethod
    def genoutput(trade_data):
        
        test_size = Tools.WSIZE+1
        train_size = trade_data.shape[0]-test_size
        
        
        trade_data_featured = Tools.create_feature_label(trade_data)
        
        train_2 = trade_data_featured.head(train_size)
        test_2 = trade_data_featured.tail(test_size)
        
        X_train = train_2[Tools.col_for_x]
        y_train = train_2['label']
        X_test = test_2[Tools.col_for_x]
        y_test = test_2['label']
        
        
        X_train_scaled = pandas.DataFrame(preprocessing.scale(X_train))
        X_train_scaled.columns = Tools.col_for_x
        
        X_test_scaled = pandas.DataFrame(preprocessing.scale(X_test))
        X_test_scaled.columns = Tools.col_for_x
        y_train.columns = 'label'
        y_test.columns = 'label'
#         
#         X_train_reshaped = Tools.reshape(X_train_scaled)
#         X_test_reshaped = Tools.reshape(X_test_scaled)
        
        X_train_tocsv = X_train_scaled
        X_train_tocsv['label'] = numpy.array(y_train)
        X_test_tocsv = X_test_scaled
        X_test_tocsv['label'] = numpy.array(y_test)
        
        
        X_train_tocsv = X_train_tocsv.iloc[Tools.VOLWINDOW:,:]
        
    
        return (X_train_tocsv,X_test_tocsv)
    
#     @staticmethod
#     def logCall(sender,args,logname='files.csv'):
#         
#         if args[0]=='all' and args[2]=='all' and '_' in sender.datasource:
#             if Tools.cause is None:
#                 Tools.cause = args[1]
#         
#         if Tools.logMode==False:
#             return
#         
#         st=type(sender).__name__
#         try:
#             if '_' not in sender.datasource:
#                 st=st+'-global'
#         except:
#             pass
#         
#         if os.path.exists(logname)==False:
#             s=open(logname,'wt')
#             s.write('Invoker,TargetDate,Sender,Ticker,AccessDate,Column,Depth\n')
#             s.close()
#         
#         previous_frame = inspect.currentframe().f_back
#         dfetch_caller = previous_frame.f_back
#         (filename, line_number,function_name, lines, index)=inspect.getframeinfo(dfetch_caller)
#         invoker=filename+":"+str(line_number)
#         
#         if Tools.cause is not None and args[0]!='all' and args[2]!='all':
#             l=open(logname,'at')
#             st1=','.join([invoker,Tools.cause,st]+list(map(lambda x:str(x),args)))
#             l.write(st1+'\n')
#             l.close()
        
#     @staticmethod
#     def create_mail_message(sender, to, bcc,subject, message_text):
#         message = MIMEMultipart()
#         message['to'] = to
#         message['subject'] = subject
#         message['from'] = sender
#         message['Bcc'] = ", ".join(bcc)
#         
#         #message.attach(MIMEText(message_text, 'plain'))
#  
#         attachment='last-positions.csv'
#         content_type, encoding = mimetypes.guess_type(attachment)
#         main_type, sub_type = content_type.split('/', 1)
#  
#         f = open(attachment, 'rb')
#  
#         myFile = MIMEBase(main_type, sub_type)
#         myFile.set_payload(f.read())
#         myFile.add_header('Content-Disposition', 'attachment', filename=attachment)
#         encoders.encode_base64(myFile)
#         f.close()
#  
#         message.attach(myFile)
#         #return {'raw': base64.urlsafe_b64encode(message.as_string().encode()).decode()}
#         return {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}
# 
#     
#     @staticmethod
#     def send_mail_message(service, user_id, message):
#         try:
#             message = (service.users().messages().send(userId=user_id, body=message).execute())
#             print('Message Id: %s' % message['id'])
#             return message
#         except Exception as error:
#             print(error)
#             
#     @staticmethod
#     def mailPredictions(body):
#         flow = InstalledAppFlow.from_client_secrets_file('credentials.json', Tools.SCOPES)
#         creds = flow.run_local_server(port=0)
#         service = build('gmail', 'v1', credentials=creds)
#         message = Tools.create_mail_message('me', Tools.TO, Tools.BCC,'PREDICTION', body)
#         Tools.send_mail_message(service=service, user_id='me', message=message)
    
    @staticmethod
    def GridSearch_table_plot(grid_clf, param_name,negative=True,graph=True):
        clf = grid_clf.best_estimator_
        clf_params = grid_clf.best_params_
        if negative:
            clf_score = -grid_clf.best_score_
        else:
            clf_score = grid_clf.best_score_
        clf_stdev = grid_clf.cv_results_['std_test_score'][grid_clf.best_index_]
        cv_results = grid_clf.cv_results_
        print("best parameters: {}".format(clf_params))
        print("best score:      {:0.5f} (+/-{:0.5f})".format(clf_score, clf_stdev))
        
        pprint.pprint(clf.get_params())
        
        scores_df = pandas.DataFrame(cv_results).sort_values(by='rank_test_score')

        best_row = scores_df.iloc[0, :]
        scores_df = scores_df.sort_values(by='param_' + param_name)

        if negative:
            means = -scores_df['mean_test_score']
        else:
            means = scores_df['mean_test_score']
        stds = scores_df['std_test_score']
        params = scores_df['param_' + param_name]

#     @staticmethod
#     def get_spy():
#         url = 'https://www.slickcharts.com/sp500'
#         request = requests.get(url,headers={'User-Agent': 'Mozilla/5.0'})
#         soup = None #BeautifulSoup(request.text, "lxml")
#         stats = soup.find('table',class_='table table-hover table-borderless table-sm')
#         df = pandas.read_html(str(stats))[0]
#         df['% Chg'] = df['% Chg'].str.strip('()-%')
#         df['% Chg'] = pandas.to_numeric(df['% Chg'])
#         df['Chg'] = pandas.to_numeric(df['Chg'])
#         return df
    
    @staticmethod
    def stlists_(s,replacers):
        spl=s.index('SHORT:')
        
        st_longs = s[len('LONG:'):spl]
        st_shorts = s[spl+len('SHORT:'):]
        
        st_longs = st_longs.replace('{','').replace('}','').replace('_',' ')
        st_shorts = st_shorts.replace('{','').replace('}','').replace('_',' ')
        
        st_longs_l = st_longs.split(' ')
        st_shorts_s = st_shorts.split(' ')
        
        lrpl = random.sample(st_longs_l, 3)
        lrps = random.sample(st_shorts_s, 3)
        
        candidates = set(replacers) - set(lrpl)
        candidates = candidates - set(lrps)
        
        for lrp in lrpl:
            lrp_n = random.sample(candidates, 1)
            st_longs_l = [lrp_n[0] if x==lrp else x for x in st_longs_l]
        
        for srp in lrps:
            srp_n = random.sample(candidates, 1)
            st_shorts_s = [srp_n[0] if x==srp else x for x in st_shorts_s]
        
        
        evt = 'LONG:{'+'_'.join(st_longs_l)+ '} ' + 'SHORT:{'+'_'.join(st_shorts_s)+'}'
        return evt
    
    @staticmethod
    def scale(arr,tmin=1,tmax=5):
        rmin=min(arr)
        rmax=max(arr)
        if rmax==rmin:
            return None
        res=[]
        for m in arr:
            memb = (m-rmin)*(tmax-tmin)
            memb = memb/(rmax-rmin)
            memb = memb+tmin
            res.append(memb)
        return res
            
    # -----------------------------------------------------------------------
    @staticmethod
    def arrGrowth1(a): 
        return [int(numpy.sign(x)) for x in numpy.diff(a)]
    
    @staticmethod
    def growth1(h):
        return int(numpy.sign(h))
    
    @staticmethod
    def invert(d):
        if d>1:
            return d-1.0
        else:
            return 2.0-d
    
    @staticmethod
    def delta1(a,v):
    
        if isinstance(v,list)==False:
            v1=[v]*len(a)
        else:
            v1=v
        return [a[i]-v1[i] for i in range(0,len(a))]
    
    @staticmethod
    def hit1(p,v):
        return p*v>0  
    
    @staticmethod
    def rel1(h,p):
        if p==0:
            return 0
        return float((h-p)/p)
    
    @staticmethod
    def rels1(hst):
        if len(hst)==0:
            return []
        mx = max(hst)
        if numpy.fabs(mx)<0.000001:
            return []
        histdiff=[]
        for i in range(0,len(hst)):
            histdiff.append(float(hst[i]/mx))
        return histdiff
    # --------------------------------------------------------------------
    
    @staticmethod
    def arrGrowth(a): 
        h=[]
        for i in range(1,len(a)):
            if a[i-1]>0:
                h.append(float(a[i]/a[i-1]))
            else:
                h.append(0.0)
        hl=[0.0]+h
        return [int(numpy.sign(x-1.0)) for x in hl]
    
    
    @staticmethod
    def growth(h): 
        if numpy.fabs(h)>0:
            return int(numpy.sign(h-1.0))
        else:
            return 0
    
    
    @staticmethod
    def hit(p,v):
        return Tools.growth(p)*Tools.growth(v)>0
    
    
    @staticmethod
    def delta(a,v):
        return a
    
    
    @staticmethod
    def rel(h,p):
        try:
            return float(h/p)
        except:
            return 0
            

    
    @staticmethod
    def rels(hst):
        if len(hst)==0:
            return []
        histdiff=[]
        for i in range(1,len(hst)):
            if hst[i-1]>0:
                histdiff.append(float(hst[i]/hst[i-1]))
            else:
                histdiff.append(0.0)
        return [0.0]+histdiff
    
    
    @staticmethod
    def memmove(column,target,startt,source):
        for i in range(0,len(source)):
            if startt+i>=len(target[column]):
                break
            target[column][startt+i] = source[i]
    
    
    @staticmethod
    def parseJointEntry(sd):      
        sd=sd.replace("_"," ")
        #sd=sd.replace("-"," ")
        sd=sd.replace("{","")
        sd=sd.replace("}","")
        #sd=sd.replace("-"," ")
        sd=sd.replace(":","")
        sd=sd.replace(";","")
        lpos = sd.find("LONG")
        spos = sd.find("SHORT")
        
        
        longstr = sd[lpos+4:spos]
        shortstr = sd[spos+5:]
        
        longl=[]
        shortl=[]
        
        if len(longstr)>1:
            longl=longstr.split(" ")
        if len(shortstr)>1:
            shortl = shortstr.split(" ")
        
        both=set(longl) & set(shortl)
        
        for b in both:
            longl.remove(b)
            shortl.remove(b)
        
        longl = list(filter(lambda e:len(e)>0,longl))
        shortl = list(filter(lambda e:len(e)>0,shortl))
        
        return (longl,shortl)
                
    
    @staticmethod
    def parseEntry(sd):      
        sd=sd.replace("_"," ")
        sd=sd.replace("{","")
        sd=sd.replace("}","")
        lpos = sd.find("LONG:")
        spos = sd.find("SHORT:")
        
        longstr = sd[lpos+5:spos]
        shortstr = sd[spos+6:]
        
        longl=[]
        shortl=[]
        
        if len(longstr)>1:
            longl=longstr.split(" ")
        if len(shortstr)>1:
            shortl = shortstr.split(" ")
        
        both=set(longl) & set(shortl)
        
        for b in both:
            longl.remove(b)
            shortl.remove(b)
        
        longm = list(filter(lambda e:len(e)>0,longl))
        shortm = list(filter(lambda e:len(e)>0,shortl))
        
        return (longm,shortm)
        
            
    @staticmethod
    def max_dd(X):
        mdd = 0
        peak = X[0]
        for x in X:
            if x > peak: 
                peak = x
            
            dd = peak - x
        
            if dd > mdd:
                mdd = dd
        return mdd 
    
    @staticmethod
    def nextBusHour(timestamp):
        dn=pandas.Timestamp(timestamp)+timedelta(minutes=Configuration.TIMESTEP)
        return dn
    
    @staticmethod
    def nextBusDay(timestamp,dr=1):
        if isinstance(timestamp, str):
            nowday = timestamp
        else:
            nowday = timestamp.strftime('%Y-%m-%d')
            
        y=nowday[:4]
        nyse = mcal.get_calendar('NYSE')
        early = nyse.schedule(start_date=y+'-01-01', end_date=y+'-12-31')
        workdays = mcal.date_range(early, frequency='1D')
        wdtext = [td.strftime('%Y-%m-%d') for td in workdays] 
        if nowday not in wdtext:
            while True:
                nowday=(pandas.Timestamp(nowday)-pandas.Timedelta("1 days")).strftime('%Y-%m-%d')
                if nowday in wdtext:
                    break
        try:
            tsi=wdtext.index(nowday)+1*dr
            tsi = min(tsi,len(wdtext)-1)
        except:
            tsi=0
        return wdtext[tsi]
    
        
    
    @staticmethod
    def cancelNan(x):
        if numpy.isnan(x) or numpy.isinf(x):
            return 0
        else:
            return x
    
    @staticmethod
    def today():
        d = datetime.datetime.today()
        dt64 = numpy.datetime64(d)
        ts = (dt64 - numpy.datetime64('1970-01-01T00:00:00Z')) / numpy.timedelta64(1,'s')
        return ts
       
        
#     @staticmethod  
#     def metrics(returns):
#         sharpe = numpy.mean(returns)/numpy.std(returns)
#         mdd = max_dd(returns)
#         stdv = numpy.std(returns)
#         calmar = (returns[-1]-returns[0])/mdd
#         
#         deltas=numpy.diff(returns)
#         pos_returns = list(filter(lambda v: v>0, deltas))
#         scc = float(len(pos_returns)/len(deltas))
#         
#         return (sharpe,mdd,stdv,calmar,scc)   
    
    @staticmethod
    def powerset(iterable):
        xs = list(iterable)
        return chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1))
    
    @staticmethod
    def power(iterable,initlen,maxlen,exact=False):
        l = list(Tools.powerset(iterable))
        if not exact:
            res = list(filter(lambda p: len(p)>=initlen and len(p)<=maxlen, l))
        else:
            res = list(filter(lambda p: len(p)==maxlen, l))
        del l
        return res
            
    
    @staticmethod
    def RSI(dataset,n=9):
        up_values = []
        down_values = []
        x = 0
        while x < n-1:
            difference = dataset[x+1] - dataset[x]
            if difference < 0:
                down_values.append(abs(difference))
            else:
                up_values.append(difference)
            x = x + 1
            
        avg_up_closes = 0
        avg_down_closes = 0 
        relative_strength = 0
        
        if len(up_values)>0:
            avg_up_closes = sum(up_values)/len(up_values)
        if len(down_values)>0:
            avg_down_closes = sum(down_values)/len(down_values)
        if avg_down_closes>0:
            relative_strength = avg_up_closes/avg_down_closes

        rsi = 100 - (100/(1 + relative_strength))
        return rsi
    
    
    @staticmethod
    def intersect(l1,l2):
        r=[]
        for e in l1:
            if e in l2:
                r.append(e)
        return r
    
    @staticmethod
    def weekday(timestamp):
        components = timestamp.split('-')
        year = int(components[0])
        if components[1] in Tools.months:
            month = Tools.months.index(components[1])+1
        else:
            month = int(components[1])
        
        day = int(components[2])
        d = calendar.weekday(year,month,day)
        return (d,Tools.daynames[d])
    
    @staticmethod
    def numberToBase(n, b):
        if n==0:
            return [0]
        digits = []
        while n:
            digits.append(int(n % b))
            n //= b
        return digits[::-1]
    
    @staticmethod
    def numberToString(n, b):
        digs = Tools.numberToBase(n, b)
        conv={0:'M',1:'H',2:'S'}
        sr = ''
        for d in digs:
            sr=sr+conv[d]
        return sr
    
            
    def __init__(self, params):
        pass
        