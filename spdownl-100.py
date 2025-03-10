import pandas
import sys
import glob
from datetime import date,timedelta
from dateutil import parser
import os
import yfinance as yf
from dateutil import parser
import shutil
from Service.Utilities import Tools
from Configuration import Configuration

def dateparse(d):    
    return parser.parse(d)

dirname='sandprecent'
command_=sys.argv[1] 

if command_=='exact':

    if os.path.exists(dirname)==False:
        sys.exit(0)   
    
    dirnameE=dirname+'E'
    
    if os.path.exists(dirnameE)==True: 
        shutil.rmtree(dirnameE)
    os.mkdir(dirnameE)                   
        
    sts=[]
    lastts = None
    firstts = None
    
    datas={}
    timestamps=set([])
    
    dr = os.path.join(os.getcwd(),dirname)
    for file in os.listdir(dr):
        stickername = file.split('.')[0]
        sts.append(stickername)
        
        stf = os.path.join(os.getcwd(),dirname,stickername+'.csv')
        fd=open(stf,'rt')
        lll=fd.readlines()
        fd.close()
        
        curtsL = lll[-1].split(' ')[0]
        curtsF = lll[0].split(' ')[0]
        
        if lastts is None:
            lastts=curtsL
        
        if firstts is None:
            firstts=curtsF
        
        if pandas.Timestamp(curtsL)>pandas.Timestamp(lastts):
            lastts=curtsL
        
        if pandas.Timestamp(curtsF)<pandas.Timestamp(firstts):
            firstts=curtsF
            
    startd=pandas.Timestamp(firstts).strftime('%Y-%m-%d')
    plts = pandas.Timestamp(lastts)
    tds = pandas.Timestamp.today().strftime('%Y-%m-%d')
    if plts.strftime('%Y-%m-%d')!=tds:
        plts = pandas.Timestamp(Tools.nextBusDay(plts,1))
    endd=plts.strftime('%Y-%m-%d')

    for st in sts:
        
        print(st)
        datlast=None
        
        tns = os.path.join(os.getcwd(), dirnameE, st + '.csv')
        data = yf.download(st, start=startd, end=endd)

        data=data[['Open', 'High', 'Low', 'Close', 'Volume']]
        timestamps = timestamps.union(set(list(data.index)))
        datas[st]=data
        
    timestamps=sorted(list(timestamps))
    
    for k in datas.keys():
        data=datas[k]
        tns = os.path.join(os.getcwd(), dirnameE, k + '.csv')
        data = data.reindex(timestamps)
        if len(timestamps)>len(datas[k]):
            for c in data.columns:
                data[c] = data[c].fillna(method='ffill').fillna(method='bfill')
        
        data.to_csv(tns,header=False,sep = ' ',date_format='%Y%m%d')
        

if command_=='last':

    if os.path.exists(dirname)==False:
        sys.exit(0)   
        
    sts=[]
    lastts = None
    dr = os.path.join(os.getcwd(),dirname)
    for file in os.listdir(dr):
        stickername = file.split('.')[0]
        sts.append(stickername)
        
        stf = os.path.join(os.getcwd(),dirname,stickername+'.csv')
        fd=open(stf,'rt')
        lll=fd.readlines()
        fd.close()
        
        curts = lll[-1].split(' ')[0]
        if lastts is None:
            lastts=curts
        if pandas.Timestamp(curts)>pandas.Timestamp(lastts):
            lastts=curts
        
    plts = pandas.Timestamp(lastts)
    tds = pandas.Timestamp.today().strftime('%Y-%m-%d')
    if plts.strftime('%Y-%m-%d')!=tds:
        plts = pandas.Timestamp(Tools.nextBusDay(plts,2))
    endd=plts.strftime('%Y-%m-%d') 
    startd=(pandas.Timestamp(Tools.nextBusDay(plts,-2))).strftime('%Y-%m-%d')
    
    for st in sts:
        
        print(st)
        datlast=None
        try:
            datq=yf.download(st, start=startd, end=endd,interval='30m')
            datlast = datq.iloc[-2]
            
        except:
            datq=yf.download(st, start=startd, end=endd)
            if len(datq)>0:
                datlast = datq.iloc[-1]
            
        if datlast is None or len(datlast)==0:
            continue
        
        ldd=[datlast['Open'],datlast['High'],datlast['Low'],datlast['Close'],datlast['Volume']]
        ldstamp = datlast.name.strftime('%Y%m%d')
        
        tns = os.path.join(os.getcwd(), dirname, st + '.csv')
        if os.path.exists(tns): #False
            appndx = [ldstamp]+[str(it.values[0]) for it in ldd]
            with open(tns, "a+") as file:
                file.write(' '.join(appndx)+"\n")

        
if command_=='hist':

    dataset=sys.argv[2]
    if sys.argv[3]=='now':
        enddateT=pandas.Timestamp.today()  
    else:
        enddateT=pandas.Timestamp(sys.argv[3])
    
    if len(sys.argv)>4:
        H=int(sys.argv[4])
    else:
        H=Configuration.seloffset+2
    
    tname = dataset+'.csv'

    df=pandas.read_csv(os.path.join(os.getcwd(),tname),header=None)
    sts = df.values
        
    startdateT=enddateT-pandas.Timedelta(days=H)
    
    startdate=startdateT.strftime('%Y-%m-%d')
    enddate = enddateT.strftime('%Y-%m-%d')

    if os.path.exists(dirname)==True: 
        shutil.rmtree(dirname)
    os.mkdir(dirname)

    
    datas={}
    timestamps=set([])
    for st in sts:
        sticker=st[0]
        sticker=sticker.replace('\t','')
        tns = os.path.join(os.getcwd(), dirname, sticker + '.csv')
        data = yf.download(sticker, start=startdate, end=enddate)
#         if len(data)<H+V-5: #250: #167
#             continue
        data=data[['Open', 'High', 'Low', 'Close', 'Volume']]
        df = df[~df.index.duplicated(keep='last')]
        timestamps = timestamps.union(set(list(data.index)))
        datas[sticker]=data
    
    timestamps=sorted(list(timestamps))
    
    for k in datas.keys():
        data=datas[k]
        tns = os.path.join(os.getcwd(), dirname, k + '.csv')
        data = data.reindex(timestamps)
        if len(timestamps)>len(datas[k]):
            for c in data.columns:
                data[c] = data[c].fillna(method='ffill').fillna(method='bfill')
                
        #data = data[-lnd:]
        
        data.to_csv(tns,header=False,sep = ' ',date_format='%Y%m%d')
    


