
config=None

# HIkkake - appears 4 rows after beginning

# https://medium.com/@chris_42047/the-hidden-meaning-of-candlesticks-python-tutorial-f05ec183415d
# https://medium.com/@matthew1992/xgboost-for-stock-trend-prices-prediction-ac97c52a9733


# API: api-key-1682860922416
# pub: IW8eBNscg90oux5e21I940Z9gk6s711iO0grKApVmtEjFJfuXAXfqfgc
# pri: iOAhWj+SvmP8bczrERagla9pXHY/aT/uVlShZaowKLGnKbKRNNgVB+0QTehSVtrUmaHUtQbal0OpoFvO1OwSmA==


import json
import os
class Configuration:
    
    QUALITY = 'close'
    DATADIR=''
    TAUS={}
    
    TIMESTEP=60 # minutes, only when HOURLY is TRUE
    RESOLUTIONS={15:'15MINS',30:'30MINS',60:'1HOUR'}
    RESOLUTION='60MINS'
    TIMEOFFSET=0
    
    
    RETRAIN=True
    
    STATEOFFS=0 
    ALGS=['ada','brf','gbf'] 
    RETRL=1
    
    adaptEngine={'RandomForestClassifier':'neg_log_loss','RUSBoostClassifier':'neg_log_loss','GNEGNEClassifier':'neg_log_loss',
                 'NGBClassifier':'neg_log_loss','BoostingClassifier':'neg_log_loss','MLPClassifier':'precision','XGBClassifier':'neg_log_loss',
                 'LGBMClassifier':'neg_log_loss','AdaBoostClassifier':'neg_log_loss','GradientBoostingClassifier':'neg_log_loss',
                 'HistGradientBoostingClassifier':'neg_log_loss'}

    
    characteristics=['open','high','low','close','volume']
    mailprefix="_maillog.csv"
    paramGrids = {'SGDRegressor':{'alpha': [0.001,0.01,0.05,0.1]},
                  'KNeighbors':{'n_neighbors': [3,5,10,12]},
                  'SVR':{'C': [0.01,1, 100],'epsilon': [0.0001, 0.001,0.01,0.1, 1, 5, 10],'degree': [4],'coef0': [0.01]},
                  'BoostingClassifier':{'learning_rate':[0.01,0.05,0.075],'alphaReg':[0.8,0.85,0.9]},
                  'RUSBoostClassifier':{'learning_rate':[0.002,0.005,0.01]}, #{'learning_rate':[0.002,0.005,0.01]}
                  
                  'NGBClassifier':{'learning_rate':[0.01,0.05,0.075],'alphaReg':[0.8,0.9]},
                  
                  'RandomForestClassifier':{'ccp_alpha':[0.01,0.02,0.03]}, # {'ccp_alpha':[0.001,0.002,0.005]}, 
                  'XGBClassifier':{'learning_rate':[0.001, 0.005, 0.01, 0.05],'reg_lambda':[0.5,1],'gamma':[0.001, 0.005, 0.01, 0.05],'max_depth': [8, 10, 12, 15]},
                  #'XGBClassifier':{'learning_rate':[0.01,0.1,0.5],'reg_lambda':[0.2,1,10],'gamma':[0.2,1,10]},
                  'AdaBoostClassifier':{'learning_rate' : [0.001,0.005,0.01]},
                  'GradientBoostingClassifier':{'learning_rate':[0.1,0.15],'subsample':[0.75],'ccp_alpha':[0.005,0.01,0.02,0.05]},
                  'CatBoostClassifier':{'learning_rate' : [0.01,0.05,0.075],'l2_leaf_reg': [5,10]},  # 10,15,20
                  
                  'HistGradientBoostingClassifier':{'learning_rate':[0.01,0.1,0.5],'l2_regularization':[0.5,1,2] }, #0.5,0.55,0.6
                  'ComplementNB':{'alpha': [0.0001,0.001,0.01,0.1,1]},
                  'MLPRegressor':{'learning_rate_init': [0.001],'alpha': [0.001,0.01,0.1,0.3]},
                  'MLPClassifier':{'alpha': [0.1,0.5,0.9]},
                  'CatBoostRegressor':{'depth' : [6,8,10],'learning_rate' : [0.01,0.05,0.1],'iterations':[300]},
                  
                  'LGBMClassifier':{'learning_rate':[0.001,0.002,0.005],'reg_lambda':[2,3,4]}, #{'learning_rate':[0.001,0.005,0.01]
                  'GNEGNEClassifier':{'eta': [0.025, 0.05, 0.1, 0.5, 1]}
                  #'LGBMClassifier':{'learning_rate':[0.1,0.15],'reg_lambda':[0.2,0.4],'reg_alpha':[0.2,0.4]}
                }
    
    HOURLY=False
    timestamp_format='%Y-%m-%d'
    RESOLUTION = '1DAY'
    
    #HOURLY=True
    #timestamp_format='%Y-%m-%d %H:%M:%S'
    #RESOLUTION = '1HOUR'
    
    
    seloffset=600
    seqhists = [600]
    TSET=600
    validationHistory = 0
    
    PEAK='peak'
    HANDLED='handled'
    
    maxvalhist=20
    
    @staticmethod
    def serialize():
        cname=os.path.join(os.getcwd(),Configuration.DATADIR,'confp.json')
        with open(cname, "w") as fp:
            json.dump(Configuration.TAUS, fp)
        #print('Serialized ' + cname)
        
    @staticmethod
    def init():
        
        Configuration.mailprefix = "_"+str(max(Configuration.seqhists))+Configuration.mailprefix
        
        algs=[]
        for a in Configuration.ALGS:
            algs.append(a+'-D')

        Configuration.ALGS=algs


            
    
       
            
            
            
        
            
        

