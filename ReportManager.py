import os
from Configuration import Configuration
import time
import pandas as pd
import numpy as np
from dataman.DataManager import DataManager
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

class ReportManager(object):
    """
    ReportManager is responsible for managing and generating reports.
    """
    
    def __init__(self, fusion):
        """
        Initialize the ReportManager with a fusion engine.
        
        Parameters:
        fusion (object): The fusion engine to use.
        """
        self.engine = fusion
        self.engine.setReportMan(self)
        self.now = time.strftime("%Y-%m-%d-%H-%M")
        self.lastTimestamps = {}
        dsname = self.engine.dataManager.datasource.split('_')[0]
        self.actlogD = os.path.join(os.getcwd(), dsname + Configuration.mailprefix)
        self.lastMessage = ''
        self.lastExtMessage = ''
        self.penMessage = ''
        self.globalDatasource = DataManager(dsname)
        self.metrics_history = {}
        self.reports_dir = os.path.join(os.getcwd(), 'reports')
        os.makedirs(self.reports_dir, exist_ok=True)

    def reportAggregationA(self, a, targetts=None):
        """
        Aggregate reports for a specific algorithm.
        
        Parameters:
        a (str): The algorithm identifier.
        targetts (str, optional): The target timestamp. Defaults to None.
        """
        try:
            dsname = self.engine.dataManager.datasource.split('_')[0]
            al = os.path.join(os.getcwd(), dsname + '_' + a + Configuration.mailprefix)
            
            if targetts is None:
                ltss = self.engine.dataManager.datesList[-1]
            else:
                ltss = targetts
                
            td = pd.Timestamp(ltss)
            timestamp = ltss
                        
            longs = set([])
            shorts = set([])

            pext = []
            inconsistent = set([])
            directions = {}
            leaders = {}
            
            agreement = True
            predictors = {}
            
            for (ticker, prediction, pcap, acc) in self.engine.bestPredictPowerA[a][timestamp]:
                alg = pcap.split('_')[1]
                
                if ticker not in predictors:
                    predictors[ticker] = set([alg])
                else:
                    predictors[ticker].add(alg)
                
                if ticker in directions:
                    if prediction * directions[ticker] < 0:
                        inconsistent.add(ticker)
                else:
                    directions[ticker] = prediction
                
                (newsr, newg) = acc
                if ticker in leaders:
                    (alg_, sr, p) = leaders[ticker]
                    if newsr > sr:
                        leaders[ticker] = (alg, newsr, prediction)
                else:
                    leaders[ticker] = (alg, newsr, prediction)
            
            if agreement == False:
                for ticker in leaders:
                    (alg, g, prediction) = leaders[ticker]
                    if prediction > 0:
                        longs.add(ticker)
                    if prediction < 0:
                        shorts.add(ticker)
                    pext.append(ticker + ':' + str(prediction) + ':' + alg)
            else:
                for (ticker, prediction, pcap, acc) in self.engine.bestPredictPowerA[a][timestamp]:
                    predtag = pcap.split('_')[1]
                    if prediction > 0:
                        longs.add(ticker)
                    if prediction < 0:
                        shorts.add(ticker)
                    pext.append(ticker + ':' + str(prediction) + ':' + pcap)
            
            self.lastMessage = 'L:{0} S:{1}'.format(' '.join(longs), ' '.join(shorts))
            self.lastExtMessage = '#'.join(pext)
                    
            del longs
            del shorts
                
            if targetts is not None:
                lines = []
                lines1 = []
                with open(al, 'rt') as rdf:
                    lines = rdf.read().splitlines()
                
                for l in lines:
                    ll = l.split(',')
                    if ll[0] == ltss:
                        addt = '{0},{1},{2}'.format(ltss, self.lastMessage, self.lastExtMessage)
                        lines1.append(addt)
                    else:
                        lines1.append(l)
                
                with open(al, 'wt') as wdf:
                    for l in lines1:
                        wdf.write(l + '\n')
                
                print('Aggregation modified')
            else:
                with open(al, 'a') as fa:
                    addt = '{0},{1},{2}\n'.format(td.strftime(Configuration.timestamp_format), self.lastMessage, self.lastExtMessage)
                    print(addt)
                    fa.write(addt)
                    fa.close()
                    print('Aggregation saved')
            
            # Add more detailed insights
            print(f"Detailed Report for Algorithm {a} on {timestamp}")
            print(f"Longs: {longs}")
            print(f"Shorts: {shorts}")
            print(f"Predictions: {pext}")
            
            # Save data to CSV for further analysis
            report_data = {
                'timestamp': [timestamp],
                'longs': [list(longs)],
                'shorts': [list(shorts)],
                'predictions': [pext]
            }
            report_df = pd.DataFrame(report_data)
            report_df.to_csv(f'report_{a}_{timestamp}.csv', index=False)
            
            # Generate graphs
            self.generateGraphs(report_df, a, timestamp)
            
        except Exception as e:
            print(f"Error in reportAggregationA: {e}")

    def reportAggregation(self, targetts=None):
        """
        Aggregate reports.
        
        Parameters:
        targetts (str, optional): The target timestamp. Defaults to None.
        """
        try:
            al = self.actlogD
            
            if targetts is None:
                ltss = self.engine.dataManager.datesList[-1]
            else:
                ltss = targetts
                
            td = pd.Timestamp(ltss)
            timestamp = ltss
                        
            longs = []
            shorts = []

            pext = []
                
            for (sticker, prediction, pcap) in self.engine.bestPredictPower[timestamp]:
                predtag = pcap.split('_')[1]
                if prediction > 0:
                    longs.append(sticker)
                if prediction < 0:
                    shorts.append(sticker)
                pext.append(sticker + ':' + str(prediction) + ':' + pcap)
                    
            self.lastMessage = 'L:{0} S:{1}'.format(' '.join(longs), ' '.join(shorts))
            self.lastExtMessage = '#'.join(pext)
                    
            del longs
            del shorts
                
            if targetts is not None:
                lines = []
                lines1 = []
                with open(al, 'rt') as rdf:
                    lines = rdf.read().splitlines()
                
                for l in lines:
                    ll = l.split(',')
                    if ll[0] == ltss:
                        addt = '{0},{1},{2}'.format(ltss, self.lastMessage, self.lastExtMessage)
                        lines1.append(addt)
                    else:
                        lines1.append(l)
                
                with open(al, 'wt') as wdf:
                    for l in lines1:
                        wdf.write(l + '\n')
                
                print('Aggregation modified')
            else:
                with open(al, 'a') as fa:
                    addt = '{0},{1},{2}\n'.format(td.strftime(Configuration.timestamp_format), self.lastMessage, self.lastExtMessage)
                    fa.write(addt)
                    fa.close()
                    print('Aggregation saved')
            
            # Add more detailed insights
            print(f"Detailed Report on {timestamp}")
            print(f"Longs: {longs}")
            print(f"Shorts: {shorts}")
            print(f"Predictions: {pext}")
            
            # Save data to CSV for further analysis
            report_data = {
                'timestamp': [timestamp],
                'longs': [longs],
                'shorts': [shorts],
                'predictions': [pext]
            }
            report_df = pd.DataFrame(report_data)
            report_df.to_csv(f'report_{timestamp}.csv', index=False)
            
            # Generate graphs
            self.generateGraphs(report_df, 'all', timestamp)
            
        except Exception as e:
            print(f"Error in reportAggregation: {e}")

    def generateGraphs(self, report_df, algorithm, timestamp):
        """
        Generate graphs for the report data.
        
        Parameters:
        report_df (DataFrame): The report data.
        algorithm (str): The algorithm identifier.
        timestamp (str): The timestamp of the report.
        """
        try:
            # Create reports directory if it doesn't exist
            reports_dir = os.path.join(os.getcwd(), 'reports', algorithm, timestamp)
            os.makedirs(reports_dir, exist_ok=True)
            
            # Add performance metrics
            self._generate_performance_metrics(report_df, algorithm, timestamp)
            
            # Enhanced visualization
            self._plot_prediction_distribution(report_df, algorithm, timestamp)
            self._plot_cumulative_returns(report_df, algorithm, timestamp)
            
        except Exception as e:
            print(f"Error in generateGraphs: {e}")

    def _generate_performance_metrics(self, report_df, algorithm, timestamp):
        """Generate detailed performance metrics"""
        metrics = {
            'accuracy': self._calculate_accuracy(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'max_drawdown': self._calculate_max_drawdown()
        }
        
        # Save metrics
        metrics_file = os.path.join('reports', algorithm, timestamp, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)

    def _calculate_metrics(self, predictions, actuals):
        """Calculate performance metrics"""
        try:
            metrics = {
                'accuracy': np.mean(np.sign(predictions) == np.sign(actuals)),
                'sharpe_ratio': self._calculate_sharpe_ratio(predictions),
                'max_drawdown': self._calculate_max_drawdown(predictions),
                'correlation': stats.pearsonr(predictions, actuals)[0],
                'volatility': np.std(predictions)
            }
            return metrics
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {}

    def _plot_prediction_distribution(self, report_df, algorithm, timestamp):
        """Plot prediction distribution and analytics"""
        try:
            plt.figure(figsize=(15, 10))
            
            # Prediction distribution
            plt.subplot(2, 2, 1)
            predictions = [float(pred.split(':')[1]) for pred in report_df['predictions'][0]]
            sns.histplot(predictions, kde=True)
            plt.title('Prediction Distribution')
            
            # Correlation matrix
            plt.subplot(2, 2, 2)
            corr_matrix = pd.DataFrame(predictions).corr()
            sns.heatmap(corr_matrix, annot=True)
            plt.title('Correlation Matrix')
            
            # Time series
            plt.subplot(2, 2, 3)
            plt.plot(predictions, marker='o')
            plt.title('Prediction Time Series')
            
            plt.savefig(os.path.join(self.reports_dir, f'analysis_{algorithm}_{timestamp}.png'))
            plt.close()
            
        except Exception as e:
            print(f"Error in plot_prediction_distribution: {e}")

    # ...existing code...
















