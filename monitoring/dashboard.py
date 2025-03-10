import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
from typing import List

class MonitoringDashboard:
    def __init__(self):
        self.app = dash.Dash(__name__)
        self.setup_layout()
        
    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1('Prediction System Monitor'),
            dcc.Graph(id='accuracy-graph'),
            dcc.Graph(id='resource-usage'),
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # in milliseconds
                n_intervals=0
            )
        ])
