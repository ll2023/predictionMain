import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Any

class DashboardApp:
    def __init__(self):
        st.set_page_config(page_title="Prediction Analysis", layout="wide")
        
    def run(self):
        st.title("Prediction Analysis Dashboard")
        
        # Load recent predictions
        predictions = self.load_recent_predictions()
        metrics = self.load_performance_metrics()
        
        # Display key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediction Accuracy", f"{metrics['accuracy']:.2%}")
        with col2:
            st.metric("Success Rate", f"{metrics['success_rate']:.2%}")
        with col3:
            st.metric("Average Latency", f"{metrics['avg_latency']:.2f}ms")
        
        self.setup_analysis_tabs()
    
    def setup_analysis_tabs(self):
        """Setup analysis tabs in the dashboard"""
        tabs = st.tabs(["Predictions", "Performance", "System Health"])
        
        with tabs[0]:
            self._render_predictions_view()
            
        with tabs[1]:
            self._render_performance_metrics()
            
        with tabs[2]:
            self._render_system_health()
    
    def _render_predictions_view(self):
        """Render predictions analysis view"""
        predictions_df = self.load_recent_predictions()
        
        # Summary metrics
        st.subheader("Prediction Summary")
        metrics = self.calculate_prediction_metrics(predictions_df)
        
        # Interactive chart
        fig = self.create_prediction_chart(predictions_df)
        st.plotly_chart(fig, use_container_width=True)
