import streamlit as st
import sys
import logging
from monitoring.monitor_manager import MonitorManager
from monitoring.health_check import SystemHealthCheck

def init_dashboard():
    """Initialize dashboard with proper error handling"""
    try:
        import talib
        return True
    except ImportError:
        st.error("TA-Lib not properly installed. Please check system configuration.")
        return False

def main():
    """Main dashboard application"""
    if not init_dashboard():
        return

    st.set_page_config(
        page_title="Prediction Platform",
        layout="wide"
    )
    
    st.title("Prediction Platform Dashboard")
    
    # Initialize components
    monitor = MonitorManager(settings={'log_level': 'INFO'})
    health_check = SystemHealthCheck()
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("System Health")
        health_metrics = health_check.verify_system_state()
        st.json(health_metrics)
        
    with col2:
        st.subheader("Performance Metrics")
        metrics = monitor.get_metrics()
        st.json(metrics)

if __name__ == "__main__":
    main()
