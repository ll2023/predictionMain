import streamlit as st
import pandas as pd
from typing import Dict, Any
from monitoring.monitor_manager import MonitorManager
from monitoring.health_check import SystemHealthCheck

class SystemDashboard:
    """Real-time system monitoring dashboard"""
    
    def __init__(self):
        self.monitor = MonitorManager(settings={'log_level': 'INFO'})
        self.health_check = SystemHealthCheck()
        
    def render(self):
        """Render dashboard components"""
        st.title("System Monitoring Dashboard")
        
        # System Health
        health_metrics = self.health_check.verify_system_state()
        self._render_health_metrics(health_metrics)
        
        # Performance Metrics
        perf_metrics = self.monitor.get_metrics()
        self._render_performance_metrics(perf_metrics)
