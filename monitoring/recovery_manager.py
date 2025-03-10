class RecoveryManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.state_manager = SystemStateManager()
        
    def handle_failure(self, component: str, error: Exception) -> bool:
        """Handle component failures with automatic recovery"""
        try:
            # Log failure
            self.logger.error(f"Component failure: {component}", exc_info=True)
            
            # Save state
            self.state_manager.save_state(component)
            
            # Attempt recovery
            if self._can_recover(component, error):
                return self._execute_recovery(component)
                
            # Fallback to backup system
            return self._activate_fallback(component)
            
        except Exception as e:
            self.logger.critical(f"Recovery failed: {e}")
            return False
