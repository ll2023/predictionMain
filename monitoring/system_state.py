from typing import Dict, Any
import threading
import time

class SystemStateManager:
    """Manages system state and component synchronization"""
    
    def __init__(self):
        self.component_states = {}
        self.state_lock = threading.Lock()
        self.state_changed = threading.Event()
        
    def update_component_state(self, component: str, state: Dict[str, Any]) -> None:
        """Update component state thread-safely"""
        with self.state_lock:
            self.component_states[component] = {
                'state': state,
                'last_updated': time.time(),
                'healthy': self._verify_component_health(state)
            }
            self.state_changed.set()
            
    def wait_for_sync(self, timeout: float = 30.0) -> bool:
        """Wait for system components to synchronize"""
        return self.state_changed.wait(timeout)
