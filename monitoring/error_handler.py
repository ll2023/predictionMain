import logging
import traceback
from typing import Optional, Any, Callable, Dict
from functools import wraps
import sys
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ErrorContext:
    timestamp: datetime
    function: str
    error_type: str
    message: str
    traceback: str

class ErrorHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_history = []
        self.max_retries = 3
        self.recovery_strategies = {
            'DataNotFoundError': self._handle_data_not_found,
            'ConnectionError': self._handle_connection_error,
            'MemoryError': self._handle_memory_error
        }
        self.recovery_strategies.update({
            'DataValidationError': self._handle_validation_error,
            'ProcessingTimeout': self._handle_timeout,
            'DataSyncError': self._handle_sync_error,
            'ComponentFailure': self._handle_component_failure,
            'ResourceExhaustion': self._handle_resource_exhaustion
        })

    def handle_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt + 1} failed: {str(e)}"
                )
                if attempt == self.max_retries - 1:
                    self._log_error(func.__name__, e)
                    raise

    def handle_error(self, error: Exception) -> Optional[Any]:
        """Smart error handling with recovery strategies"""
        error_type = type(error).__name__
        if error_type in self.recovery_strategies:
            return self.recovery_strategies[error_type](error)
        
        self._log_error("Unhandled error", error)
        return None

    def _log_error(self, function: str, error: Exception) -> None:
        """Log error with context"""
        context = ErrorContext(
            timestamp=datetime.now(),
            function=function,
            error_type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc()
        )
        self.error_history.append(context)
        self.logger.error(f"Error in {function}: {context}")

    def _handle_memory_error(self, error: MemoryError) -> None:
        """Handle memory errors by clearing caches"""
        gc.collect()
        self.clear_caches()
        self.logger.warning("Memory error recovered by clearing caches")

    def _handle_validation_error(self, error: Exception) -> None:
        """Handle data validation errors"""
        self.logger.warning(f"Validation error: {error}")
        self._send_validation_alert(error)
        return None

    def _handle_timeout(self, error: Exception) -> None:
        """Handle processing timeout errors"""
        self.logger.error(f"Processing timeout: {error}")
        self._cleanup_stuck_processes()
        return None

    def _handle_sync_error(self, error: Exception) -> None:
        """Enhanced sync error handling with recovery"""
        try:
            self.logger.warning(f"Sync error detected: {error}")
            self._save_error_state()
            if self._trigger_replication():
                return self._restore_from_backup()
            return self._failover_to_secondary()
        except Exception as e:
            self.logger.critical(f"Recovery failed: {e}")
            return None

    def _save_error_state(self):
        """Save system state before recovery attempt"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'components': self.component_states,
            'resources': self.resource_usage
        }
        with open('logs/error_states.json', 'a') as f:
            json.dump(state, f)

    def _handle_component_failure(self, error: Exception) -> None:
        """Handle component failures with fallback"""
        self.logger.error(f"Component failure: {error}")
        return self._activate_fallback_component()

    def handle_critical_error(self, error: Exception, context: Dict[str, Any]):
        """Handle critical errors with detailed context"""
        error_report = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'message': str(error),
            'stack_trace': traceback.format_exc(),
            'context': context,
            'system_state': self._capture_system_state()
        }
        
        # Log error
        self.logger.critical(f"Critical error: {error_report}")
        
        # Save detailed report
        self._save_error_report(error_report)
        
        # Notify administrators
        self._send_error_notification(error_report)
        
        return error_report

def handle_errors(logger: logging.Logger):
    """Decorator for comprehensive error handling"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
                return None
        return wrapper
    return decorator
