class ResourceManager:
    """Manages system resources and enforces limits"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.resource_locks = {}
        self._setup_resource_limits()

    def acquire_resources(self, task_name: str, requirements: Dict[str, float]) -> bool:
        """Attempt to acquire resources for a task"""
        with threading.Lock():
            if self._check_resource_availability(requirements):
                self._allocate_resources(task_name, requirements)
                return True
            return False

    def release_resources(self, task_name: str) -> None:
        """Release resources held by a task"""
        with threading.Lock():
            if task_name in self.resource_locks:
                self._deallocate_resources(task_name)
                self.resource_locks.pop(task_name)
