from typing import Dict, List
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Task:
    name: str
    priority: str
    effort: str
    dependencies: List[str]
    status: str = 'pending'

class ImprovementTasks:
    """Manage and track improvement tasks"""
    
    def __init__(self):
        self.tasks = self._load_tasks()
        
    def _load_tasks(self) -> Dict[str, List[Task]]:
        """Load tasks from configuration"""
        tasks_file = Path('docs/IMPROVEMENT_PLAN.md')
        if not tasks_file.exists():
            return self._create_default_tasks()
            
        return self._parse_tasks(tasks_file)
    
    def generate_timeline(self) -> str:
        """Generate implementation timeline"""
        timeline = {
            'phase1': self._get_high_priority_tasks(),
            'phase2': self._get_medium_priority_tasks(),
            'phase3': self._get_low_priority_tasks()
        }
        
        output_file = Path('docs/TIMELINE.md')
        with open(output_file, 'w') as f:
            yaml.dump(timeline, f)
        
        return str(output_file)
