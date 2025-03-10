import sys
import pkg_resources
import logging
from typing import List, Tuple
from pathlib import Path

def check_requirements() -> Tuple[bool, List[str]]:
    """Check if all requirements are properly installed"""
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        requirements_file = Path(__file__).parent.parent / 'requirements.txt'
        if not requirements_file.exists():
            print("Error: requirements.txt not found")
            return False, ["Missing requirements.txt"]

        with open(requirements_file) as f:
            requirements = [
                line.strip() 
                for line in f 
                if line.strip() and not line.startswith('#')
            ]
        
        missing = []
        conflicts = []
        
        for req in requirements:
            try:
                pkg_resources.require(req)
                print(f"✓ {req}")
            except pkg_resources.DistributionNotFound:
                missing.append(req)
                print(f"✗ Missing: {req}")
            except pkg_resources.VersionConflict as e:
                conflicts.append(str(e))
                print(f"✗ Conflict: {e}")
        
        success = not (missing or conflicts)
        if success:
            print("All requirements satisfied!")
        
        return success, missing + conflicts
        
    except Exception as e:
        print(f"Error checking requirements: {e}")
        return False, [str(e)]

if __name__ == "__main__":
    success, issues = check_requirements()
    sys.exit(0 if success else 1)
