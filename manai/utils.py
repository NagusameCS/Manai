"""
Utility functions for Manai
"""

import re
import ast
import sys
from pathlib import Path
from typing import Optional, Any
import subprocess


def validate_python_syntax(code: str) -> tuple[bool, Optional[str]]:
    """
    Validate Python syntax without executing.
    
    Args:
        code: Python code to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def extract_class_names(code: str) -> list[str]:
    """Extract all class names from Python code."""
    pattern = r'class\s+(\w+)\s*\('
    return re.findall(pattern, code)


def extract_scene_classes(code: str) -> list[str]:
    """Extract classes that inherit from Scene or ThreeDScene."""
    pattern = r'class\s+(\w+)\s*\(\s*(?:Scene|ThreeDScene|MovingCameraScene|ZoomedScene)'
    return re.findall(pattern, code)


def estimate_animation_duration(code: str) -> float:
    """
    Estimate the total animation duration from code.
    
    Looks for self.wait() calls and run_time parameters.
    """
    duration = 0.0
    
    # Find self.wait(n) calls
    wait_pattern = r'self\.wait\(([^)]*)\)'
    for match in re.finditer(wait_pattern, code):
        arg = match.group(1).strip()
        if arg:
            try:
                duration += float(arg)
            except ValueError:
                duration += 1.0  # Default wait time
        else:
            duration += 1.0
    
    # Find run_time parameters
    runtime_pattern = r'run_time\s*=\s*(\d+\.?\d*)'
    for match in re.finditer(runtime_pattern, code):
        try:
            duration += float(match.group(1))
        except ValueError:
            pass
    
    return duration


def format_manim_code(code: str) -> str:
    """Format Manim code for consistency."""
    # Ensure proper imports
    if 'from manim import' not in code:
        code = "from manim import *\n\n" + code
    
    # Ensure numpy is available
    if 'np.' in code and 'import numpy' not in code:
        code = code.replace(
            'from manim import *',
            'from manim import *\nimport numpy as np'
        )
    
    return code


def get_manim_version() -> Optional[str]:
    """Get the installed Manim version."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "manim", "--version"],
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except Exception:
        return None


def check_dependencies() -> dict[str, bool]:
    """Check if all required dependencies are available."""
    deps = {
        "manim": False,
        "ollama": False,
        "ffmpeg": False,
        "latex": False,
    }
    
    # Check Python packages
    try:
        import manim
        deps["manim"] = True
    except ImportError:
        pass
    
    try:
        import ollama
        deps["ollama"] = True
    except ImportError:
        pass
    
    # Check system dependencies
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True
        )
        deps["ffmpeg"] = result.returncode == 0
    except FileNotFoundError:
        pass
    
    try:
        result = subprocess.run(
            ["latex", "--version"],
            capture_output=True
        )
        deps["latex"] = result.returncode == 0
    except FileNotFoundError:
        pass
    
    return deps


def create_minimal_scene(description: str) -> str:
    """Create a minimal Manim scene template."""
    safe_name = "".join(c if c.isalnum() else "" for c in description)[:20]
    class_name = f"Scene{safe_name}" if safe_name else "GeneratedScene"
    
    return f'''from manim import *

class {class_name}(Scene):
    """
    {description}
    """
    
    def construct(self):
        # TODO: Implement animation
        title = Text("{description[:50]}...")
        self.play(Write(title))
        self.wait(2)
'''


def merge_scenes(codes: list[str], combined_name: str = "CombinedScene") -> str:
    """Merge multiple scene codes into one file."""
    # Collect all imports
    imports = set()
    classes = []
    
    for code in codes:
        # Extract imports
        for line in code.split('\n'):
            if line.startswith('import ') or line.startswith('from '):
                imports.add(line.strip())
        
        # Extract class definitions
        class_pattern = r'(class\s+\w+\s*\([^)]+\):\s*(?:"""[^"]*""")?\s*(?:def\s+\w+[^}]+)+)'
        matches = re.findall(class_pattern, code, re.DOTALL)
        classes.extend(matches)
    
    # Combine
    result = '\n'.join(sorted(imports)) + '\n\n'
    result += '\n\n'.join(classes)
    
    return result


class CodeHistory:
    """Track and manage code generation history."""
    
    def __init__(self, max_size: int = 100):
        self.history: list[dict[str, Any]] = []
        self.max_size = max_size
    
    def add(self, code: str, description: str, success: bool, output_path: Optional[str] = None):
        """Add an entry to history."""
        entry = {
            "code": code,
            "description": description,
            "success": success,
            "output_path": output_path,
        }
        
        self.history.append(entry)
        
        # Trim if necessary
        if len(self.history) > self.max_size:
            self.history = self.history[-self.max_size:]
    
    def get_last(self, n: int = 1) -> list[dict]:
        """Get the last n entries."""
        return self.history[-n:]
    
    def get_successful(self) -> list[dict]:
        """Get all successful generations."""
        return [e for e in self.history if e["success"]]
    
    def search(self, query: str) -> list[dict]:
        """Search history by description."""
        query_lower = query.lower()
        return [
            e for e in self.history
            if query_lower in e["description"].lower()
        ]
    
    def clear(self):
        """Clear history."""
        self.history = []
