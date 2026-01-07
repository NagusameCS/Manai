"""
Manai - AI-powered Math and Physics Video Generator
Uses local LLMs (Ollama) + Manim for educational content creation
"""

__version__ = "2.0.0"
__author__ = "Manai Team"

from manai.client import ManaiClient
from manai.generator import ManimGenerator
from manai.ollama_client import OllamaClient
from manai.agent import Agent, ThinkingMode
from manai.math_engine import MathEngine, math_engine
from manai.ui import UI, ui
from manai.templates_2d import get_2d_template, list_2d_templates
from manai.templates_3d import get_3d_template, list_3d_templates
from manai.transitions import create_transition, list_transitions
from manai.knowledge import manim_docs

__all__ = [
    # Core
    "ManaiClient",
    "ManimGenerator", 
    "OllamaClient",
    # Agent
    "Agent",
    "ThinkingMode",
    # Math
    "MathEngine",
    "math_engine",
    # UI
    "UI",
    "ui",
    # Templates
    "get_2d_template",
    "list_2d_templates",
    "get_3d_template",
    "list_3d_templates",
    # Transitions
    "create_transition",
    "list_transitions",
    # Knowledge
    "manim_docs",
]
