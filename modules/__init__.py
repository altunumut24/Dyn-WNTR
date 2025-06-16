"""
Water Network Simulator Modules Package

This package contains all the modular components for the Interactive Water Network
Event Simulator. Each module has a specific responsibility and they work together
to provide a complete simulation environment.

Module Organization:
├── config.py                    # Central configuration and constants
├── simulation.py                # Core WNTR simulation logic and event handling
├── visualization.py             # Network plotting and data visualization
├── ui_components.py             # Streamlit UI components and forms
├── batch_simulator_functions.py # Batch mode simulation capabilities
└── event_generator.py           # Automatic event generation for testing

Architecture principles:
- Each module has a single, clear responsibility
- Modules communicate through well-defined interfaces
- Configuration is centralized in config.py
- UI logic is separated from business logic
- Visualization is independent of simulation logic

Import pattern:
- Use `from modules import function_name` in main application
- All functions are available at package level
- Maintains clean separation while providing easy access
"""

# Import all main components for easy access
# This allows: from modules import create_network_plot, run_simulation_step, etc.

from .config import *                    # Configuration constants and settings
from .simulation import *                # Core simulation and event handling functions
from .visualization import *             # Network plotting and visualization functions
from .ui_components import *             # Streamlit UI components and forms
from .batch_simulator_functions import * # Batch simulation capabilities
from .event_generator import *           # Automatic event generation functions 