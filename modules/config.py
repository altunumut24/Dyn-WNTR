"""
Configuration file for Interactive Network Simulator.

This module contains all the constants, settings, and event definitions used throughout
the application. Think of this as the "settings file" where we define:
- What network file to load
- How long simulations should run
- What events are available for different network elements
- Color schemes for visualizations
- Default values for the UI

"""

from typing import Dict, List, Any

# =============================================================================
# NETWORK CONFIGURATION
# =============================================================================
# These settings control the basic simulation parameters

# The water network file to load by default
# This should be an EPANET .inp file in the same directory as the app
INP_FILE = 'NET_4.inp'

# How long the simulation should run (in seconds)
# 2 * 3600 = 2 hours = 7200 seconds
SIMULATION_DURATION_SECONDS = 2 * 3600  # 2 hours

# How often to calculate hydraulics (in seconds)
# Smaller values = more accurate but slower simulation
# 60 seconds = calculate every minute
HYDRAULIC_TIMESTEP_SECONDS = 60  # 1 minute steps

# How often to save results (in seconds)
# This controls how much data we store during simulation
REPORT_TIMESTEP_SECONDS = 60

# =============================================================================
# EVENT DEFINITIONS
# =============================================================================
# These dictionaries define what events are available for each type of network element
# 
# Structure explanation:
# - First level: Element type (Junction, Tank, etc.)
# - Second level: Event name (start_leak, close_pipe, etc.)
# - Third level: Event configuration
#   - 'params': List of parameter names this event needs
#   - 'defaults': Default values for those parameters
#
# To add a new event type:
# 1. Add it to the appropriate dictionary below
# 2. Define what parameters it needs
# 3. Provide sensible default values
# 4. Implement the event logic in simulation.py

# Events available for NODE elements (Junctions, Tanks, Reservoirs)
NODE_EVENTS: Dict[str, Dict[str, Dict[str, Any]]] = {
    'Junction': {
        # Leak events - simulate pipe breaks or leaks at junctions
        'start_leak': {
            'params': ['leak_area', 'leak_discharge_coefficient'], 
            'defaults': [0.01, 0.75]  # Small leak area, standard discharge coefficient
        },
        'stop_leak': {
            'params': [], 
            'defaults': []  # No parameters needed to stop a leak
        },
        
        # Demand events - change water consumption at junctions
        'add_demand': {
            'params': ['base_demand', 'pattern_name', 'category'], 
            'defaults': [0.1, None, 'user_added']  # 0.1 m³/s additional demand
        },
        'remove_demand': {
            'params': ['name'], 
            'defaults': ['user_added']  # Remove the demand we added
        },
        
        # Emergency events - simulate fire fighting water usage
        'add_fire_fighting_demand': {
            'params': ['fire_flow_demand', 'fire_start', 'fire_end'], 
            'defaults': [0.5, 300, 1800]  # 0.5 m³/s for 25 minutes (300-1800 seconds)
        }
    },
    'Tank': {
        # Leak events - tanks can also leak
        'start_leak': {
            'params': ['leak_area', 'leak_discharge_coefficient'], 
            'defaults': [0.01, 0.75]
        },
        'stop_leak': {
            'params': [], 
            'defaults': []
        },
        
        # Tank level control - manually set tank water level
        'set_tank_head': {
            'params': ['head'], 
            'defaults': [50.0]  # 50 meters water level
        }
    }
}

# Events available for LINK elements (Pipes, Pumps, Valves)
LINK_EVENTS: Dict[str, Dict[str, Dict[str, Any]]] = {
    'Pipe': {
        # Pipe control events - simulate maintenance or failures
        'close_pipe': {
            'params': [], 
            'defaults': []  # Simply close the pipe (no water flow)
        },
        'open_pipe': {
            'params': [], 
            'defaults': []  # Reopen a closed pipe
        },
        
        # Pipe modification events - change pipe characteristics
        'set_pipe_diameter': {
            'params': ['diameter'], 
            'defaults': [0.3]  # 0.3 meters diameter (30cm)
        }
    },
    'Pump': {
        # Pump control events - simulate pump failures or maintenance
        'close_pump': {
            'params': [], 
            'defaults': []  # Turn off the pump
        },
        'open_pump': {
            'params': [], 
            'defaults': []  # Turn on the pump
        },
        
        # Pump operation events - change how the pump works
        'set_pump_speed': {
            'params': ['speed'], 
            'defaults': [1.0]  # 1.0 = normal speed, 0.5 = half speed, etc.
        },
        'set_pump_head_curve': {
            'params': ['head_curve'], 
            'defaults': ['default_curve']  # Change pump performance curve
        }
    },
    'Valve': {
        # Valve control events - open/close valves for flow control
        'close_valve': {
            'params': [], 
            'defaults': []  # Close valve (no flow)
        },
        'open_valve': {
            'params': [], 
            'defaults': []  # Open valve (allow flow)
        }
    }
}

# =============================================================================
# UI CONFIGURATION
# =============================================================================
# These settings control the user interface appearance and behavior

# How many nodes and links to monitor by default in charts
# Monitoring too many elements can slow down the interface
DEFAULT_MONITORED_NODES_COUNT = 5
DEFAULT_MONITORED_LINKS_COUNT = 5

# =============================================================================
# COLOR SCHEMES
# =============================================================================
# These define the colors used in network visualizations
# Colors are defined as RGB values (Red, Green, Blue) from 0-255

# Colors for pressure visualization (nodes)
# Low pressure = purple/dark, High pressure = yellow/bright
PRESSURE_COLOR_RANGES = {
    'low': {'r': 68, 'g': 1, 'b': 84},          # Dark purple - concerning low pressure
    'medium_low': {'r': 59, 'g': 82, 'b': 139}, # Blue - below normal pressure  
    'medium': {'r': 33, 'g': 145, 'b': 140},    # Teal - normal pressure
    'medium_high': {'r': 94, 'g': 201, 'b': 98}, # Green - good pressure
    'high': {'r': 253, 'g': 231, 'b': 37}       # Yellow - high pressure
}

# Colors for flow visualization (links/pipes)
# Different colors help identify flow direction and magnitude
FLOW_COLOR_RANGES = {
    'reverse': {'r': 68, 'g': 1, 'b': 84},      # Purple - reverse flow (unusual)
    'low': {'r': 33, 'g': 145, 'b': 140},       # Teal - low flow
    'medium': {'r': 94, 'g': 201, 'b': 98},     # Green - normal flow
    'high': {'r': 253, 'g': 231, 'b': 37}       # Yellow - high flow
}

# =============================================================================
# CHART CONFIGURATION
# =============================================================================
# These control the size and appearance of charts and visualizations

# Standard heights for different chart types (in pixels)
CHART_HEIGHT = 400        # Main monitoring charts
LEGEND_HEIGHT = 100       # Color scale legends
TIMELINE_HEIGHT = 300     # Event timeline charts

# Flow visualization parameters for pipe width
# Pipes with higher flow are drawn thicker
MIN_FLOW_WIDTH = 1        # Minimum pipe width (pixels)
MAX_FLOW_WIDTH = 10       # Maximum pipe width (pixels) 