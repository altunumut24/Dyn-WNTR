"""
Configuration file for Interactive Network Simulator.
Contains all constants, settings, and event definitions.
"""

from typing import Dict, List, Any

# Network Configuration
INP_FILE = 'NET_4.inp'
SIMULATION_DURATION_SECONDS = 2 * 3600  # 2 hours
HYDRAULIC_TIMESTEP_SECONDS = 60  # 1 minute steps
REPORT_TIMESTEP_SECONDS = 60

# Define available events for each element type
NODE_EVENTS: Dict[str, Dict[str, Dict[str, Any]]] = {
    'Junction': {
        'start_leak': {'params': ['leak_area', 'leak_discharge_coefficient'], 'defaults': [0.01, 0.75]},
        'stop_leak': {'params': [], 'defaults': []},
        'add_demand': {'params': ['base_demand', 'pattern_name', 'category'], 'defaults': [0.1, None, 'user_added']},
        'remove_demand': {'params': ['name'], 'defaults': ['user_added']},
        'add_fire_fighting_demand': {'params': ['fire_flow_demand', 'fire_start', 'fire_end'], 'defaults': [0.5, 300, 1800]}
    },
    'Tank': {
        'start_leak': {'params': ['leak_area', 'leak_discharge_coefficient'], 'defaults': [0.01, 0.75]},
        'stop_leak': {'params': [], 'defaults': []},
        'set_tank_head': {'params': ['head'], 'defaults': [50.0]}
    }
}

LINK_EVENTS: Dict[str, Dict[str, Dict[str, Any]]] = {
    'Pipe': {
        'close_pipe': {'params': [], 'defaults': []},
        'open_pipe': {'params': [], 'defaults': []},
        'set_pipe_diameter': {'params': ['diameter'], 'defaults': [0.3]}
    },
    'Pump': {
        'close_pump': {'params': [], 'defaults': []},
        'open_pump': {'params': [], 'defaults': []},
        'set_pump_speed': {'params': ['speed'], 'defaults': [1.0]},
        'set_pump_head_curve': {'params': ['head_curve'], 'defaults': ['default_curve']}
    },
    'Valve': {
        'close_valve': {'params': [], 'defaults': []},
        'open_valve': {'params': [], 'defaults': []}
    }
}

# UI Configuration
DEFAULT_MONITORED_NODES_COUNT = 5
DEFAULT_MONITORED_LINKS_COUNT = 5

# Color scheme constants for consistency
PRESSURE_COLOR_RANGES = {
    'low': {'r': 68, 'g': 1, 'b': 84},      # Dark purple
    'medium_low': {'r': 59, 'g': 82, 'b': 139},    # Blue
    'medium': {'r': 33, 'g': 145, 'b': 140},       # Teal
    'medium_high': {'r': 94, 'g': 201, 'b': 98},   # Green
    'high': {'r': 253, 'g': 231, 'b': 37}          # Yellow
}

FLOW_COLOR_RANGES = {
    'reverse': {'r': 68, 'g': 1, 'b': 84},         # Purple for reverse flow
    'low': {'r': 33, 'g': 145, 'b': 140},          # Teal for low flow
    'medium': {'r': 94, 'g': 201, 'b': 98},        # Green for medium flow
    'high': {'r': 253, 'g': 231, 'b': 37}          # Yellow for high flow
}

# Chart configuration
CHART_HEIGHT = 400
LEGEND_HEIGHT = 100
TIMELINE_HEIGHT = 300

# Flow visualization parameters
MIN_FLOW_WIDTH = 1
MAX_FLOW_WIDTH = 10 