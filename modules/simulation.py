"""
Simulation functions for Interactive Network Simulator.

This module contains the core simulation logic that bridges between the WNTR
(Water Network Tool for Resilience) library and our Streamlit application.

Key responsibilities:
- Loading water network models from INP files
- Managing simulation state and execution
- Applying events to the network during simulation
- Collecting and organizing simulation results
- Handling event scheduling and execution

The main flow is:
1. Load network model from INP file
2. Initialize simulator with time settings
3. Schedule events to happen at specific times
4. Run simulation step by step
5. Collect pressure/flow data at each step
6. Update the network visualization
"""

import json
import datetime
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st

# Import WNTR components for water network simulation
from mwntr.sim.interactive_network_simulator import InteractiveWNTRSimulator
from mwntr.network import WaterNetworkModel

# Import our configuration settings
from .config import (
    INP_FILE, SIMULATION_DURATION_SECONDS, 
    HYDRAULIC_TIMESTEP_SECONDS, REPORT_TIMESTEP_SECONDS,
    NODE_EVENTS, LINK_EVENTS
)


def load_network_model(inp_file: str) -> Optional[WaterNetworkModel]:
    """
    Load the water network model from an EPANET INP file.
    
    This function:
    1. Reads the INP file (EPANET format) containing network topology
    2. Creates a WNTR WaterNetworkModel object
    3. Configures simulation options (time settings, demand model)
    4. Returns the configured network model
    
    Args:
        inp_file (str): Path to the EPANET INP file
        
    Returns:
        Optional[WaterNetworkModel]: Loaded network model, or None if loading failed
        
    Technical details:
    - INP files contain network topology (nodes, pipes, pumps, etc.)
    - PDD = Pressure Driven Demand (more realistic than demand driven)
    - Time settings control how long and how often simulation calculates
    """
    try:
        # Load the network from the INP file
        wn = WaterNetworkModel(inp_file)
        
        # Configure simulation options
        # PDD = Pressure Driven Demand (more realistic than Demand Driven)
        wn.options.hydraulic.demand_model = 'PDD'
        
        # Set time parameters from our config
        wn.options.time.duration = SIMULATION_DURATION_SECONDS 
        wn.options.time.hydraulic_timestep = HYDRAULIC_TIMESTEP_SECONDS 
        wn.options.time.report_timestep = REPORT_TIMESTEP_SECONDS 
        
        return wn
        
    except FileNotFoundError:
        # File doesn't exist - show user-friendly error
        st.error(f"Error: INP file '{inp_file}' not found. Place it in the same directory as the script.")
        return None
    except Exception as e:
        # Other errors (corrupted file, wrong format, etc.)
        st.error(f"Error loading INP file '{inp_file}': {e}")
        return None


def apply_event_to_simulator(sim: InteractiveWNTRSimulator, wn: WaterNetworkModel, event: Dict) -> Tuple[bool, str]:
    """
    Apply a scheduled event to the network simulation.
    
    This function takes an event (like "close pipe" or "start leak") and applies it
    to the network model. Events change the network behavior during simulation.
    
    Args:
        sim (InteractiveWNTRSimulator): The simulation engine
        wn (WaterNetworkModel): The network model to modify
        event (Dict): Event dictionary containing:
            - element_name: Which network element to affect
            - event_type: What type of event (close_pipe, start_leak, etc.)
            - element_category: 'node' or 'link'
            - parameters: Event-specific parameters
            
    Returns:
        Tuple[bool, str]: (Success flag, Status message)
        
    Event system:
    - Events are changes that happen during simulation
    - Node events affect junctions, tanks, reservoirs
    - Link events affect pipes, pumps, valves
    - Each event type has different parameters and effects
    """
    try:
        # Extract event information
        element_name = event['element_name']        # Which element (e.g., "PIPE1", "JUNCTION5")
        event_type = event['event_type']            # What to do (e.g., "close_pipe", "start_leak")
        element_category = event['element_category'] # "node" or "link"
        parameters = event.get('parameters', {})    # Event-specific settings
        
        # Handle NODE events (junctions, tanks, reservoirs)
        if element_category == 'node':
            # Get the specific node object from the network
            node = wn.get_node(element_name)
            
            # Apply the appropriate event based on event_type
            if event_type == 'start_leak':
                # Simulate a leak at this node (water loss)
                sim.start_leak(node, **parameters)
                
            elif event_type == 'stop_leak':
                # Stop an existing leak at this node
                sim.stop_leak(node)
                
            elif event_type == 'add_demand':
                # Add additional water demand at this node
                base_demand = parameters.get('base_demand', 0.0)
                pattern_name = parameters.get('pattern_name', None)
                category = parameters.get('category', None)
                
                # Create a demand pattern if needed
                if pattern_name is None and category is not None:
                    pattern_name = category
                    # Add a constant pattern (multiplier = 1.0) if it doesn't exist
                    if pattern_name not in wn.pattern_name_list:
                        wn.add_pattern(pattern_name, [1.0])
                
                sim.add_demand(node, base_demand, name=pattern_name, category=category)
                
            elif event_type == 'remove_demand':
                # Remove a previously added demand
                name = parameters.get('name', None)
                sim.remove_demand(node, name=name)
                
            elif event_type == 'add_fire_fighting_demand':
                # Add high water demand for fire fighting
                sim.add_fire_fighting_demand(node, **parameters)
                
            elif event_type == 'set_tank_head':
                # Manually set the water level in a tank
                sim.set_tank_head(node, **parameters)
                
            else:
                # Unknown event type for nodes
                return False, f"Unknown node event type: {event_type}"
                
        # Handle LINK events (pipes, pumps, valves)        
        elif element_category == 'link':
            # Get the specific link object from the network
            link = wn.get_link(element_name)
            
            # Apply the appropriate event based on event_type
            if event_type == 'close_pipe':
                # Close a pipe (no water flow through it)
                sim.close_pipe(link)
                
            elif event_type == 'open_pipe':
                # Reopen a previously closed pipe
                sim.open_pipe(link)
                
            elif event_type == 'close_pump':
                # Turn off a pump (no water pumping)
                sim.close_pump(link)
                
            elif event_type == 'open_pump':
                # Turn on a pump (resume pumping)
                sim.open_pump(link)
                
            elif event_type == 'close_valve':
                # Close a valve (no flow through it)
                sim.close_valve(link)
                
            elif event_type == 'open_valve':
                # Open a valve (allow flow through it)
                sim.open_valve(link)
                
            elif event_type == 'set_pipe_diameter':
                # Change the diameter of a pipe (affects flow capacity)
                sim.set_pipe_diameter(link, **parameters)
                
            elif event_type == 'set_pump_speed':
                # Change pump speed (affects pumping rate)
                sim.set_pump_speed(link, **parameters)
                
            elif event_type == 'set_pump_head_curve':
                # Change pump performance characteristics
                sim.set_pump_head_curve(link, **parameters)
                
            else:
                # Unknown event type for links
                return False, f"Unknown link event type: {event_type}"
                
        else:
            # Element category is neither 'node' nor 'link'
            return False, f"Unknown element category: {element_category}"
            
        # Event applied successfully
        return True, f"Applied {event_type} to {element_name}"
        
    except Exception as e:
        # Something went wrong during event application
        return False, f"Error applying event: {str(e)}"


from .rl.agents.dqn_agent import DQNAgent
from .rl.envs.wdn_env import WDNEnv

def run_simulation_step(sim: InteractiveWNTRSimulator, wn: WaterNetworkModel, agent: DQNAgent, env: WDNEnv, state, edge_feats) -> Tuple[bool, float, str]:
    """
    Execute one time step of the hydraulic simulation.
    
    This function:
    1. Checks if simulation can continue
    2. Advances simulation by one time step
    3. Gets the latest results from WNTR
    4. Updates network elements with current pressure/flow values
    5. Returns success status and current simulation time
    
    Args:
        sim (InteractiveWNTRSimulator): The simulation engine
        wn (WaterNetworkModel): The network model to update
        
    Returns:
        Tuple[bool, float, str]: (Success, Current time in seconds, Status message)
        
    Technical details:
    - Each step advances simulation by one hydraulic timestep
    - Results contain pressure at nodes and flow in links
    - Network model is updated so visualization can show current values
    """
    try:
        # Check if simulation has reached its end
        if sim.is_terminated():
            return False, sim.get_sim_time(), "Simulation terminated"
        
        # Execute the next hydraulic timestep (this is where WNTR does the math)
        #sim.step_sim()

        # Integrate with RL agent
        action = agent.act(state, edge_feats, eps=1)
        info = env.step(action)
        (next_state, next_edge_feats), reward, done = info.observation, info.reward, info.done
        

        # Get current simulation time
        current_time = sim.get_sim_time()
        
        # Get the latest simulation results from WNTR
        results = sim.get_results()
        current_time_step = current_time
        
        # Update node pressures in the network model
        # This allows the visualization to show current pressure values
        if hasattr(results, 'node') and 'pressure' in results.node:
            pressure_data = results.node['pressure']
            if current_time_step in pressure_data.index:
                for node_name in pressure_data.columns:
                    if node_name in wn.node_name_list:
                        node = wn.get_node(node_name)
                        # Store current pressure in the node object
                        node.pressure = pressure_data.loc[current_time_step, node_name]
        
        # Update link flows in the network model
        # This allows the visualization to show current flow values
        if hasattr(results, 'link') and 'flowrate' in results.link:
            flow_data = results.link['flowrate']
            if current_time_step in flow_data.index:
                for link_name in flow_data.columns:
                    if link_name in wn.link_name_list:
                        link = wn.get_link(link_name)
                        # Store current flow in the link object
                        link.flow = flow_data.loc[current_time_step, link_name]
        
        # Return success with formatted time message
        return True, current_time, f"Simulation step completed at {datetime.timedelta(seconds=int(current_time))}", next_state, next_edge_feats, reward
        
    except Exception as e:
        # Something went wrong during simulation step
        print(e)
        return False, sim.get_sim_time(), f"Simulation error: {str(e)}", state, edge_feats, reward


def load_events_from_json(uploaded_file) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load a list of events from an uploaded JSON file.
    
    This function is used for batch mode where users can upload a JSON file
    containing multiple pre-defined events to run automatically.
    
    Expected JSON format:
    {
        "events": [
            {
                "element_name": "PIPE1",
                "event_type": "close_pipe",
                "element_category": "link",
                "scheduled_time": 3600,
                "parameters": {}
            }
        ],
        "metadata": {
            "description": "Emergency scenario",
            "author": "Engineer Name"
        }
    }
    
    Args:
        uploaded_file: Streamlit uploaded file object containing JSON
        
    Returns:
        Tuple[List[Dict], Dict]: (List of events, Metadata dictionary)
        
    Usage notes:
    - Enables batch processing of multiple events
    - JSON format allows complex event scenarios to be saved and reused
    - Metadata can contain scenario descriptions, author info, etc.
    """
    try:
        # Read the file content
        content = uploaded_file.getvalue()
        # Parse JSON data
        data = json.loads(content)
        
        # Extract events list (empty list if not found)
        events = data.get('events', [])
        # Extract metadata (empty dict if not found)
        metadata = data.get('metadata', {})
        
        return events, metadata
        
    except json.JSONDecodeError as e:
        # File is not valid JSON
        st.error(f"Invalid JSON format: {e}")
        return [], {}
    except Exception as e:
        # Other errors (file read issues, etc.)
        st.error(f"Error loading events: {e}")
        return [], {}


def get_available_events(element_type: str, element_category: str) -> Dict[str, Dict]:
    """
    Get the list of available events for a specific element type.
    
    This function looks up what events can be applied to a given element
    based on its type (Junction, Pipe, etc.) and category (node or link).
    
    Args:
        element_type (str): Type of element (e.g., 'Junction', 'Pipe', 'Pump')
        element_category (str): Category ('node' or 'link')
        
    Returns:
        Dict[str, Dict]: Dictionary of available events with their parameters
        
    Implementation details:
    - Different element types support different events
    - Junctions can have leaks and demand changes
    - Pipes can be closed/opened and have diameter changes
    - This function helps the UI show only relevant events
    """
    if element_category == 'node':
        # Look up node events from config (Junction, Tank, Reservoir)
        return NODE_EVENTS.get(element_type, {})
    elif element_category == 'link':
        # Look up link events from config (Pipe, Pump, Valve)
        return LINK_EVENTS.get(element_type, {})
    else:
        # Unknown category
        return {}


def create_event(element_name: str, element_type: str, element_category: str, 
                event_type: str, scheduled_time: float, parameters: Dict) -> Dict[str, Any]:
    """
    Create a new event dictionary with all required fields.
    
    This function standardizes event creation by ensuring all events
    have the same structure and required fields.
    
    Args:
        element_name (str): Name of the network element (e.g., "PIPE1", "JUNCTION5")
        element_type (str): Type of element (e.g., "Pipe", "Junction")
        element_category (str): Category ("node" or "link")
        event_type (str): Type of event (e.g., "close_pipe", "start_leak")
        scheduled_time (float): When to apply the event (seconds from start)
        parameters (Dict): Event-specific parameters
        
    Returns:
        Dict[str, Any]: Complete event dictionary ready for scheduling
        
    Event structure:
    - Events are dictionaries with standardized fields
    - 'scheduled_time' determines when the event happens
    - 'parameters' contains event-specific settings
    - This function ensures consistency across the application
    """
    return {
        'element_name': element_name,           # Which element to affect
        'element_type': element_type,           # What type of element it is
        'element_category': element_category,   # Node or link
        'event_type': event_type,               # What to do to the element
        'scheduled_time': scheduled_time,       # When to apply (seconds)
        'time': scheduled_time,                 # For compatibility with batch events
        'parameters': parameters,               # Event-specific settings
        'description': f"{event_type} on {element_name}"  # Human-readable description
    }


def collect_simulation_data(wn: WaterNetworkModel, monitored_nodes: List[str], 
                          monitored_links: List[str], simulation_data: Dict) -> None:
    """
    Collect current pressure and flow data for monitored elements.
    
    This function extracts the current pressure and flow values from the
    network model and stores them in the simulation data structure for
    later visualization in charts.
    
    Args:
        wn (WaterNetworkModel): The network model with current values
        monitored_nodes (List[str]): List of node names to monitor
        monitored_links (List[str]): List of link names to monitor
        simulation_data (Dict): Data structure to store time-series data
        
    Data collection:
    - Builds the data that gets plotted in monitoring charts
    - Called after each simulation step to capture current values
    - Data is organized by element name for easy plotting
    - Pressure is measured in meters, flow in cubic meters per second
    """
    # Get the current number of time steps
    num_timesteps = len(simulation_data['time'])
    
    # Collect pressure data for monitored nodes
    for node_name in monitored_nodes:
        if node_name in wn.node_name_list:
            node = wn.get_node(node_name)
            
            # Initialize list for this node if first time
            if node_name not in simulation_data['pressures']:
                simulation_data['pressures'][node_name] = []
                # If this is a new element added mid-simulation, backfill with zeros
                # This ensures all data series have the same length for plotting
                simulation_data['pressures'][node_name] = [0.0] * (num_timesteps - 1)
            
            # Get current pressure (0 if not available)
            pressure = getattr(node, 'pressure', 0) if hasattr(node, 'pressure') else 0
            pressure = float(pressure) if pressure is not None else 0.0
            # Add to time series data
            simulation_data['pressures'][node_name].append(pressure)
    
    # Collect flow data for monitored links
    for link_name in monitored_links:
        if link_name in wn.link_name_list:
            link = wn.get_link(link_name)
            
            # Initialize list for this link if first time
            if link_name not in simulation_data['flows']:
                simulation_data['flows'][link_name] = []
                # If this is a new element added mid-simulation, backfill with zeros
                # This ensures all data series have the same length for plotting
                simulation_data['flows'][link_name] = [0.0] * (num_timesteps - 1)
            
            # Get current flow (0 if not available)
            flow = getattr(link, 'flow', 0) if hasattr(link, 'flow') else 0
            flow = float(flow) if flow is not None else 0.0
            # Add to time series data
            simulation_data['flows'][link_name].append(flow)


def get_current_element_values(wn: WaterNetworkModel, monitored_nodes: List[str], 
                             monitored_links: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Get the current pressure and flow values for display purposes.
    
    This function extracts the current values from monitored elements
    for display in the UI (like current values tables).
    
    Args:
        wn (WaterNetworkModel): The network model with current values
        monitored_nodes (List[str]): List of node names to get values for
        monitored_links (List[str]): List of link names to get values for
        
    Returns:
        Tuple[Dict[str, float], Dict[str, float]]: (Current pressures, Current flows)
        
    Display features:
    - Used for "current values" displays in the UI
    - Limited to 6 elements each to avoid cluttering the interface
    - Returns dictionaries mapping element names to current values
    """
    current_pressures = {}
    current_flows = {}
    
    # Get current pressures (limit to 6 for display purposes)
    for node_name in monitored_nodes[:6]:  # Limit to 6 for display
        if node_name in wn.node_name_list:
            node = wn.get_node(node_name)
            pressure = getattr(node, 'pressure', 0) if hasattr(node, 'pressure') else 0
            current_pressures[node_name] = pressure
    
    # Get current flows (limit to 6 for display purposes)
    for link_name in monitored_links[:6]:  # Limit to 6 for display
        if link_name in wn.link_name_list:
            link = wn.get_link(link_name)
            flow = getattr(link, 'flow', 0) if hasattr(link, 'flow') else 0
            current_flows[link_name] = flow
    
    return current_pressures, current_flows


def initialize_simulation_data() -> Dict[str, Any]:
    """
    Initialize an empty simulation data structure.
    
    This creates the data structure used to store time-series data
    during simulation for later plotting and analysis.
    
    Returns:
        Dict[str, Any]: Empty data structure with:
            - 'time': List of simulation times
            - 'pressures': Dict mapping node names to pressure lists
            - 'flows': Dict mapping link names to flow lists
            
    Data structure:
    - This structure grows as simulation progresses
    - Each simulation step adds new data points
    - Used by visualization functions to create charts
    """
    return {
        'time': [],          # List of simulation times (seconds)
        'pressures': {},     # Dict: node_name -> [pressure values over time]
        'flows': {}          # Dict: link_name -> [flow values over time]
    }


def reset_simulation_state(wn: WaterNetworkModel) -> InteractiveWNTRSimulator:
    """
    Reset the simulation by creating a fresh simulator instance.
    
    This function creates a new simulator with the original network
    configuration, effectively resetting all changes made during
    previous simulations.
    
    Args:
        wn (WaterNetworkModel): The network model to reset
        
    Returns:
        InteractiveWNTRSimulator: Fresh simulator instance
        
    Reset behavior:
    - Undoes all events and changes from previous simulations
    - Returns the network to its original state
    - Necessary when starting a new simulation scenario
    """
    return InteractiveWNTRSimulator(wn) 