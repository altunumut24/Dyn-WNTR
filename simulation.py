"""
Simulation functions for Interactive Network Simulator.
Contains simulation logic, event handling, and network model functions.
"""

import json
import datetime
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st

from mwntr.sim.interactive_network_simulator import MWNTRInteractiveSimulator
from mwntr.network import WaterNetworkModel
from config import (
    INP_FILE, SIMULATION_DURATION_SECONDS, 
    HYDRAULIC_TIMESTEP_SECONDS, REPORT_TIMESTEP_SECONDS,
    NODE_EVENTS, LINK_EVENTS
)


def load_network_model(inp_file: str) -> Optional[WaterNetworkModel]:
    """Load the water network model from INP file."""
    try:
        wn = WaterNetworkModel(inp_file)
        wn.options.hydraulic.demand_model = 'PDD'
        wn.options.time.duration = SIMULATION_DURATION_SECONDS 
        wn.options.time.hydraulic_timestep = HYDRAULIC_TIMESTEP_SECONDS 
        wn.options.time.report_timestep = REPORT_TIMESTEP_SECONDS 
        return wn
    except FileNotFoundError:
        st.error(f"Error: INP file '{inp_file}' not found. Place it in the same directory as the script.")
        return None
    except Exception as e:
        st.error(f"Error loading INP file '{inp_file}': {e}")
        return None


def apply_event_to_simulator(sim: MWNTRInteractiveSimulator, wn: WaterNetworkModel, event: Dict) -> Tuple[bool, str]:
    """Apply an event to the simulator."""
    try:
        element_name = event['element_name']
        event_type = event['event_type']
        element_category = event['element_category']
        parameters = event.get('parameters', {})
        
        if element_category == 'node':
            node = wn.get_node(element_name)
            if event_type == 'start_leak':
                sim.start_leak(node, **parameters)
            elif event_type == 'stop_leak':
                sim.stop_leak(node)
            elif event_type == 'add_demand':
                base_demand = parameters.get('base_demand', 0.0)
                pattern_name = parameters.get('pattern_name', None)
                category = parameters.get('category', None)
                
                if pattern_name is None and category is not None:
                    pattern_name = category
                    if pattern_name not in wn.pattern_name_list:
                        wn.add_pattern(pattern_name, [1.0])
                
                sim.add_demand(node, base_demand, name=pattern_name, category=category)
            elif event_type == 'remove_demand':
                name = parameters.get('name', None)
                sim.remove_demand(node, name=name)
            elif event_type == 'add_fire_fighting_demand':
                sim.add_fire_fighting_demand(node, **parameters)
            elif event_type == 'set_tank_head':
                sim.set_tank_head(node, **parameters)
            else:
                return False, f"Unknown node event type: {event_type}"
                
        elif element_category == 'link':
            link = wn.get_link(element_name)
            if event_type == 'close_pipe':
                sim.close_pipe(link)
            elif event_type == 'open_pipe':
                sim.open_pipe(link)
            elif event_type == 'close_pump':
                sim.close_pump(link)
            elif event_type == 'open_pump':
                sim.open_pump(link)
            elif event_type == 'close_valve':
                sim.close_valve(link)
            elif event_type == 'open_valve':
                sim.open_valve(link)
            elif event_type == 'set_pipe_diameter':
                sim.set_pipe_diameter(link, **parameters)
            elif event_type == 'set_pump_speed':
                sim.set_pump_speed(link, **parameters)
            elif event_type == 'set_pump_head_curve':
                sim.set_pump_head_curve(link, **parameters)
            else:
                return False, f"Unknown link event type: {event_type}"
        else:
            return False, f"Unknown element category: {element_category}"
            
        return True, f"Applied {event_type} to {element_name}"
        
    except Exception as e:
        return False, f"Error applying event: {str(e)}"


def run_simulation_step(sim: MWNTRInteractiveSimulator, wn: WaterNetworkModel) -> Tuple[bool, float, str]:
    """Run a single simulation step."""
    try:
        if sim.is_terminated():
            return False, sim.get_sim_time(), "Simulation terminated"
        
        # Execute the next hydraulic timestep
        sim.step_sim()
        
        current_time = sim.get_sim_time()
        
        # Update network state with current results
        results = sim.get_results()
        current_time_step = current_time
        
        # Update node pressures
        if hasattr(results, 'node') and 'pressure' in results.node:
            pressure_data = results.node['pressure']
            if current_time_step in pressure_data.index:
                for node_name in pressure_data.columns:
                    if node_name in wn.node_name_list:
                        node = wn.get_node(node_name)
                        node.pressure = pressure_data.loc[current_time_step, node_name]
        
        # Update link flows
        if hasattr(results, 'link') and 'flowrate' in results.link:
            flow_data = results.link['flowrate']
            if current_time_step in flow_data.index:
                for link_name in flow_data.columns:
                    if link_name in wn.link_name_list:
                        link = wn.get_link(link_name)
                        link.flow = flow_data.loc[current_time_step, link_name]
        
        return True, current_time, f"Simulation step completed at {datetime.timedelta(seconds=int(current_time))}"
        
    except Exception as e:
        return False, sim.get_sim_time(), f"Simulation error: {str(e)}"


def load_events_from_json(uploaded_file) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load events from uploaded JSON file."""
    try:
        content = uploaded_file.getvalue()
        data = json.loads(content)
        
        events = data.get('events', [])
        metadata = data.get('metadata', {})
        
        return events, metadata
        
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON format: {e}")
        return [], {}
    except Exception as e:
        st.error(f"Error loading events: {e}")
        return [], {}


def get_available_events(element_type: str, element_category: str) -> Dict[str, Dict]:
    """Get available events for an element type and category."""
    if element_category == 'node':
        return NODE_EVENTS.get(element_type, {})
    elif element_category == 'link':
        return LINK_EVENTS.get(element_type, {})
    else:
        return {}


def create_event(element_name: str, element_type: str, element_category: str, 
                event_type: str, scheduled_time: float, parameters: Dict) -> Dict[str, Any]:
    """Create a new event dictionary."""
    return {
        'element_name': element_name,
        'element_type': element_type,
        'element_category': element_category,
        'event_type': event_type,
        'scheduled_time': scheduled_time,
        'time': scheduled_time,  # For compatibility with batch events
        'parameters': parameters,
        'description': f"{event_type} on {element_name}"
    }


def collect_simulation_data(wn: WaterNetworkModel, monitored_nodes: List[str], 
                          monitored_links: List[str], simulation_data: Dict) -> None:
    """Collect pressure and flow data for monitored elements."""
    current_time = simulation_data['time'][-1] if simulation_data['time'] else 0
    
    # Collect pressure data for monitored nodes
    for node_name in monitored_nodes:
        if node_name in wn.node_name_list:
            node = wn.get_node(node_name)
            if node_name not in simulation_data['pressures']:
                simulation_data['pressures'][node_name] = []
            pressure = getattr(node, 'pressure', 0) if hasattr(node, 'pressure') else 0
            simulation_data['pressures'][node_name].append(pressure)
    
    # Collect flow data for monitored links
    for link_name in monitored_links:
        if link_name in wn.link_name_list:
            link = wn.get_link(link_name)
            if link_name not in simulation_data['flows']:
                simulation_data['flows'][link_name] = []
            flow = getattr(link, 'flow', 0) if hasattr(link, 'flow') else 0
            simulation_data['flows'][link_name].append(flow)


def get_current_element_values(wn: WaterNetworkModel, monitored_nodes: List[str], 
                             monitored_links: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Get current pressure and flow values for monitored elements."""
    current_pressures = {}
    current_flows = {}
    
    for node_name in monitored_nodes[:6]:  # Limit to 6 for display
        if node_name in wn.node_name_list:
            node = wn.get_node(node_name)
            pressure = getattr(node, 'pressure', 0) if hasattr(node, 'pressure') else 0
            current_pressures[node_name] = pressure
    
    for link_name in monitored_links[:6]:  # Limit to 6 for display
        if link_name in wn.link_name_list:
            link = wn.get_link(link_name)
            flow = getattr(link, 'flow', 0) if hasattr(link, 'flow') else 0
            current_flows[link_name] = flow
    
    return current_pressures, current_flows


def initialize_simulation_data() -> Dict[str, Any]:
    """Initialize empty simulation data structure."""
    return {
        'time': [],
        'pressures': {},
        'flows': {}
    }


def reset_simulation_state(wn: WaterNetworkModel) -> MWNTRInteractiveSimulator:
    """Reset and create a new simulator instance."""
    return MWNTRInteractiveSimulator(wn) 