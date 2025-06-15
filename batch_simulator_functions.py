import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import json

from mwntr.sim.interactive_network_simulator import MWNTRInteractiveSimulator
from mwntr.network import WaterNetworkModel, Node, Link, LinkStatus
from mwntr.network.elements import Junction, Tank, Reservoir, Pipe, Pump, Valve

def load_events_from_json(uploaded_file) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load events from uploaded JSON file."""
    try:
        content = uploaded_file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        data = json.loads(content)
        
        if 'events' in data and 'metadata' in data:
            return data['events'], data['metadata']
        else:
            # Assume the entire JSON is just a list of events
            return data, {}
    except Exception as e:
        st.error(f"Error loading JSON file: {str(e)}")
        return [], {}

def display_event_timeline(events: List[Dict[str, Any]], current_time: float):
    """Display a timeline of events with current position."""
    if not events:
        return
        
    # Create timeline visualization
    fig = go.Figure()
    
    # Group events by type for color coding
    event_types = {}
    for i, event in enumerate(events):
        event_type = event['event_type']
        if event_type not in event_types:
            event_types[event_type] = []
        event_types[event_type].append((event['time'], i, event['element_name']))
    
    # Color map for different event types
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    for i, (event_type, event_data) in enumerate(event_types.items()):
        times = [ed[0] for ed in event_data]
        indices = [ed[1] for ed in event_data]
        names = [ed[2] for ed in event_data]
        
        fig.add_trace(go.Scatter(
            x=times,
            y=[event_type] * len(times),
            mode='markers',
            marker=dict(
                size=8,
                color=colors[i % len(colors)],
                symbol='circle'
            ),
            text=[f"{event_type}<br>Element: {name}<br>Time: {t}s" for t, name in zip(times, names)],
            hovertemplate='%{text}<extra></extra>',
            name=event_type,
            showlegend=True
        ))
    
    # Add current time line
    if current_time > 0:
        fig.add_vline(
            x=current_time,
            line=dict(color="red", width=3, dash="dash"),
            annotation_text=f"Current Time: {current_time}s"
        )
    
    fig.update_layout(
        title="Event Timeline",
        xaxis_title="Time (seconds)",
        yaxis_title="Event Type",
        height=300,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def apply_event_to_batch_simulator(sim: MWNTRInteractiveSimulator, wn: WaterNetworkModel, event: Dict):
    """Apply an event to the batch simulator."""
    element_name = event['element_name']
    event_type = event['event_type']
    params = event['parameters']
    
    try:
        if event_type == 'start_leak':
            sim.start_leak(element_name, params.get('leak_area', 0.01), params.get('leak_discharge_coefficient', 0.75))
        elif event_type == 'stop_leak':
            sim.stop_leak(element_name)
        elif event_type == 'add_demand':
            sim.add_demand(element_name, params.get('base_demand', 0.1), params.get('pattern_name'), params.get('category'))
        elif event_type == 'remove_demand':
            sim.remove_demand(element_name, params.get('name'))
        elif event_type == 'close_pipe':
            sim.close_pipe(element_name)
        elif event_type == 'open_pipe':
            sim.open_pipe(element_name)
        elif event_type == 'close_pump':
            sim.close_pump(element_name)
        elif event_type == 'open_pump':
            sim.open_pump(element_name)
        elif event_type == 'close_valve':
            sim.close_valve(element_name)
        elif event_type == 'open_valve':
            sim.open_valve(element_name)
        elif event_type == 'set_tank_head':
            sim.set_tank_head(element_name, params.get('head', 50.0))
        elif event_type == 'set_pump_speed':
            sim.set_pump_speed(element_name, params.get('speed', 1.0))
        elif event_type == 'set_pipe_diameter':
            sim.set_pipe_diameter(element_name, params.get('diameter', 0.3))
        
        return True, f"Successfully applied {event_type} to {element_name}"
    except Exception as e:
        return False, f"Error applying {event_type} to {element_name}: {str(e)}"

def get_pending_events(events: List[Dict[str, Any]], current_time: float) -> List[Dict[str, Any]]:
    """Get events that should be applied at the current time."""
    return [e for e in events if e['time'] <= current_time and not e.get('applied', False)]

def batch_simulation_step(sim: MWNTRInteractiveSimulator, wn: WaterNetworkModel, 
                         events: List[Dict[str, Any]], current_time: float,
                         timestep: int = 60) -> Tuple[bool, float, str, List[Dict[str, Any]]]:
    """Run one step of batch simulation with event application."""
    applied_events = []
    
    try:
        # Apply any pending events
        pending_events = get_pending_events(events, current_time)
        
        for event in pending_events:
            success, message = apply_event_to_batch_simulator(sim, wn, event)
            if success:
                event['applied'] = True
                event['applied_at'] = current_time
                applied_events.append(event)
            else:
                return False, current_time, f"Event application failed: {message}", applied_events
        
        # Run simulation step
        if not sim.initialized_simulation:
            sim.init_simulation(global_timestep=timestep, duration=max([e['time'] for e in events]) + 3600)
        
        if not sim.is_terminated():
            sim.step_sim()
            new_time = sim.get_sim_time()
            return True, new_time, "Step completed successfully", applied_events
        else:
            return False, sim.get_sim_time(), "Simulation completed", applied_events
            
    except Exception as e:
        return False, current_time, f"Simulation error: {str(e)}", applied_events 