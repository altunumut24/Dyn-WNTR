import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


from mwntr.sim.interactive_network_simulator import MWNTRInteractiveSimulator
from mwntr.network import WaterNetworkModel, Node, Link, LinkStatus
from mwntr.network.elements import Junction, Tank, Reservoir, Pipe, Pump, Valve
from mwntr.graphics.network import plot_interactive_network
MWNTR_AVAILABLE = True

# Configuration
INP_FILE = 'NET_4.inp'
SIMULATION_DURATION_SECONDS = 2 * 3600  # 2 hours
HYDRAULIC_TIMESTEP_SECONDS = 60  # 1 minute steps
REPORT_TIMESTEP_SECONDS = 60

# Define available events for each element type
NODE_EVENTS = {
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
    },
    'Reservoir': {
        # Reservoirs typically don't have events in this context
    }
}

LINK_EVENTS = {
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

def get_pressure_color(pressure: float, min_pressure: float, max_pressure: float) -> str:
    """Generate continuous color based on pressure value using viridis-like colorscale."""
    if max_pressure == min_pressure:
        return 'rgb(33, 145, 140)'  # Default teal color if no variation
    
    # Normalize pressure to 0-1 range
    normalized = (pressure - min_pressure) / (max_pressure - min_pressure)
    normalized = max(0, min(1, normalized))  # Clamp to [0,1]
    
    # Create color gradient similar to reference image: Purple (low) -> Blue -> Teal -> Green -> Yellow (high)
    if normalized <= 0.2:
        # Dark purple to blue
        t = normalized / 0.2
        r = int(68 + (59 - 68) * t)
        g = int(1 + (82 - 1) * t)
        b = int(84 + (139 - 84) * t)
    elif normalized <= 0.4:
        # Blue to teal
        t = (normalized - 0.2) / 0.2
        r = int(59 + (33 - 59) * t)
        g = int(82 + (145 - 82) * t)
        b = int(139 + (140 - 139) * t)
    elif normalized <= 0.6:
        # Teal to green
        t = (normalized - 0.4) / 0.2
        r = int(33 + (94 - 33) * t)
        g = int(145 + (201 - 145) * t)
        b = int(140 + (98 - 140) * t)
    elif normalized <= 0.8:
        # Green to yellow
        t = (normalized - 0.6) / 0.2
        r = int(94 + (253 - 94) * t)
        g = int(201 + (231 - 201) * t)
        b = int(98 + (37 - 98) * t)
    else:
        # Yellow to white
        t = (normalized - 0.8) / 0.2
        r = int(253 + (255 - 253) * t)
        g = int(231 + (255 - 231) * t)
        b = int(37 + (255 - 37) * t)
    
    return f'rgb({r},{g},{b})'

def get_flow_color_and_width(flow: float, min_flow: float, max_flow: float) -> Tuple[str, float]:
    """Generate color and width based on flow value using viridis-like colorscale."""
    abs_flow = abs(flow)
    max_abs_flow = max(abs(min_flow), abs(max_flow))
    
    if max_abs_flow == 0:
        return 'rgb(255, 255, 255)', 2  # White if no flow
    
    # Width based on absolute flow (1-10 range)
    width = 1 + 9 * (abs_flow / max_abs_flow)
    
    # Normalize flow to 0-1 range for color mapping
    if max_flow == min_flow:
        normalized = 0.5  # Middle value if no variation
    else:
        normalized = (flow - min_flow) / (max_flow - min_flow)
        normalized = max(0, min(1, normalized))  # Clamp to [0,1]
    
    # Create color gradient: Purple (reverse) -> Teal (no flow) -> Green -> Yellow (forward)
    if normalized <= 0.3:
        # Purple to teal (reverse flow)
        t = normalized / 0.3
        r = int(68 + (33 - 68) * t)
        g = int(1 + (145 - 1) * t)
        b = int(84 + (140 - 84) * t)
    elif normalized <= 0.7:
        # Teal to green (low to medium flow)
        t = (normalized - 0.3) / 0.4
        r = int(33 + (94 - 33) * t)
        g = int(145 + (201 - 145) * t)
        b = int(140 + (98 - 140) * t)
    else:
        # Green to yellow (high flow)
        t = (normalized - 0.7) / 0.3
        r = int(94 + (253 - 94) * t)
        g = int(201 + (231 - 201) * t)
        b = int(98 + (37 - 98) * t)
    
    color = f'rgb({r},{g},{b})'
    return color, width

def create_pressure_colorbar(min_pressure: float, max_pressure: float) -> go.Figure:
    """Create horizontal pressure colorbar with viridis-like colors."""
    if max_pressure == min_pressure:
        # Handle case where there's no pressure variation
        pressure_range = np.array([min_pressure - 1, max_pressure + 1])
    else:
        pressure_range = np.linspace(min_pressure, max_pressure, 100)
    
    colors = [get_pressure_color(p, min_pressure, max_pressure) for p in pressure_range]
    
    fig = go.Figure()
    
    # Create colorbar using scatter plot
    fig.add_trace(go.Scatter(
        x=pressure_range,
        y=[0] * len(pressure_range),
        mode='markers',
        marker=dict(
            size=15,
            color=colors,
            line=dict(width=0)
        ),
        showlegend=False,
        hovertemplate='Pressure: %{x:.2f} m<extra></extra>'
    ))
    
    fig.update_layout(
        title="Node Pressure Scale (m)",
        title_x=0.5,
        xaxis=dict(
            title="Pressure (m)",
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            range=[-0.5, 0.5]
        ),
        height=100,
        margin=dict(l=50, r=50, t=50, b=40)
    )
    
    return fig

def create_flow_colorbar(min_flow: float, max_flow: float) -> go.Figure:
    """Create horizontal flow colorbar with viridis-like colors."""
    if max_flow == min_flow:
        # Handle case where there's no flow variation
        flow_range = np.array([min_flow - 0.01, max_flow + 0.01])
    else:
        flow_range = np.linspace(min_flow, max_flow, 100)
    
    colors = []
    widths = []
    for f in flow_range:
        color, width = get_flow_color_and_width(f, min_flow, max_flow)
        colors.append(color)
        widths.append(width)
    
    fig = go.Figure()
    
    # Create colorbar using scatter plot
    fig.add_trace(go.Scatter(
        x=flow_range,
        y=[0] * len(flow_range),
        mode='markers',
        marker=dict(
            size=[w + 5 for w in widths],  # Adjust size for visibility
            color=colors,
            line=dict(width=0)
        ),
        showlegend=False,
        hovertemplate='Flow: %{x:.4f} m³/s<extra></extra>'
    ))
    
    fig.update_layout(
        title="Link Flow Scale (m³/s) - Color: Direction, Size: Magnitude",
        title_x=0.5,
        xaxis=dict(
            title="Flow (m³/s)",
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            range=[-0.5, 0.5]
        ),
        height=100,
        margin=dict(l=50, r=50, t=50, b=40)
    )
    
    return fig

def get_network_layout(wn: WaterNetworkModel) -> Tuple[Dict, List, Dict, Dict]:
    """Extract network layout information for plotting."""
    # Get node positions
    node_positions = {}
    for node_name, node in wn.nodes():
        if node.coordinates is not None:
            node_positions[node_name] = node.coordinates
        else:
            # Generate random positions if coordinates not available
            node_positions[node_name] = (np.random.random(), np.random.random())
    
    # Get edge list
    edge_list = []
    edge_info = {}
    for link_name, link in wn.links():
        start_pos = node_positions[link.start_node_name]
        end_pos = node_positions[link.end_node_name]
        edge_list.append((link.start_node_name, link.end_node_name))
        edge_info[link_name] = {
            'start': link.start_node_name,
            'end': link.end_node_name,
            'type': link.link_type,
            'mid_pos': ((start_pos[0] + end_pos[0])/2, (start_pos[1] + end_pos[1])/2)
        }
    
    # Get node info
    node_info = {}
    for node_name, node in wn.nodes():
        node_info[node_name] = {
            'type': node.node_type,
            'pos': node_positions[node_name]
        }
    
    return node_positions, edge_list, node_info, edge_info

def create_network_plot(wn: WaterNetworkModel, selected_nodes: List[str] = None, selected_links: List[str] = None, 
                        show_simulation_data: bool = False, sim_initialized: bool = False):
    """Create interactive plotly network visualization with click handling and optional simulation data overlay."""
    node_positions, edge_list, node_info, edge_info = get_network_layout(wn)
    
    if selected_nodes is None:
        selected_nodes = []
    if selected_links is None:
        selected_links = []
    
    fig = go.Figure()
    
    # Get current simulation data if available
    node_pressures = {}
    link_flows = {}
    
    if show_simulation_data and sim_initialized:
        # Collect current pressures
        for node_name, node in wn.nodes():
            try:
                pressure = getattr(node, 'pressure', 0) if hasattr(node, 'pressure') else 0
                # Ensure we have a valid numeric value
                node_pressures[node_name] = float(pressure) if pressure is not None else 0.0
            except:
                node_pressures[node_name] = 0.0
        
        # Collect current flows
        for link_name, link in wn.links():
            try:
                flow = getattr(link, 'flow', 0) if hasattr(link, 'flow') else 0
                # Ensure we have a valid numeric value
                link_flows[link_name] = float(flow) if flow is not None else 0.0
            except:
                link_flows[link_name] = 0.0
    
    # Calculate flow ranges for color scaling
    if link_flows:
        # Filter out None values and convert to float
        flow_values = [float(f) for f in link_flows.values() if f is not None]
        if flow_values:
            min_flow = min(flow_values)
            max_flow = max(flow_values)
        else:
            min_flow = max_flow = 0
    else:
        min_flow = max_flow = 0
    
    # Add edges (connect nodes with lines)
    for link_name, info in edge_info.items():
        start_pos = node_positions[info['start']]
        end_pos = node_positions[info['end']]
        
        # Determine color and width based on simulation data or defaults
        if show_simulation_data and link_name in link_flows:
            flow = link_flows[link_name]
            color, width = get_flow_color_and_width(flow, min_flow, max_flow)
            if link_name in selected_links:
                width = max(width + 3, 8)  # Make selected links thicker
                color = 'red'  # Override with selection color
        else:
            # Default colors (thinner lines)
            if link_name in selected_links:
                color = 'red'
                width = 4
            else:
                if info['type'] == 'Pipe':
                    color = 'darkblue'
                elif info['type'] == 'Pump':
                    color = 'green'
                elif info['type'] == 'Valve':
                    color = 'orange'
                else:
                    color = 'gray'
                width = 2
        
        # Create hover text
        hover_text = f"<b>{link_name}</b><br>Type: {info['type']}"
        if show_simulation_data and link_name in link_flows:
            hover_text += f"<br>Flow: {link_flows[link_name]:.4f} m³/s"
        
        fig.add_trace(go.Scatter(
            x=[start_pos[0], end_pos[0]],
            y=[start_pos[1], end_pos[1]],
            mode='lines',
            line=dict(color=color, width=width),
            name=f"{info['type']}s",
            legendgroup=info['type'],
            showlegend=link_name == list(edge_info.keys())[0] or info['type'] not in [edge_info[k]['type'] for k in list(edge_info.keys())[:list(edge_info.keys()).index(link_name)]],
            hovertemplate=hover_text + "<extra></extra>"
        ))
    
    # Add clickable markers at link midpoints for link selection
    link_x = []
    link_y = []
    link_names = []
    link_types = []
    link_customdata = []
    link_symbols = []
    link_colors = []
    
    for link_name, info in edge_info.items():
        link_x.append(info['mid_pos'][0])
        link_y.append(info['mid_pos'][1])
        link_names.append(link_name)
        link_types.append(info['type'])
        # Store name and type as customdata for click detection
        link_customdata.append([link_name, info['type'], 'link'])
        
        # Professional symbols for different link types (smaller and less prominent)
        if info['type'] == 'Pipe':
            link_symbols.append('square')
            color = 'red' if link_name in selected_links else 'rgba(173, 216, 230, 0.7)'  # Semi-transparent
        elif info['type'] == 'Pump':
            link_symbols.append('triangle-up')
            color = 'red' if link_name in selected_links else 'rgba(0, 128, 0, 0.7)'  # Semi-transparent
        elif info['type'] == 'Valve':
            link_symbols.append('diamond')
            color = 'red' if link_name in selected_links else 'rgba(255, 165, 0, 0.7)'  # Semi-transparent
        else:
            link_symbols.append('square')
            color = 'red' if link_name in selected_links else 'rgba(128, 128, 128, 0.7)'  # Semi-transparent
        
        link_colors.append(color)
    
    fig.add_trace(go.Scatter(
        x=link_x,
        y=link_y,
        mode='markers+text',
        marker=dict(
            size=18,  # Increased from 14 for better visibility
            color=link_colors,
            line=dict(width=2, color='darkblue'),  # Increased border width
            symbol=link_symbols
        ),
        text=link_names,
        textfont=dict(size=8, color='black', family="Arial"),  # Increased from 6
        # CRITICAL: Store name, type, and category in customdata
        customdata=link_customdata,
        hovertemplate="<b>%{customdata[0]}</b><br>Type: %{customdata[1]}<br>Click to select<extra></extra>",
        name='Links (clickable)',
        showlegend=False
    ))
    
    # Calculate pressure ranges for color scaling
    if node_pressures:
        # Filter out None values and convert to float
        pressure_values = [float(p) for p in node_pressures.values() if p is not None]
        if pressure_values:
            min_pressure = min(pressure_values)
            max_pressure = max(pressure_values)
        else:
            min_pressure = max_pressure = 0
    else:
        min_pressure = max_pressure = 0
    
    # Add nodes with proper customdata and professional styling
    node_x = []
    node_y = []
    node_names = []
    node_colors = []
    node_customdata = []
    node_hover_text = []
    node_symbols = []
    node_sizes = []
    
    for node_name, info in node_info.items():
        node_x.append(info['pos'][0])
        node_y.append(info['pos'][1])
        node_names.append(node_name)
        # Store name, type, and category as customdata for click detection
        node_customdata.append([node_name, info['type'], 'node'])
        
        # Professional symbols and sizes for different node types (increased for better visibility)
        if info['type'] == 'Junction':
            symbol = 'circle'
            size = 25  # Increased from 20
            default_color = 'lightblue'
        elif info['type'] == 'Tank':
            symbol = 'square'
            size = 30  # Increased from 24
            default_color = 'navy'
        elif info['type'] == 'Reservoir':
            symbol = 'hexagon'
            size = 28  # Increased from 22
            default_color = 'darkgreen'
        else:
            symbol = 'circle'
            size = 23  # Increased from 18
            default_color = 'gray'
        
        node_symbols.append(symbol)
        node_sizes.append(size)
        
        # Determine color based on simulation data or defaults
        if node_name in selected_nodes:
            color = 'red'
        elif show_simulation_data and node_name in node_pressures:
            pressure = node_pressures[node_name]
            color = get_pressure_color(pressure, min_pressure, max_pressure)
        else:
            color = default_color
        
        node_colors.append(color)
        
        # Create hover text
        hover_text = f"<b>{node_name}</b><br>Type: {info['type']}"
        if show_simulation_data and node_name in node_pressures:
            hover_text += f"<br>Pressure: {node_pressures[node_name]:.2f} m"
        hover_text += "<br>Click to select"
        node_hover_text.append(hover_text)
    
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes, 
            color=node_colors, 
            line=dict(width=2, color='darkblue'),
            symbol=node_symbols
        ),
        text=node_names,
        textposition='middle center',
        textfont=dict(size=9, color='white', family="Arial Black"),
        # CRITICAL: Store name, type, and category in customdata
        customdata=node_customdata,
        hovertemplate='%{hovertext}<extra></extra>',
        hovertext=node_hover_text,
        name='Nodes (clickable)',
        showlegend=False
    ))
    
    # Create title based on visualization mode
    if show_simulation_data and sim_initialized:
        title = "🌊 Live Network Simulation - Pressure (Node Colors) & Flow (Link Width/Color)"
        if node_pressures:
            title += f"<br><sub>Pressure Range: {min_pressure:.1f} - {max_pressure:.1f} m | Flow Range: {min_flow:.3f} - {max_flow:.3f} m³/s</sub>"
    else:
        title = "Interactive Water Network - Click on elements to select them"
    
    fig.update_layout(
        title=title,
        showlegend=True,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=800  # Increased from 600 for better visibility
    )
    
    return fig

def display_element_properties(wn: WaterNetworkModel, element_name: str, element_type: str):
    """Display properties of selected element."""
    if element_type in ['Junction', 'Tank', 'Reservoir']:
        element = wn.get_node(element_name)
        st.subheader(f"Node: {element_name}")
        st.write(f"**Type:** {element.node_type}")
        
        if hasattr(element, 'elevation'):
            st.write(f"**Elevation:** {element.elevation} m")
        if hasattr(element, 'base_demand') and element.node_type == 'Junction':
            st.write(f"**Base Demand:** {element.base_demand} m³/s")
        if hasattr(element, 'init_level') and element.node_type == 'Tank':
            st.write(f"**Initial Level:** {element.init_level} m")
            st.write(f"**Diameter:** {element.diameter} m")
        if hasattr(element, 'base_head') and element.node_type == 'Reservoir':
            st.write(f"**Base Head:** {element.base_head} m")
            
    else:  # Link
        element = wn.get_link(element_name)
        st.subheader(f"Link: {element_name}")
        st.write(f"**Type:** {element.link_type}")
        st.write(f"**Start Node:** {element.start_node_name}")
        st.write(f"**End Node:** {element.end_node_name}")
        
        if hasattr(element, 'length'):
            st.write(f"**Length:** {element.length} m")
        if hasattr(element, 'diameter'):
            st.write(f"**Diameter:** {element.diameter} m")
        if hasattr(element, 'roughness'):
            st.write(f"**Roughness:** {element.roughness}")

def display_event_interface(element_name: str, element_type: str, element_category: str):
    """Display compact event configuration interface with expandable parameters."""
    
    # Get available events based on element type
    if element_category == 'node':
        available_events = NODE_EVENTS.get(element_type, {})
    else:
        available_events = LINK_EVENTS.get(element_type, {})
    
    if not available_events:
        st.info(f"No events available for {element_type}")
        return None
    
    # Event selection - more compact
    event_names = list(available_events.keys())
    selected_event = st.selectbox(
        "Event Type:", 
        event_names, 
        key=f"event_{element_name}",
        help=f"Choose event for {element_name}"
    )
    
    if selected_event:
        event_config = available_events[selected_event]
        params = event_config['params']
        defaults = event_config['defaults']
        
        # Quick add button with defaults
        if st.button(f"➕ Add {selected_event}", key=f"quick_add_{element_name}", use_container_width=True):
            # Use default values
            param_values = {}
            for i, param in enumerate(params):
                default_val = defaults[i] if i < len(defaults) else None
                if default_val is not None:
                    param_values[param] = default_val
            
            return {
                'element_name': element_name,
                'element_type': element_type,
                'element_category': element_category,
                'event_type': selected_event,
                'parameters': param_values,
                'scheduled_time': st.session_state.get('current_sim_time', 0)
            }
        
        # Expandable parameter customization
        if params:  # Only show if there are parameters to customize
            with st.expander("🔧 Customize Parameters", expanded=False):
                st.markdown("*Adjust parameters if needed (defaults are already set)*")
                
                param_values = {}
                for i, param in enumerate(params):
                    default_val = defaults[i] if i < len(defaults) else None
                    
                    if param in ['leak_area', 'diameter', 'base_demand', 'fire_flow_demand', 'head', 'speed']:
                        param_values[param] = st.number_input(
                            f"{param.replace('_', ' ').title()}:",
                            value=float(default_val) if default_val is not None else 0.0,
                            step=0.01,
                            key=f"{param}_custom_{element_name}"
                        )
                    elif param in ['leak_discharge_coefficient']:
                        param_values[param] = st.slider(
                            f"{param.replace('_', ' ').title()}:",
                            min_value=0.1, max_value=1.0, 
                            value=float(default_val) if default_val is not None else 0.75,
                            step=0.05,
                            key=f"{param}_custom_{element_name}"
                        )
                    elif param in ['fire_start', 'fire_end']:
                        param_values[param] = st.number_input(
                            f"{param.replace('_', ' ').title()} (seconds):",
                            value=int(default_val) if default_val is not None else 0,
                            step=60,
                            key=f"{param}_custom_{element_name}"
                        )
                    else:
                        param_values[param] = st.text_input(
                            f"{param.replace('_', ' ').title()}:",
                            value=str(default_val) if default_val is not None else "",
                            key=f"{param}_custom_{element_name}"
                        )
                
                # Custom add button
                if st.button(f"➕ Add {selected_event} (Custom)", key=f"custom_add_{element_name}", use_container_width=True):
                    return {
                        'element_name': element_name,
                        'element_type': element_type,
                        'element_category': element_category,
                        'event_type': selected_event,
                        'parameters': param_values,
                        'scheduled_time': st.session_state.get('current_sim_time', 0)
                    }
    
    return None

def apply_event_to_simulator(sim: MWNTRInteractiveSimulator, wn: WaterNetworkModel, event: Dict):
    """Apply an event to the simulator."""
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

def run_simulation_step(sim: MWNTRInteractiveSimulator, wn: WaterNetworkModel):
    """Run one simulation step."""
    try:
        if not sim.initialized_simulation:
            sim.init_simulation(global_timestep=HYDRAULIC_TIMESTEP_SECONDS, duration=SIMULATION_DURATION_SECONDS)
        
        if not sim.is_terminated():
            sim.step_sim()
            return True, sim.get_sim_time(), "Step completed successfully"
        else:
            return False, sim.get_sim_time(), "Simulation completed"
    except Exception as e:
        return False, sim.get_sim_time(), f"Simulation error: {str(e)}"



# Main Streamlit App
st.set_page_config(
    layout="wide", 
    page_title="Interactive Network Event Simulator",
    page_icon="🔧",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #1565c0, #2e7d32);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-1px);
    }
    
    /* Metric styling */
    .metric-container {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-left: 4px solid #2196f3;
        border-radius: 8px;
    }
    
    /* Success boxes */
    .stSuccess {
        background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
        border-left: 4px solid #4caf50;
        border-radius: 8px;
    }
    
    /* Warning boxes */
    .stWarning {
        background: linear-gradient(135deg, #fff3e0, #ffe0b2);
        border-left: 4px solid #ff9800;
        border-radius: 8px;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        border: 2px solid #e0e0e0;
        border-radius: 8px;
    }
    
    /* Toggle styling */
    .stToggle > div {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border-radius: 20px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f1f3f4, #e8eaed);
        border-radius: 8px;
        border: 1px solid #dadce0;
    }
    
    /* Container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Section headers */
    .section-header {
        color: #1565c0;
        border-bottom: 2px solid #e3f2fd;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🔧 Interactive Water Network Event Simulator</h1>', unsafe_allow_html=True)

# Initialize session state
if 'wn' not in st.session_state:
    st.session_state.wn = None
if 'sim' not in st.session_state:
    st.session_state.sim = None
if 'selected_nodes' not in st.session_state:
    st.session_state.selected_nodes = []
if 'selected_links' not in st.session_state:
    st.session_state.selected_links = []
if 'current_element' not in st.session_state:
    st.session_state.current_element = None
if 'scheduled_events' not in st.session_state:
    st.session_state.scheduled_events = []
if 'applied_events' not in st.session_state:
    st.session_state.applied_events = []
if 'current_sim_time' not in st.session_state:
    st.session_state.current_sim_time = 0
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = {'time': [], 'pressures': {}, 'flows': {}}

# Load network with better UI feedback
if st.session_state.wn is None:
    with st.spinner("🔄 Loading network model..."):
        st.session_state.wn = load_network_model(INP_FILE)
        if st.session_state.wn:
            st.session_state.sim = MWNTRInteractiveSimulator(st.session_state.wn)
            st.success(f"✅ Network '{INP_FILE}' loaded successfully!")
            time.sleep(1)  # Brief pause to show success message
        else:
            st.stop()

wn = st.session_state.wn
sim = st.session_state.sim

# Network Status Dashboard
with st.container():
    st.markdown('<h3 class="section-header">📊 Network Status</h3>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Nodes", len(wn.node_name_list), help="Junction, Tank, and Reservoir nodes")
    with col2:
        st.metric("Total Links", len(wn.link_name_list), help="Pipes, Pumps, and Valves")
    with col3:
        if st.session_state.current_element:
            st.metric("Selected Element", st.session_state.current_element['name'], 
                     delta=st.session_state.current_element['type'], delta_color="normal")
        else:
            st.metric("Selected Element", "None", help="Click on network elements to select")
    with col4:
        st.metric("Scheduled Events", len(st.session_state.scheduled_events), 
                 delta=len(st.session_state.applied_events), delta_color="normal")

st.markdown("---")

# Interactive Network Visualization with Side Panel
st.markdown('<h3 class="section-header">🗺️ Interactive Network Map & Event Configuration</h3>', unsafe_allow_html=True)

# Main layout: Network map on left, controls on right
map_col, control_col = st.columns([3, 1])

with map_col:
    # Instructions and controls above the map
    st.info("🖱️ **Click on nodes (circles) or links (squares) to select and configure events**")
    
    # Visualization controls
    viz_col1, viz_col2, viz_col3 = st.columns([2, 1, 1])
    with viz_col1:
        pass  # Reserved for future controls
    with viz_col2:
        show_sim_data = st.toggle(
            "🌊 Live Visualization", 
            value=sim.initialized_simulation,
            disabled=not sim.initialized_simulation,
            help="Show real-time pressure and flow data"
        )
    with viz_col3:
        if st.button("🧭 Legend", help="Show network element types"):
            with st.expander("🎨 Network Legend", expanded=True):
                st.markdown("""
                **Nodes:** 🔵 Junctions | 🔷 Tanks | 🟢 Reservoirs  
                **Links:** ▬ Pipes | ▬ Pumps | ▬ Valves  
                **Selected:** 🔴 Red highlights
                """)

    # Network plot
    fig = create_network_plot(
        wn, 
        st.session_state.selected_nodes, 
        st.session_state.selected_links,
        show_simulation_data=show_sim_data,
        sim_initialized=sim.initialized_simulation
    )
    
    # Display plot with click handling
    chart_data = st.plotly_chart(
        fig, 
        use_container_width=True, 
        key="network_plot_interaction",
        on_select="rerun",
        selection_mode="points"
    )
    
    # Color scale legends below the map
    if show_sim_data and sim.initialized_simulation:
        st.markdown('<h4 class="section-header">🎨 Color Scales</h4>', unsafe_allow_html=True)
        
        # Get current data for legends (simplified)
        node_pressures = {}
        link_flows = {}
        
        for node_name, node in wn.nodes():
            try:
                pressure = getattr(node, 'pressure', 0) if hasattr(node, 'pressure') else 0
                node_pressures[node_name] = float(pressure) if pressure is not None else 0.0
            except:
                node_pressures[node_name] = 0.0
        
        for link_name, link in wn.links():
            try:
                flow = getattr(link, 'flow', 0) if hasattr(link, 'flow') else 0
                link_flows[link_name] = float(flow) if flow is not None else 0.0
            except:
                link_flows[link_name] = 0.0
        
        # Calculate ranges
        if node_pressures:
            pressure_values = [float(p) for p in node_pressures.values() if p is not None]
            if pressure_values:
                min_pressure = min(pressure_values)
                max_pressure = max(pressure_values)
            else:
                min_pressure = max_pressure = 0
        else:
            min_pressure = max_pressure = 0
            
        if link_flows:
            flow_values = [float(f) for f in link_flows.values() if f is not None]
            if flow_values:
                min_flow = min(flow_values)
                max_flow = max(flow_values)
            else:
                min_flow = max_flow = 0
        else:
            min_flow = max_flow = 0
        
        # Display legends side by side
        legend_col1, legend_col2 = st.columns(2)
        
        with legend_col1:
            if node_pressures and (min_pressure != max_pressure or min_pressure != 0):
                pressure_fig = create_pressure_colorbar(min_pressure, max_pressure)
                st.plotly_chart(pressure_fig, use_container_width=True)
            else:
                st.info("🔵 **Pressure**: No data yet")
        
        with legend_col2:
            if link_flows and (min_flow != max_flow or min_flow != 0):
                flow_fig = create_flow_colorbar(min_flow, max_flow)
                st.plotly_chart(flow_fig, use_container_width=True)
            else:
                st.info("🔗 **Flow**: No data yet")

# Process click events from the network plot
if chart_data and chart_data.get('selection') and chart_data['selection']['points']:
    selected_point = chart_data['selection']['points'][0]
    
    if selected_point.get('customdata'):
        custom_data = selected_point['customdata']
        element_name = custom_data[0]
        element_type = custom_data[1]
        element_category = custom_data[2]  # 'node' or 'link'
        
        # Update session state only if it's a new selection
        if not st.session_state.current_element or st.session_state.current_element['name'] != element_name:
            st.session_state.current_element = {
                'name': element_name,
                'category': element_category,
                'type': element_type
            }
            
            if element_category == 'node':
                st.session_state.selected_nodes = [element_name]
                st.session_state.selected_links = []
            else:  # link
                st.session_state.selected_links = [element_name]
                st.session_state.selected_nodes = []
            
            st.rerun()

with control_col:
    st.markdown('<h3 class="section-header">🎮 Controls</h3>', unsafe_allow_html=True)
    
    # Simulation controls at the top of right panel
    st.markdown('<h4 class="section-header">⚡ Simulation</h4>', unsafe_allow_html=True)
    
    # Status display
    if sim.initialized_simulation:
        current_time = datetime.timedelta(seconds=int(st.session_state.current_sim_time))
        st.success(f"🟢 **Active** | Time: {current_time}")
    else:
        st.warning("🟡 **Not Initialized**")
    
    # Control buttons
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        if st.button("🚀 Initialize", disabled=sim.initialized_simulation, use_container_width=True):
            try:
                with st.spinner("Initializing..."):
                    sim.init_simulation(global_timestep=HYDRAULIC_TIMESTEP_SECONDS, duration=SIMULATION_DURATION_SECONDS)
                    st.session_state.current_sim_time = 0
                st.success("✅ Ready!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed: {e}")
    
    with sim_col2:
        step_disabled = not sim.initialized_simulation or sim.is_terminated()
        if st.button("⏭️ Step", disabled=step_disabled, use_container_width=True):
            with st.spinner("Processing..."):
                # Apply scheduled events
                events_to_apply = [e for e in st.session_state.scheduled_events if e['scheduled_time'] <= st.session_state.current_sim_time]
                
                for event in events_to_apply:
                    success, message = apply_event_to_simulator(sim, wn, event)
                    if success:
                        st.session_state.applied_events.append(event)
                        st.success(f"⚡ {message}")
                    else:
                        st.error(f"❌ {message}")
                    st.session_state.scheduled_events.remove(event)
                
                # Run simulation step
                success, sim_time, message = run_simulation_step(sim, wn)
                st.session_state.current_sim_time = sim_time
                
                if success:
                    # Collect data
                    st.session_state.simulation_data['time'].append(sim_time)
                    
                    # Sample pressure and flow data for monitored elements
                    # Get monitored nodes (default to first 5 if not set)
                    monitored_nodes = st.session_state.get('monitored_nodes', list(wn.node_name_list)[:5])
                    for node_name in monitored_nodes:
                        if node_name in wn.node_name_list:  # Ensure node exists
                            node = wn.get_node(node_name)
                            if node_name not in st.session_state.simulation_data['pressures']:
                                st.session_state.simulation_data['pressures'][node_name] = []
                            pressure = getattr(node, 'pressure', 0) if hasattr(node, 'pressure') else 0
                            st.session_state.simulation_data['pressures'][node_name].append(pressure)
                    
                    # Get monitored links (default to first 5 if not set)
                    monitored_links = st.session_state.get('monitored_links', list(wn.link_name_list)[:5])
                    for link_name in monitored_links:
                        if link_name in wn.link_name_list:  # Ensure link exists
                            link = wn.get_link(link_name)
                            if link_name not in st.session_state.simulation_data['flows']:
                                st.session_state.simulation_data['flows'][link_name] = []
                            flow = getattr(link, 'flow', 0) if hasattr(link, 'flow') else 0
                            st.session_state.simulation_data['flows'][link_name].append(flow)
                    
                    st.success("✅ Step completed!")
                else:
                    st.warning(f"⚠️ {message}")
            
            time.sleep(0.5)
            st.rerun()
    
    # Reset button
    if st.button("🔄 Reset", use_container_width=True):
        with st.spinner("Resetting..."):
            st.session_state.sim = MWNTRInteractiveSimulator(wn)
            st.session_state.current_sim_time = 0
            st.session_state.scheduled_events = []
            st.session_state.applied_events = []
            st.session_state.simulation_data = {'time': [], 'pressures': {}, 'flows': {}}
        st.success("🔄 Reset complete!")
        time.sleep(1)
        st.rerun()
    
    st.markdown("---")
    
    # Element configuration section
    st.markdown('<h4 class="section-header">🎯 Element Configuration</h4>', unsafe_allow_html=True)
    
    # Display selected element and event interface
    if st.session_state.current_element:
        element = st.session_state.current_element
        
        # Selected element display
        st.success(f"**Selected:** {element['name']}")
        st.info(f"**Type:** {element['type']} ({element['category']})")
        
        # Deselect button
        if st.button("❌ Deselect", use_container_width=True):
            st.session_state.selected_nodes = []
            st.session_state.selected_links = []
            st.session_state.current_element = None
            st.rerun()
        
        # Compact event interface
        st.markdown("**⚡ Quick Events:**")
        new_event = display_event_interface(element['name'], element['type'], element['category'])
        if new_event:
            st.session_state.scheduled_events.append(new_event)
            st.success(f"✅ Event scheduled!")
            time.sleep(1)
            st.rerun()
    
    else:
        st.info("👆 **Select an element** from the network map to configure events")
        
        # Alternative selection for backup
        with st.expander("🔍 Manual Selection", expanded=False):
            node_options = ["None"] + list(wn.node_name_list)
            selected_node = st.selectbox("Node:", node_options, key="node_selector")
            
            link_options = ["None"] + list(wn.link_name_list)
            selected_link = st.selectbox("Link:", link_options, key="link_selector")
            
            if selected_node != "None":
                st.session_state.current_element = {
                    'name': selected_node,
                    'category': 'node',
                    'type': wn.get_node(selected_node).node_type
                }
                st.session_state.selected_nodes = [selected_node]
                st.session_state.selected_links = []
                st.rerun()
                
            if selected_link != "None":
                st.session_state.current_element = {
                    'name': selected_link,
                    'category': 'link',
                    'type': wn.get_link(selected_link).link_type
                }
                st.session_state.selected_links = [selected_link]
                st.session_state.selected_nodes = []
                st.rerun()
    
    # Events summary
    st.markdown("---")
    st.markdown('<h4 class="section-header">📅 Events</h4>', unsafe_allow_html=True)
    
    event_col1, event_col2 = st.columns(2)
    with event_col1:
        st.metric("Scheduled", len(st.session_state.scheduled_events))
    with event_col2:
        st.metric("Applied", len(st.session_state.applied_events))
    
    # Quick event list
    if st.session_state.scheduled_events:
        with st.expander("⏰ Upcoming Events", expanded=False):
            for event in st.session_state.scheduled_events[-3:]:  # Show last 3
                st.write(f"- {event['event_type']} → {event['element_name']}")
    
    if st.session_state.applied_events:
        with st.expander("✅ Recent Events", expanded=False):
            for event in st.session_state.applied_events[-3:]:  # Show last 3
                st.write(f"- {event['event_type']} → {event['element_name']}")

# Footer
st.markdown("---")

# Results Visualization (moved to bottom for better workflow)
if len(st.session_state.simulation_data['time']) > 1:
    with st.container():
        st.markdown('<h3 class="section-header">📊 Simulation Results</h3>', unsafe_allow_html=True)
        
        # Selection controls for monitoring different elements
        monitoring_col1, monitoring_col2 = st.columns(2)
        
        with monitoring_col1:
            st.markdown("**📍 Select Nodes to Monitor:**")
            available_nodes = list(wn.node_name_list)
            if 'monitored_nodes' not in st.session_state:
                st.session_state.monitored_nodes = available_nodes[:5]  # Default first 5
            
            monitored_nodes = st.multiselect(
                "Choose nodes for pressure monitoring:",
                available_nodes,
                default=st.session_state.monitored_nodes,
                key="pressure_monitoring_nodes"
            )
            st.session_state.monitored_nodes = monitored_nodes
        
        with monitoring_col2:
            st.markdown("**🔗 Select Links to Monitor:**")
            available_links = list(wn.link_name_list)
            if 'monitored_links' not in st.session_state:
                st.session_state.monitored_links = available_links[:5]  # Default first 5
            
            monitored_links = st.multiselect(
                "Choose links for flow monitoring:",
                available_links,
                default=st.session_state.monitored_links,
                key="flow_monitoring_links"
            )
            st.session_state.monitored_links = monitored_links
        
        # Create separate pressure and flow charts side by side
        pressure_data = st.session_state.simulation_data['pressures']
        flow_data = st.session_state.simulation_data['flows']
        
        chart_col1, chart_col2 = st.columns(2)
        
        # Pressure chart
        with chart_col1:
            if pressure_data and monitored_nodes:
                filtered_pressure_data = {node: pressure_data[node] for node in monitored_nodes if node in pressure_data}
                if filtered_pressure_data:
                    fig_pressure = go.Figure()
                    for node_name, pressure_values in filtered_pressure_data.items():
                        fig_pressure.add_trace(go.Scatter(
                            x=st.session_state.simulation_data['time'],
                            y=pressure_values,
                            mode='lines+markers',
                            name=node_name,
                            line=dict(width=3),
                            marker=dict(size=6)
                        ))
                    
                    fig_pressure.update_layout(
                        title="🔵 Node Pressure Over Time",
                        xaxis_title="Time (s)",
                        yaxis_title="Pressure (m)",
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_pressure, use_container_width=True)
                else:
                    st.info("📍 No pressure data available for selected nodes")
            else:
                st.info("📍 Select nodes above to see pressure monitoring")
        
        # Flow chart
        with chart_col2:
            if flow_data and monitored_links:
                filtered_flow_data = {link: flow_data[link] for link in monitored_links if link in flow_data}
                if filtered_flow_data:
                    fig_flow = go.Figure()
                    for link_name, flow_values in filtered_flow_data.items():
                        fig_flow.add_trace(go.Scatter(
                            x=st.session_state.simulation_data['time'],
                            y=flow_values,
                            mode='lines+markers',
                            name=link_name,
                            line=dict(width=3),
                            marker=dict(size=6)
                        ))
                    
                    fig_flow.update_layout(
                        title="🔗 Link Flow Over Time",
                        xaxis_title="Time (s)",
                        yaxis_title="Flow Rate (m³/s)",
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_flow, use_container_width=True)
                else:
                    st.info("🔗 No flow data available for selected links")
            else:
                st.info("🔗 Select links above to see flow monitoring")
        
        # Current values display for monitored elements
        if st.session_state.current_sim_time > 0:
            st.markdown('<h4 class="section-header">📈 Current Values for Monitored Elements</h4>', unsafe_allow_html=True)
            
            current_col1, current_col2 = st.columns(2)
            
            with current_col1:
                if monitored_nodes:
                    st.write("**Current Node Pressures:**")
                    for node_name in monitored_nodes[:6]:  # Show max 6 for space
                        if node_name in wn.node_name_list:
                            node = wn.get_node(node_name)
                            pressure = getattr(node, 'pressure', 0) if hasattr(node, 'pressure') else 0
                            st.metric(f"{node_name}", f"{pressure:.2f} m", delta=None)
                else:
                    st.info("📍 No nodes selected for monitoring")
            
            with current_col2:
                if monitored_links:
                    st.write("**Current Link Flow Rates:**")
                    for link_name in monitored_links[:6]:  # Show max 6 for space
                        if link_name in wn.link_name_list:
                            link = wn.get_link(link_name)
                            flow = getattr(link, 'flow', 0) if hasattr(link, 'flow') else 0
                            st.metric(f"{link_name}", f"{flow:.4f} m³/s", delta=None)
                else:
                    st.info("🔗 No links selected for monitoring")

st.markdown("---")
st.caption(f"Interactive Network Simulator | Network: {INP_FILE} | Duration: {SIMULATION_DURATION_SECONDS//3600}h") 