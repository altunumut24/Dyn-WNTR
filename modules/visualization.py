"""
Visualization functions for Interactive Network Simulator.

This module handles all the visual aspects of the water network simulation:
- Interactive network maps with clickable elements
- Color-coded pressure and flow visualization
- Time-series monitoring charts
- Event timeline displays
- Color scale legends

Key responsibilities:
- Converting network data into visual representations
- Mapping pressure/flow values to colors
- Creating interactive Plotly charts
- Handling user clicks on network elements
- Displaying simulation results over time

The main visualization flow:
1. Extract network layout (node positions, connections)
2. Map simulation data to colors (pressure → node colors, flow → pipe colors)
3. Create interactive Plotly figures
4. Handle user clicks for element selection
5. Display time-series data in monitoring charts
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
import json
import datetime

# Import WNTR network components
from mwntr.network import WaterNetworkModel, Node, Link, LinkStatus
from mwntr.network.elements import Junction, Tank, Reservoir, Pipe, Pump, Valve
from mwntr.graphics.network import plot_interactive_network

# Import our configuration constants
from .config import (
    PRESSURE_COLOR_RANGES, FLOW_COLOR_RANGES, 
    CHART_HEIGHT, LEGEND_HEIGHT, TIMELINE_HEIGHT,
    MIN_FLOW_WIDTH, MAX_FLOW_WIDTH
)


def get_network_layout(wn: WaterNetworkModel):
    """
    Extract network layout information for plotting.
    
    This function processes the water network model to extract the spatial
    layout information needed for visualization. It organizes nodes and links
    with their positions and properties.
    
    Args:
        wn (WaterNetworkModel): The WNTR water network model
        
    Returns:
        Tuple containing:
        - node_positions: Dict mapping node names to (x, y) coordinates
        - edge_list: List of (start_node, end_node) tuples
        - node_info: Dict with node properties (type, position)
        - edge_info: Dict with link properties (start, end, type, midpoint)
        
    Technical details:
    - Converts WNTR network data into plotting-friendly format
    - Node coordinates determine where elements appear on the map
    - Edge info is used to draw pipes/links between nodes
    - Midpoints are used for placing link labels and click detection
    """
    # Extract node positions from the network model
    node_positions = {}
    for node_name, node in wn.nodes():
        if node.coordinates is not None:
            # Use actual coordinates from the INP file
            node_positions[node_name] = node.coordinates
        else:
            # Generate random positions if coordinates not available
            # This is a fallback for networks without spatial data
            node_positions[node_name] = (np.random.random(), np.random.random())
    
    # Build edge list and detailed edge information
    edge_list = []
    edge_info = {}
    for link_name, link in wn.links():
        # Get start and end positions for this link
        start_pos = node_positions[link.start_node_name]
        end_pos = node_positions[link.end_node_name]
        
        # Add to simple edge list (for basic connectivity)
        edge_list.append((link.start_node_name, link.end_node_name))
        
        # Store detailed information for visualization
        edge_info[link_name] = {
            'start': link.start_node_name,           # Starting node name
            'end': link.end_node_name,               # Ending node name
            'type': link.link_type,                  # Pipe, Pump, or Valve
            'mid_pos': ((start_pos[0] + end_pos[0])/2, (start_pos[1] + end_pos[1])/2)  # Midpoint for labels
        }
    
    # Organize node information for visualization
    node_info = {}
    for node_name, node in wn.nodes():
        node_info[node_name] = {
            'type': node.node_type,              # Junction, Tank, or Reservoir
            'pos': node_positions[node_name]     # (x, y) coordinates
        }
    
    return node_positions, edge_list, node_info, edge_info


def get_pressure_color(pressure, min_pressure, max_pressure):
    """
    Generate color based on pressure value using viridis-like color mapping.
    
    This function maps pressure values to colors for node visualization.
    Higher pressures get "warmer" colors (yellow), lower pressures get
    "cooler" colors (purple/blue).
    
    Args:
        pressure (float): The pressure value to map
        min_pressure (float): Minimum pressure in the dataset
        max_pressure (float): Maximum pressure in the dataset
        
    Returns:
        str: RGB color string (e.g., 'rgb(253, 231, 37)')
        
    Color mapping:
    - Yellow = high pressure (good), Purple = low pressure (concerning)
    - The color scale helps identify pressure problems visually
    - Uses viridis color scheme which is colorblind-friendly
    """
    # Handle edge case where all pressures are the same
    if max_pressure == min_pressure:
        return 'rgb(253, 231, 37)'  # Default yellow
    
    # Normalize pressure to 0-1 range
    normalized = (pressure - min_pressure) / (max_pressure - min_pressure)
    normalized = max(0, min(1, normalized))  # Clamp to valid range
    
    # Viridis-like color mapping: yellow → green → blue → purple
    # This creates a smooth color transition across pressure ranges
    
    if normalized < 0.25:
        # Yellow to green transition (high to medium-high pressure)
        t = normalized * 4  # Scale to 0-1 within this segment
        r = int(253 - (253 - 68) * t)   # Red: 253 → 68
        g = int(231 - (231 - 1) * t)    # Green: 231 → 1
        b = int(37 + (84 - 37) * t)     # Blue: 37 → 84
        
    elif normalized < 0.5:
        # Green to teal transition (medium-high to medium pressure)
        t = (normalized - 0.25) * 4
        r = int(68 - (68 - 33) * t)     # Red: 68 → 33
        g = int(1 + (144 - 1) * t)      # Green: 1 → 144
        b = int(84 + (140 - 84) * t)    # Blue: 84 → 140
        
    elif normalized < 0.75:
        # Teal to blue transition (medium to medium-low pressure)
        t = (normalized - 0.5) * 4
        r = int(33 - (33 - 59) * t)     # Red: 33 → 59
        g = int(144 - (144 - 82) * t)   # Green: 144 → 82
        b = int(140 + (139 - 140) * t)  # Blue: 140 → 139
        
    else:
        # Blue to purple transition (medium-low to low pressure)
        t = (normalized - 0.75) * 4
        r = int(59 + (68 - 59) * t)     # Red: 59 → 68
        g = int(82 - (82 - 1) * t)      # Green: 82 → 1
        b = int(139 + (84 - 139) * t)   # Blue: 139 → 84
    
    return f'rgb({r}, {g}, {b})'


def get_flow_color_and_width(flow, min_flow, max_flow):
    """Generate color and width based on flow value using viridis-like colors."""
    abs_flow = abs(flow)
    max_abs_flow = max(abs(min_flow), abs(max_flow))
    
    if max_abs_flow == 0:
        return 'rgb(253, 231, 37)', 2  # Default yellow
    
    # Width based on flow magnitude
    width = 2 + 8 * (abs_flow / max_abs_flow)
    
    # Use viridis-like color mapping for flow intensity
    intensity = abs_flow / max_abs_flow
    
    # Apply the same viridis color scheme as pressure
    if intensity < 0.25:
        # Yellow to green
        t = intensity * 4
        r = int(253 - (253 - 68) * t)
        g = int(231 - (231 - 1) * t)
        b = int(37 + (84 - 37) * t)
    elif intensity < 0.5:
        # Green to teal
        t = (intensity - 0.25) * 4
        r = int(68 - (68 - 33) * t)
        g = int(1 + (144 - 1) * t)
        b = int(84 + (140 - 84) * t)
    elif intensity < 0.75:
        # Teal to blue
        t = (intensity - 0.5) * 4
        r = int(33 - (33 - 59) * t)
        g = int(144 - (144 - 82) * t)
        b = int(140 + (139 - 140) * t)
    else:
        # Blue to purple
        t = (intensity - 0.75) * 4
        r = int(59 + (68 - 59) * t)
        g = int(82 - (82 - 1) * t)
        b = int(139 + (84 - 139) * t)
    
    # For negative flows, add a slight red tint to distinguish direction
    if flow < 0:
        r = min(255, r + 30)  # Add red component for reverse flow
    
    return f'rgb({r}, {g}, {b})', width


def create_network_plot(wn: WaterNetworkModel, selected_nodes: List[str] = None, selected_links: List[str] = None, 
                        show_simulation_data: bool = False, sim_initialized: bool = False, height: int = 900, node_size_scale: float = 1.0):
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
    link_hover_text = []
    
    for link_name, info in edge_info.items():
        link_x.append(info['mid_pos'][0])
        link_y.append(info['mid_pos'][1])
        link_names.append(link_name)
        link_types.append(info['type'])
        # Store name and type as customdata for click detection
        link_customdata.append([link_name, info['type'], 'link'])
        
        # Create hover text with flow information
        hover_text = f"<b>{link_name}</b><br>Type: {info['type']}"
        if link_name in link_flows:
            flow = link_flows[link_name]
            if flow is not None:
                hover_text += f"<br>Flow Rate: {flow:.4f} m³/s"
            else:
                hover_text += f"<br>Flow Rate: N/A"
        else:
            hover_text += f"<br>Flow Rate: No data"
        hover_text += "<br>Click to select"
        link_hover_text.append(hover_text)
        
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
    
    # Add white text shadow for link labels (larger for better visibility)
    fig.add_trace(go.Scatter(
        x=link_x,
        y=link_y,
        mode='text',
        text=link_names,
        textfont=dict(
            size=14,  # Increased from 10 to 14
            color='white',
            family="Arial Black"  # Bold font for shadow
        ),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add main link markers and text
    fig.add_trace(go.Scatter(
        x=link_x,
        y=link_y,
        mode='markers+text',
        marker=dict(
            size=8,   # Reduced marker size to 8 to make text more prominent
            color=link_colors,
            line=dict(width=1, color='#34495e'),  # Thinner border
            symbol=link_symbols,
            opacity=0.6  # More transparent markers
        ),
        text=link_names,
        textfont=dict(
            size=12,  # Increased from 9 to 12
            color='#1a252f',  # Darker color for better contrast
            family="Arial Black"  # Bold font for main text
        ),
        customdata=link_customdata,
        hovertemplate='%{hovertext}<extra></extra>',
        hovertext=link_hover_text,
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
        
        # Professional symbols and sizes for different node types (optimized for cloud deployments)
        if info['type'] == 'Junction':
            symbol = 'circle'
            size = 16  # Reduced from 25 for better cloud compatibility
            default_color = 'lightblue'
        elif info['type'] == 'Tank':
            symbol = 'square'
            size = 20  # Reduced from 30 for better cloud compatibility
            default_color = 'navy'
        elif info['type'] == 'Reservoir':
            symbol = 'hexagon'
            size = 18  # Reduced from 28 for better cloud compatibility
            default_color = 'darkgreen'
        else:
            symbol = 'circle'
            size = 14  # Reduced from 23 for better cloud compatibility
            default_color = 'gray'
        
        node_symbols.append(symbol)
        node_sizes.append(size * node_size_scale)
        
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
    
    # Add white text shadow for better readability
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='text',
        text=node_names,
        textposition='top center',
        textfont=dict(
            size=13,  # Increased from 12 to 13
            color='white',
            family="Arial Black"  # Bold font for shadow
        ),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add main node markers and text
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes, 
            color=node_colors, 
            line=dict(width=2, color='#34495e'),  # Professional dark gray
            symbol=node_symbols
        ),
        text=node_names,
        textposition='top center',
        textfont=dict(
            size=12,  # Increased from 11 to 12
            color='#1a252f',  # Darker color for better contrast
            family="Arial Black"  # Bold font for main text
        ),
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
        height=height,
        autosize=True  # Make it more responsive to container size
    )
    
    return fig


def create_pressure_colorbar(min_pressure, max_pressure):
    """Create a simple pressure colorbar."""
    fig = go.Figure()
    
    # Create a simple colorbar
    pressures = np.linspace(min_pressure, max_pressure, 100)
    colors = [get_pressure_color(p, min_pressure, max_pressure) for p in pressures]
    
    fig.add_trace(go.Scatter(
        x=pressures,
        y=[0] * len(pressures),
        mode='markers',
        marker=dict(
            size=10,
            color=colors,
            line=dict(width=0)
        ),
        showlegend=False,
        hovertemplate='Pressure: %{x:.2f} m<extra></extra>'
    ))
    
    fig.update_layout(
        title="🔵 Node Pressure",
        xaxis_title="Pressure (m)",
        yaxis=dict(showticklabels=False, showgrid=False),
        height=120,
        margin=dict(l=50, r=50, t=40, b=40)
    )
    
    return fig


def create_flow_colorbar(min_flow, max_flow):
    """Create a simple flow colorbar."""
    fig = go.Figure()
    
    # Create a simple colorbar
    flows = np.linspace(min_flow, max_flow, 100)
    colors = []
    widths = []
    
    for f in flows:
        color, width = get_flow_color_and_width(f, min_flow, max_flow)
        colors.append(color)
        widths.append(max(6, min(width, 15)))  # Scale for visibility
    
    fig.add_trace(go.Scatter(
        x=flows,
        y=[0] * len(flows),
        mode='markers',
        marker=dict(
            size=widths,
            color=colors,
            line=dict(width=0)
        ),
        showlegend=False,
        hovertemplate='Flow: %{x:.4f} m³/s<extra></extra>'
    ))
    
    fig.update_layout(
        title="🔗 Link Flow",
        xaxis_title="Flow (m³/s)",
        yaxis=dict(showticklabels=False, showgrid=False),
        height=120,
        margin=dict(l=50, r=50, t=40, b=40)
    )
    
    return fig


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


def create_monitoring_charts(simulation_data: Dict, monitored_nodes: List[str], monitored_links: List[str]) -> Tuple[go.Figure, go.Figure]:
    """Create professional monitoring charts for pressure and flow data with enhanced styling."""
    fig_pressure = None
    fig_flow = None
    
    # Professional color palette for traces
    professional_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf'   # Cyan
    ]
    
    # Pressure chart
    if monitored_nodes:
        fig_pressure = go.Figure()
        
        if 'pressures' in simulation_data and simulation_data['pressures']:
            pressure_data = simulation_data['pressures']
            time_data = simulation_data['time']
            
            # Add traces for each monitored node
            traces_added = 0
            for node_name in monitored_nodes:
                if node_name in pressure_data:
                    node_pressure_data = pressure_data[node_name]
                    
                    # Handle cases where data length doesn't match time length
                    if len(node_pressure_data) > 0:
                        # Ensure data lengths match
                        if len(node_pressure_data) == len(time_data):
                            # Perfect match - use all data
                            x_data = time_data
                            y_data = node_pressure_data
                        elif len(node_pressure_data) < len(time_data):
                            # Data is shorter - use matching portion
                            x_data = time_data[-len(node_pressure_data):]
                            y_data = node_pressure_data
                        else:
                            # Data is longer - truncate to match time
                            x_data = time_data
                            y_data = node_pressure_data[:len(time_data)]
                        
                        color = professional_colors[traces_added % len(professional_colors)]
                        fig_pressure.add_trace(go.Scatter(
                            x=x_data,
                            y=y_data,
                            mode='lines+markers',
                            name=f"🔵 {node_name}",
                            line=dict(width=3, color=color),
                            marker=dict(size=7, color=color, line=dict(width=2, color='white')),
                            hovertemplate='<b>%{fullData.name}</b><br>' +
                                        'Time: %{x:.0f}s<br>' +
                                        'Pressure: %{y:.2f}m<br>' +
                                        '<extra></extra>'
                        ))
                        traces_added += 1
                else:
                    # Element not in data yet - add a placeholder trace with single current point
                    if time_data:  # Only if we have time data
                        try:
                            # Add a single point at the current time with current value
                            current_time = time_data[-1]
                            # This will be populated by the immediate initialization above
                            fig_pressure.add_trace(go.Scatter(
                                x=[current_time],
                                y=[0],  # Will be updated in next simulation step
                                mode='markers',
                                name=f"{node_name} (new)",
                                line=dict(width=3, dash='dot'),
                                marker=dict(size=8, symbol='diamond')
                            ))
                            traces_added += 1
                        except:
                            pass
            
            # If no traces were added, create an empty chart with message
            if traces_added == 0:
                fig_pressure.add_trace(go.Scatter(
                    x=[0], y=[0],
                    mode='markers',
                    marker=dict(size=0, opacity=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                fig_pressure.add_annotation(
                    text="No data available for selected nodes<br>Start simulation to see pressure data",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=14, color="gray")
                )
        else:
            # No simulation data available yet
            fig_pressure.add_trace(go.Scatter(
                x=[0], y=[0],
                mode='markers',
                marker=dict(size=0, opacity=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig_pressure.add_annotation(
                text="No simulation data yet<br>Initialize and run simulation to see pressure data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray")
            )
        
        fig_pressure.update_layout(
            title=dict(
                text=f"🔵 Node Pressure Over Time ({len(monitored_nodes)} nodes selected)",
                font=dict(size=18, color='#2c3e50'),
                x=0.5
            ),
            xaxis=dict(
                title="Time (seconds)",
                titlefont=dict(size=14, color='#34495e'),
                gridcolor='#ecf0f1',
                gridwidth=1,
                showgrid=True,
                zeroline=True,
                zerolinecolor='#bdc3c7',
                zerolinewidth=2
            ),
            yaxis=dict(
                title="Pressure (meters)",
                titlefont=dict(size=14, color='#34495e'),
                gridcolor='#ecf0f1',
                gridwidth=1,
                showgrid=True,
                zeroline=True,
                zerolinecolor='#bdc3c7',
                zerolinewidth=2
            ),
            height=400,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='#bdc3c7',
                borderwidth=1,
                font=dict(size=12)
            ),
            margin=dict(l=50, r=50, t=60, b=50)
        )
    
    # Flow chart
    if monitored_links:
        fig_flow = go.Figure()
        
        if 'flows' in simulation_data and simulation_data['flows']:
            flow_data = simulation_data['flows']
            time_data = simulation_data['time']
            
            # Add traces for each monitored link
            traces_added = 0
            for link_name in monitored_links:
                if link_name in flow_data:
                    link_flow_data = flow_data[link_name]
                    
                    # Handle cases where data length doesn't match time length
                    if len(link_flow_data) > 0:
                        # Ensure data lengths match
                        if len(link_flow_data) == len(time_data):
                            # Perfect match - use all data
                            x_data = time_data
                            y_data = link_flow_data
                        elif len(link_flow_data) < len(time_data):
                            # Data is shorter - use matching portion
                            x_data = time_data[-len(link_flow_data):]
                            y_data = link_flow_data
                        else:
                            # Data is longer - truncate to match time
                            x_data = time_data
                            y_data = link_flow_data[:len(time_data)]
                        
                        color = professional_colors[traces_added % len(professional_colors)]
                        fig_flow.add_trace(go.Scatter(
                            x=x_data,
                            y=y_data,
                            mode='lines+markers',
                            name=f"🔗 {link_name}",
                            line=dict(width=3, color=color),
                            marker=dict(size=7, color=color, line=dict(width=2, color='white')),
                            hovertemplate='<b>%{fullData.name}</b><br>' +
                                        'Time: %{x:.0f}s<br>' +
                                        'Flow: %{y:.4f}m³/s<br>' +
                                        '<extra></extra>'
                        ))
                        traces_added += 1
                else:
                    # Element not in data yet - add a placeholder trace with single current point
                    if time_data:  # Only if we have time data
                        try:
                            # Add a single point at the current time with current value
                            current_time = time_data[-1]
                            # This will be populated by the immediate initialization above
                            fig_flow.add_trace(go.Scatter(
                                x=[current_time],
                                y=[0],  # Will be updated in next simulation step
                                mode='markers',
                                name=f"{link_name} (new)",
                                line=dict(width=3, dash='dot'),
                                marker=dict(size=8, symbol='diamond')
                            ))
                            traces_added += 1
                        except:
                            pass
            
            # If no traces were added, create an empty chart with message
            if traces_added == 0:
                fig_flow.add_trace(go.Scatter(
                    x=[0], y=[0],
                    mode='markers',
                    marker=dict(size=0, opacity=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                fig_flow.add_annotation(
                    text="No data available for selected links<br>Start simulation to see flow data",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=14, color="gray")
                )
        else:
            # No simulation data available yet
            fig_flow.add_trace(go.Scatter(
                x=[0], y=[0],
                mode='markers',
                marker=dict(size=0, opacity=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig_flow.add_annotation(
                text="No simulation data yet<br>Initialize and run simulation to see flow data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="gray")
            )
        
        fig_flow.update_layout(
            title=dict(
                text=f"🔗 Link Flow Over Time ({len(monitored_links)} links selected)",
                font=dict(size=18, color='#2c3e50'),
                x=0.5
            ),
            xaxis=dict(
                title="Time (seconds)",
                titlefont=dict(size=14, color='#34495e'),
                gridcolor='#ecf0f1',
                gridwidth=1,
                showgrid=True,
                zeroline=True,
                zerolinecolor='#bdc3c7',
                zerolinewidth=2
            ),
            yaxis=dict(
                title="Flow Rate (m³/s)",
                titlefont=dict(size=14, color='#34495e'),
                gridcolor='#ecf0f1',
                gridwidth=1,
                showgrid=True,
                zeroline=True,
                zerolinecolor='#bdc3c7',
                zerolinewidth=2
            ),
            height=400,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='#bdc3c7',
                borderwidth=1,
                font=dict(size=12)
            ),
            margin=dict(l=50, r=50, t=60, b=50)
        )
    
    return fig_pressure, fig_flow 