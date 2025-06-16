"""
Visualization functions for Interactive Network Simulator.
Contains all plotting, color mapping, and network visualization functions.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
import json
import datetime

from mwntr.network import WaterNetworkModel, Node, Link, LinkStatus
from mwntr.network.elements import Junction, Tank, Reservoir, Pipe, Pump, Valve
from mwntr.graphics.network import plot_interactive_network

from .config import (
    PRESSURE_COLOR_RANGES, FLOW_COLOR_RANGES, 
    CHART_HEIGHT, LEGEND_HEIGHT, TIMELINE_HEIGHT,
    MIN_FLOW_WIDTH, MAX_FLOW_WIDTH
)


def get_network_layout(wn: WaterNetworkModel):
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


def get_pressure_color(pressure, min_pressure, max_pressure):
    """Generate color based on pressure value using viridis-like yellow to purple mapping."""
    if max_pressure == min_pressure:
        return 'rgb(253, 231, 37)'  # Default yellow
    
    normalized = (pressure - min_pressure) / (max_pressure - min_pressure)
    normalized = max(0, min(1, normalized))
    
    # Viridis-like color mapping: yellow -> green -> blue -> purple
    if normalized < 0.25:
        # Yellow to green
        t = normalized * 4
        r = int(253 - (253 - 68) * t)
        g = int(231 - (231 - 1) * t)
        b = int(37 + (84 - 37) * t)
    elif normalized < 0.5:
        # Green to teal
        t = (normalized - 0.25) * 4
        r = int(68 - (68 - 33) * t)
        g = int(1 + (144 - 1) * t)
        b = int(84 + (140 - 84) * t)
    elif normalized < 0.75:
        # Teal to blue
        t = (normalized - 0.5) * 4
        r = int(33 - (33 - 59) * t)
        g = int(144 - (144 - 82) * t)
        b = int(140 + (139 - 140) * t)
    else:
        # Blue to purple
        t = (normalized - 0.75) * 4
        r = int(59 + (68 - 59) * t)
        g = int(82 - (82 - 1) * t)
        b = int(139 + (84 - 139) * t)
    
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
    """Create monitoring charts for pressure and flow data."""
    fig_pressure = None
    fig_flow = None
    
    # Pressure chart
    if monitored_nodes and 'pressures' in simulation_data:
        pressure_data = simulation_data['pressures']
        filtered_pressure_data = {node: pressure_data[node] for node in monitored_nodes if node in pressure_data}
        
        if filtered_pressure_data:
            fig_pressure = go.Figure()
            for node_name, pressure_values in filtered_pressure_data.items():
                fig_pressure.add_trace(go.Scatter(
                    x=simulation_data['time'],
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
    
    # Flow chart
    if monitored_links and 'flows' in simulation_data:
        flow_data = simulation_data['flows']
        filtered_flow_data = {link: flow_data[link] for link in monitored_links if link in flow_data}
        
        if filtered_flow_data:
            fig_flow = go.Figure()
            for link_name, flow_values in filtered_flow_data.items():
                fig_flow.add_trace(go.Scatter(
                    x=simulation_data['time'],
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
    
    return fig_pressure, fig_flow 