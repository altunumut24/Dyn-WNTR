"""
Interactive Water Network Event Simulator - Dash Application

This is the main Dash application file that orchestrates the entire
water network simulation interface. It brings together all the modules
to create a cohesive user experience.

Key responsibilities:
- Setting up the Dash app configuration and layout
- Managing app state using dcc.Store components
- Coordinating between different modules (simulation, visualization, UI)
- Handling user interactions through callbacks
- Managing both interactive and batch simulation modes

Architecture overview:
1. App setup and configuration
2. State initialization using dcc.Store
3. Network model loading
4. Tab-based navigation (Interactive vs Batch mode)
5. Event handling and simulation control through callbacks
6. Results visualization and analysis
"""

import dash
from dash import dcc, html, Input, Output, State, callback, ALL, MATCH, ctx, no_update, Patch
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import time
import datetime
import json
import uuid
import os
import tempfile
import base64
import copy
import numpy as np
from typing import Dict, Any, List, Optional
import pandas as pd

# Import our modularized components
from modules.config import INP_FILE, SIMULATION_DURATION_SECONDS, HYDRAULIC_TIMESTEP_SECONDS
from modules.simulation import (
    load_network_model, apply_event_to_simulator, run_simulation_step,
    load_events_from_json, collect_simulation_data, 
    initialize_simulation_data, reset_simulation_state
)
from modules.visualization import (
    create_network_plot, create_pressure_colorbar, create_flow_colorbar,
    display_event_timeline, create_monitoring_charts
)

def create_simple_monitoring_chart(elements, selected_elements, chart_type="pressure", sim_data=None):
    """Create monitoring chart with selected elements - from plotly_restyle_test.py"""
    import plotly.graph_objects as go
    import numpy as np
    
    fig = go.Figure()
    
    if chart_type == "pressure":
        chart_title = f"🔵 Pressure Monitoring ({len(selected_elements)} nodes)"
        y_label = "Pressure (m)"
        base_value = 30
        data_key = "pressures"
    else:
        chart_title = f"🔗 Flow Monitoring ({len(selected_elements)} links)"
        y_label = "Flow (m³/s)"
        base_value = 0.5
        data_key = "flows"
    
    # Colors for traces
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Add traces for selected elements
    for i, element in enumerate(selected_elements):
        if element in elements:
            element_index = elements.index(element)
            
            # Use real simulation data if available, otherwise generate sample data
            if sim_data and 'time' in sim_data and data_key in sim_data:
                time_data = list(sim_data['time'])
                element_series = sim_data[data_key].get(element, None)
                if element_series is not None and len(element_series) == len(time_data):
                    y_data = element_series
                else:
                    # fallback to base pattern if lengths mismatch
                    y_data = [base_value + element_index*2 + 0.1*t for t in time_data]
            else:
                # Generate sample data for demonstration
                time_data = list(range(24))
                y_data = [base_value + element_index*2 + 0.1*t + 0.5*np.sin(t*0.5) for t in time_data]
            
            fig.add_trace(go.Scatter(
                x=time_data,
                y=y_data,
                mode='lines+markers',
                name=f"{element}",
                line=dict(color=colors[element_index % len(colors)], width=2),
                marker=dict(size=4, color=colors[element_index % len(colors)], 
                          line=dict(width=1, color='white')),
                hovertemplate=f"<b>{element}</b><br>" +
                            f"Time: %{{x}}<br>" +
                            f"{y_label}: %{{y:.2f}}<br>" +
                            "<extra></extra>"
            ))
    
    # Update layout with professional styling
    fig.update_layout(
        title=dict(
            text=chart_title,
            x=0.5,
            font=dict(size=16, family="Arial")
        ),
        xaxis_title="Time (hours)",
        yaxis_title=y_label,
        template="plotly_white",
        height=400,
        showlegend=True,
        legend=dict(
            x=1.01,
            y=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        margin=dict(t=60, l=50, r=100, b=50)
    )
    
    return fig
from modules.dash_ui_components import create_event_configuration_modal

# Import the simple monitoring system

# Global variables to store actual WNTR objects (can't serialize these)
global_state = {
    'wn': None,
    'sim': None
}

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Interactive Water Network Event Simulator"

# Store components for session state management
def create_stores():
    """Create all dcc.Store components for state management."""
    return [
        # Core network and simulation state
        dcc.Store(id='network-loaded', data=False),
        dcc.Store(id='current-inp-file', data=INP_FILE),
        dcc.Store(id='sim-initialized', data=False),
        
        # User interface selections
        dcc.Store(id='selected-nodes', data=[]),
        dcc.Store(id='selected-links', data=[]),
        dcc.Store(id='current-element', data=None),
        
        # Event management
        dcc.Store(id='scheduled-events', data=[]),
        dcc.Store(id='applied-events', data=[]),
        dcc.Store(id='current-sim-time', data=0),
        dcc.Store(id='simulation-data', data=initialize_simulation_data()),
        dcc.Store(id='event-notification-store', data=None),
        
        # Monitoring selections
        dcc.Store(id='pressure-monitoring-nodes', data=[]),
        dcc.Store(id='flow-monitoring-links', data=[]),
        
        # Simulation planning
        dcc.Store(id='simulation-duration-hours', data=24),
        dcc.Store(id='simulation-timestep-minutes', data=60),
        
        # Network metadata
        dcc.Store(id='network-metadata', data={}),
        
        # Batch simulation stores
        dcc.Store(id='batch-events', data=[]),
        dcc.Store(id='batch-metadata', data={}),
        dcc.Store(id='batch-animation-running', data=False),
        dcc.Store(id='batch-current-time', data=0),
        dcc.Store(id='batch-simulation-data', data=initialize_simulation_data()),
        dcc.Store(id='batch-applied-events', data=[]),
        dcc.Store(id='batch-animation-speed', data=1.0),
        
        # Real-time monitoring refresh interval
        dcc.Interval(
            id='monitoring-refresh-interval',
            interval=30*1000,  # 30 seconds in milliseconds
            n_intervals=0,
            disabled=False
        ),
        
        # Label visibility control
        dcc.Store(id='show-labels-always', data=True),
        
        # View mode control (dual vs single)
        dcc.Store(id='view-mode', data='dual'),
        dcc.Store(id='batch-view-mode', data='dual'),
    ]

# Batch layout removed - to be implemented from scratch

def create_layout():
    """Create the main application layout."""
    return dbc.Container([
        # Store components
        html.Div(create_stores()),
        
        # Toast container for notifications
        html.Div(id="toast-container", style={
            "position": "fixed",
            "top": "2rem",
            "right": "2rem",
            "zIndex": 1050
        }),
        
        # Hidden monitoring dropdowns for automatic selection
        html.Div([
            dcc.Dropdown(id="pressure-nodes-dropdown", multi=True, style={"display": "none"}),
            dcc.Dropdown(id="flow-links-dropdown", multi=True, style={"display": "none"})
        ], style={"display": "none"}),
        
        # Main title
        dbc.Row([
            dbc.Col([
                html.H1("🌊 Interactive Water Network Event Simulator", 
                       className="text-center mb-4 text-primary")
            ])
        ]),
        
        # Status messages area
        html.Div(id="status-messages-area"),
        
        # Tab navigation
        dbc.Row([
            dbc.Col([
                dbc.Tabs([
                    dbc.Tab(label="🎮 Interactive Mode", tab_id="interactive-tab"),
                    dbc.Tab(label="🎬 Batch Simulator", tab_id="batch-tab")
                ], id="main-tabs", active_tab="interactive-tab", className="mb-4")
            ])
        ]),
        
        # Tab content
        html.Div(id="tab-content"),
        
        # Footer
        html.Hr(),
        html.Footer([
            html.P("Interactive Network Simulator | Built with Dash & Plotly", 
                  className="text-center text-muted small")
        ]),
    ], fluid=True, className="px-4 py-3")

def create_network_file_selector():
    """Create network file selection interface."""
    return dbc.Card([
        dbc.CardHeader(html.H4("📁 Network Configuration")),
        dbc.CardBody([
            dbc.Label("Network Source:", className="fw-bold"),
            dbc.RadioItems([
                {"label": "📊 Use Example Networks", "value": "example"},
                {"label": "📤 Upload Custom Network", "value": "upload"}
            ], value="example", id="file-source-radio", className="mb-3"),
            
            # Example networks dropdown
            html.Div([
                dbc.Label("Choose an example network:", className="fw-bold"),
                dbc.Select([
                    {"label": "NET_2.inp - Large Distribution Network", "value": "NET_2.inp"},
                    {"label": "NET_3.inp - Medium Distribution Network", "value": "NET_3.inp"},
                    {"label": "NET_4.inp - Sample Distribution Network", "value": "NET_4.inp"},
                    {"label": "GRID.inp - Custom Grid Network", "value": "custom_wdn_pump.inp"}
                ], value="NET_4.inp", id="example-network-select")
            ], id="example-networks-div"),
            
            # File upload
            html.Div([
                dbc.Label("Upload Network File:", className="fw-bold"),
                dcc.Upload([
                    html.Div([
                        "Drag and Drop or ",
                        html.A("Select Files", className="text-primary")
                    ], className="text-center p-4 border border-dashed rounded")
                ], id="upload-network-file", multiple=False),
                html.Div(id="upload-status")
            ], id="upload-div", style={"display": "none"}),
            
            # Load button
            dbc.Button("🔄 Load Network", id="load-network-btn", 
                      color="primary", className="mt-3")
        ])
    ])

# Add modals for selecting nodes and links

# Helper function for creating interactive selection plots
def create_interactive_selection_plots(available_nodes, available_links, monitored_nodes, monitored_links):
    """Create interactive plots for selecting nodes and links to monitor."""
    import plotly.graph_objects as go
    
    # Node selection plot
    node_selection_fig = go.Figure()
    for i, node in enumerate(available_nodes):
        color = 'darkgreen' if node in (monitored_nodes or []) else 'lightblue'
        marker_size = 15 if node in (monitored_nodes or []) else 10
        node_selection_fig.add_trace(go.Scatter(
            x=[i % 8],  # 8 columns
            y=[i // 8],
            mode='markers+text',
            marker=dict(size=marker_size, color=color, line=dict(width=2, color='white')),
            text=node,
            textposition='middle center',
            textfont=dict(size=8, color='white' if node in (monitored_nodes or []) else 'black'),
            name=node,
            customdata=[node],
            hovertemplate=f"<b>Node: {node}</b><br>📊 Click to toggle monitoring<br>Status: {'🟢 Monitored' if node in (monitored_nodes or []) else '⚪ Available'}<extra></extra>",
            showlegend=False
        ))
    
    node_selection_fig.update_layout(
        title="🔵 Click Nodes to Add/Remove from Pressure Monitoring",
        title_font_size=14,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=250,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor='#f8f9fa'
    )
    
    # Link selection plot
    link_selection_fig = go.Figure()
    for i, link in enumerate(available_links):
        color = 'darkred' if link in (monitored_links or []) else 'lightcoral'
        marker_size = 15 if link in (monitored_links or []) else 10
        link_selection_fig.add_trace(go.Scatter(
            x=[i % 8],  # 8 columns
            y=[i // 8],
            mode='markers+text',
            marker=dict(size=marker_size, color=color, symbol='square', line=dict(width=2, color='white')),
            text=link,
            textposition='middle center',
            textfont=dict(size=8, color='white' if link in (monitored_links or []) else 'black'),
            name=link,
            customdata=[link],
            hovertemplate=f"<b>Link: {link}</b><br>📊 Click to toggle monitoring<br>Status: {'🟢 Monitored' if link in (monitored_links or []) else '⚪ Available'}<extra></extra>",
            showlegend=False
        ))
    
    link_selection_fig.update_layout(
        title="🔗 Click Links to Add/Remove from Flow Monitoring",
        title_font_size=14,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=250,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor='#f8f9fa'
    )
    
    return node_selection_fig, link_selection_fig

def create_node_selection_modal():
    """Create modal for selecting nodes to add to monitoring."""
    return dbc.Modal([
        dbc.ModalHeader("Select Nodes to Monitor"),
        dbc.ModalBody([
            html.P("Select nodes to add to pressure monitoring:"),
            dcc.Dropdown(
                id="node-selection-dropdown",
                multi=True,
                placeholder="Select nodes...",
                style={"marginBottom": "10px"}
            ),
            dbc.Alert("Selected nodes will be added to the pressure monitoring chart.", 
                     color="info", className="mt-2")
        ]),
        dbc.ModalFooter([
            dbc.Button("Add Selected", id="confirm-add-nodes", color="success", className="me-2"),
            dbc.Button("Cancel", id="cancel-add-nodes", color="secondary")
        ])
    ], id="node-selection-modal", size="lg")

def create_link_selection_modal():
    """Create modal for selecting links to add to monitoring."""
    return dbc.Modal([
        dbc.ModalHeader("Select Links to Monitor"),
        dbc.ModalBody([
            html.P("Select links to add to flow monitoring:"),
            dcc.Dropdown(
                id="link-selection-dropdown",
                multi=True,
                placeholder="Select links...",
                style={"marginBottom": "10px"}
            ),
            dbc.Alert("Selected links will be added to the flow monitoring chart.", 
                     color="info", className="mt-2")
        ]),
        dbc.ModalFooter([
            dbc.Button("Add Selected", id="confirm-add-links", color="success", className="me-2"),
            dbc.Button("Cancel", id="cancel-add-links", color="secondary")
        ])
    ], id="link-selection-modal", size="lg")

# Set the app layout
app.layout = create_layout()

# Tab switching callback
@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'active_tab')
)
def render_tab_content(active_tab):
    """Render content based on selected tab."""
    if active_tab == "interactive-tab":
        return create_interactive_content()
    elif active_tab == "batch-tab":
        return create_batch_content()
    return "Select a tab"

def create_interactive_content():
    """Create the interactive mode content."""
    interactive_main_content = [
        # Map controls row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Alert("🖱️ Click on nodes (circles) or links (squares) to select and configure events", 
                                color="info", className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Map Height"),
                                dbc.Select([
                                    {"label": "600px", "value": 600},
                                    {"label": "700px", "value": 700},
                                    {"label": "800px", "value": 800},
                                    {"label": "900px", "value": 900}
                                ], value=700, id="map-height-select")
                            ], width=2),
                            dbc.Col([
                                dbc.Label("Node Size"),
                                dbc.Select([
                                    {"label": "60%", "value": 0.6},
                                    {"label": "80%", "value": 0.8},
                                    {"label": "100%", "value": 1.0},
                                    {"label": "120%", "value": 1.2}
                                ], value=1.0, id="node-size-select")
                            ], width=2),
                            dbc.Col([
                                dbc.Label("Labels"),
                                dbc.ButtonGroup([
                                    dbc.Button("Always", id="labels-always-btn", color="primary", size="sm"),
                                    dbc.Button("Hover", id="labels-hover-btn", color="outline-primary", size="sm")
                                ], className="w-100")
                            ], width=2),
                            dbc.Col([
                                dbc.Label("View"),
                                dbc.ButtonGroup([
                                    dbc.Button("Dual View", id="dual-view-btn", color="success", size="sm"),
                                    dbc.Button("Single View", id="single-view-btn", color="outline-success", size="sm")
                                ], className="w-100")
                            ], width=3)
                        ])
                    ])
                ], className="mb-3")
            ], width=12)
        ]),
        
        # Primary Network Map (Pressure/Flow) - Always shown
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("🌊 Pressure & Flow Visualization", className="mb-0")),
                    dbc.CardBody([
                        html.Div(id="network-map-container"),
                        html.Div(id="color-legends-container", className="mt-2")
                    ])
                ])
            ], width=12, id="primary-map-col")
        ], className="mb-3"),
        
        # Secondary State Map (Demands/Leaks/Closures) - Shown below in dual view
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("🎯 Network Operational States", className="mb-0")),
                    dbc.CardBody([
                        html.Div(id="state-map-container"),
                        dbc.Alert("Shows: 🔵 Active Demands | 🔴 Leaks | ❌ Closed Links", 
                                color="light", className="mt-2 small")
                    ])
                ])
            ], width=12, id="state-map-col")
        ], id="state-map-row"),
        
        # Control panel row
        dbc.Row([
            dbc.Col([
                # Simulation controls
                dbc.Card([
                    dbc.CardHeader(html.H5("⚡ Simulation Controls")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div(id="sim-status"),
                                # Simulation progress
                                html.Div(id="simulation-progress-display", className="mb-3")
                            ], width=8),
                            dbc.Col([
                                # Planning controls
                                dbc.Row([
                                    dbc.Col([
                                        dbc.Label("Duration (hours)"),
                                        dbc.Input(id="duration-input", type="number", 
                                                value=24, min=1, max=87600, step=1)
                                    ], width=6),
                                    dbc.Col([
                                        dbc.Label("Timestep (minutes)"),
                                        dbc.Select([
                                            {"label": "15 min", "value": 15},
                                            {"label": "30 min", "value": 30},
                                            {"label": "60 min", "value": 60},
                                            {"label": "120 min", "value": 120},
                                        ], value=60, id="timestep-select")
                                    ], width=6)
                                ], className="mb-3"),
                                
                                # Control buttons
                                dbc.ButtonGroup([
                                    dbc.Button("🚀 Initialize", id="init-btn", color="success"),
                                    dbc.Button("⏭️ Step", id="step-btn", color="primary", disabled=True),
                                    dbc.Button("▶️ Play", id="play-step-btn", color="success", disabled=True),
                                    dbc.Button("⏸️ Pause", id="pause-step-btn", color="warning", disabled=True),
                                    dbc.Button("🔄 Reset", id="reset-btn", color="secondary")
                                ], className="w-100")
                            ], width=4)
                        ])
                    ])
                ])
            ], width=12)
        ], className="mt-3"),
        
        # Results section
        html.Hr(),
        html.Div(id="simulation-results-container"),
        
        # Event configuration modal
        create_event_configuration_modal()
    ]
    
    return [
        dbc.Row([dbc.Col([create_network_file_selector()])], className="mb-4"),
        html.Div(id="network-status-display"),
        html.Div(id="interactive-main-area-info", children=[
             dbc.Alert("👆 Please load a network file to begin simulation", color="info")
        ]),
        html.Div(id="interactive-main-area", style={'display': 'none'}, children=interactive_main_content),
        # Interval for auto-stepping when Play is active
        dcc.Interval(id="interactive-play-interval", interval=1000, n_intervals=0, disabled=True)
    ]

def create_batch_content():
    """Create the simplified batch simulator content."""
    return [
        # Network file selector (reused)
        dbc.Row([dbc.Col([create_network_file_selector()])], className="mb-4"),
        
        # Simple batch controls
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("🎬 Batch Event Simulation")),
                    dbc.CardBody([
                        dbc.Alert("Load event schedule for batch simulation.", color="info", className="mb-3"),
                        
                        # Event file source selection
                        dbc.Label("Event Source:", className="fw-bold"),
                        dbc.RadioItems([
                            {"label": "📋 Use Example Events", "value": "example"},
                            {"label": "📤 Upload Custom Events", "value": "upload"}
                        ], value="example", id="event-file-source-radio", className="mb-3"),

                        # Example events dropdown
                        html.Div([
                            dbc.Label("Choose example events:", className="fw-bold"),
                            dbc.Select([
                                {"label": "Generated Events 1", "value": "generated_events1.json"},
                                {"label": "Generated Events 2", "value": "generated_events2.json"}
                            ], value="generated_events1.json", id="example-events-select")
                        ], id="example-events-div"),

                        # Upload JSON
                        html.Div([
                            dbc.Label("Upload Events JSON:", className="fw-bold"),
                            dcc.Upload([
                                html.Div(["Drag and Drop or ", html.A("Select File", className="text-primary")], className="text-center p-4 border border-dashed rounded")
                            ], id="upload-events-file", multiple=False),
                            html.Div(id="upload-events-status"),
                            dbc.Button("📋 Load Uploaded Events", id="load-uploaded-events-btn", color="success", className="mt-2", size="lg")
                        ], id="upload-events-div", style={"display": "none"}),
                        
                        # Buttons shown only for example source
                        html.Div([
                            dbc.Button("📋 Load Events", id="load-events-btn", color="success", size="lg"),
                            dbc.Button("⬇️ Download Example Events", id="download-events-btn", color="info", className="ms-2", size="lg"),
                            dcc.Download(id="download-example-events")
                        ], id="example-events-buttons-div")
                    ])
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Network status display (shared)
        html.Div(id="batch-network-status-display"),
        
        # Batch simulation area
        html.Div(id="batch-main-area")
    ]

# Callback for file source radio
@app.callback(
    [Output('example-networks-div', 'style'),
     Output('upload-div', 'style')],
    Input('file-source-radio', 'value'),
    prevent_initial_call=True
)
def toggle_file_source(source):
    """Toggle between example networks and file upload."""
    if source == "example":
        return {"display": "block"}, {"display": "none"}
    else:
        return {"display": "none"}, {"display": "block"}

# Label visibility toggle callback (Interactive mode)
@app.callback(
    [Output('show-labels-always', 'data'),
     Output('labels-always-btn', 'color'),
     Output('labels-hover-btn', 'color')],
    [Input('labels-always-btn', 'n_clicks'),
     Input('labels-hover-btn', 'n_clicks')],
    State('show-labels-always', 'data'),
    prevent_initial_call=True
)
def toggle_label_visibility(always_clicks, hover_clicks, current_state):
    """Toggle between always showing labels and showing on hover only."""
    ctx_triggered = ctx.triggered_id if hasattr(ctx, "triggered_id") else None
    
    if ctx_triggered == 'labels-always-btn':
        # Always on mode
        return True, "primary", "outline-primary"
    elif ctx_triggered == 'labels-hover-btn':
        # Hover only mode
        return False, "outline-primary", "primary"
    
    # Default state - return current with appropriate button colors
    if current_state:
        return True, "primary", "outline-primary"
    else:
        return False, "outline-primary", "primary"

# Label visibility toggle callback (Batch mode)
@app.callback(
    [Output('show-labels-always', 'data', allow_duplicate=True),
     Output('batch-labels-always-btn', 'color'),
     Output('batch-labels-hover-btn', 'color')],
    [Input('batch-labels-always-btn', 'n_clicks'),
     Input('batch-labels-hover-btn', 'n_clicks')],
    State('show-labels-always', 'data'),
    prevent_initial_call=True
)
def toggle_batch_label_visibility(always_clicks, hover_clicks, current_state):
    """Toggle between always showing labels and showing on hover only in batch mode."""
    ctx_triggered = ctx.triggered_id if hasattr(ctx, "triggered_id") else None
    
    if ctx_triggered == 'batch-labels-always-btn':
        # Always on mode
        return True, "primary", "outline-primary"
    elif ctx_triggered == 'batch-labels-hover-btn':
        # Hover only mode
        return False, "outline-primary", "primary"
    
    # Default state - return current with appropriate button colors
    if current_state:
        return True, "primary", "outline-primary"
    else:
        return False, "outline-primary", "primary"

# View mode toggle callback (vertical stacking)
@app.callback(
    [Output('view-mode', 'data'),
     Output('dual-view-btn', 'color'),
     Output('single-view-btn', 'color'),
     Output('state-map-row', 'style')],
    [Input('dual-view-btn', 'n_clicks'),
     Input('single-view-btn', 'n_clicks')],
    State('view-mode', 'data'),
    prevent_initial_call=True
)
def toggle_view_mode(dual_clicks, single_clicks, current_mode):
    """Toggle between dual and single view modes - vertical stacking."""
    ctx_triggered = ctx.triggered_id if hasattr(ctx, "triggered_id") else None
    
    if ctx_triggered == 'dual-view-btn':
        # Dual view mode - show state plot below pressure/flow plot
        return 'dual', "success", "outline-success", {}
    elif ctx_triggered == 'single-view-btn':
        # Single view mode - hide state plot, show only pressure/flow plot
        return 'single', "outline-success", "success", {"display": "none"}
    
    # Default state based on current mode
    if current_mode == 'dual':
        return 'dual', "success", "outline-success", {}
    else:
        return 'single', "outline-success", "success", {"display": "none"}

# Batch view mode toggle callback
@app.callback(
    [Output('batch-view-mode', 'data'),
     Output('batch-dual-view-btn', 'color'),
     Output('batch-single-view-btn', 'color'),
     Output('batch-state-map-row', 'style')],
    [Input('batch-dual-view-btn', 'n_clicks'),
     Input('batch-single-view-btn', 'n_clicks')],
    State('batch-view-mode', 'data'),
    prevent_initial_call=True
)
def toggle_batch_view_mode(dual_clicks, single_clicks, current_mode):
    """Toggle between dual and single view modes in batch mode - vertical stacking."""
    ctx_triggered = ctx.triggered_id if hasattr(ctx, "triggered_id") else None
    
    if ctx_triggered == 'batch-dual-view-btn':
        # Dual view mode - show state plot below pressure/flow plot
        return 'dual', "success", "outline-success", {}
    elif ctx_triggered == 'batch-single-view-btn':
        # Single view mode - hide state plot, show only pressure/flow plot
        return 'single', "outline-success", "success", {"display": "none"}
    
    # Default state based on current mode
    if current_mode == 'dual':
        return 'dual', "success", "outline-success", {}
    else:
        return 'single', "outline-success", "success", {"display": "none"}

# Upload status callback - provides immediate feedback when file is selected
@app.callback(
    Output('upload-status', 'children'),
    Input('upload-network-file', 'filename'),
    prevent_initial_call=True
)
def show_upload_status(filename):
    """Show upload status when file is selected."""
    if filename:
        return dbc.Alert([
            html.I(className="fas fa-file-check me-2"),
            f"📁 Selected: {filename}"
        ], color="info", className="mt-2")
    return ""

# Removed: Batch file source radio callback (no longer needed for simplified batch mode)

# Callback for loading network
@app.callback(
    [Output('network-loaded', 'data'),
     Output('status-messages-area', 'children'),
     Output('network-metadata', 'data'),
     Output('pressure-monitoring-nodes', 'data'),
     Output('flow-monitoring-links', 'data')],
    Input('load-network-btn', 'n_clicks'),
    [State('file-source-radio', 'value'),
     State('example-network-select', 'value'),
     State('upload-network-file', 'contents'),
     State('upload-network-file', 'filename')],
    prevent_initial_call=True
)
def load_network_callback(n_clicks, source, example_file, upload_contents, upload_filename):
    """Load network model based on user selection."""
    if n_clicks is None or n_clicks == 0:
        return False, "", {}, [], []
    
    try:
        # Handle source parameter safely
        source = source or "example"
        wn = None
        file_display_name = ""
        
        if source == "example" and example_file:
            # Load from example file
            file_path = example_file
            wn = load_network_model(file_path)
            file_display_name = os.path.basename(file_path)
            
        elif source == "upload" and upload_contents and upload_filename:
            # Load from uploaded file
            try:
                # Decode the uploaded file
                content_type, content_string = upload_contents.split(',')
                decoded = base64.b64decode(content_string)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.inp', delete=False) as tmp_file:
                    tmp_file.write(decoded.decode('utf-8'))
                    temp_path = tmp_file.name
                
                # Load the network model from temporary file
                wn = load_network_model(temp_path)
                file_display_name = upload_filename
                file_path = upload_filename  # Use uploaded filename, not temp path
                
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as upload_error:
                error_msg = dbc.Alert(f"❌ Error processing uploaded file: {str(upload_error)}", color="danger")
                return False, error_msg, {}, [], []
        else:
            return False, dbc.Alert("❌ Please select a file", color="danger"), {}, [], []
        
        # Load the network model
        if not wn:
            return False, dbc.Alert("❌ Failed to load network file", color="danger"), {}, [], []
        
        if wn:
            # Store network in global state

            wn.add_pattern('gauss_pattern', [1.0, 0.5, 2.0, 0.0, 0.0, 0.0, 0.0, 1.5, 1.0, 0.5, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0])

            global_state['wn'] = wn
            
            # Keep an untouched copy of the original network for reset purposes
            global_state['original_wn'] = copy.deepcopy(wn)
            global_state['network_file_path'] = file_path
            
            # Create simulator instance
            from mwntr.sim.interactive_network_simulator import MWNTRInteractiveSimulator
            global_state['sim'] = MWNTRInteractiveSimulator(wn)
            
            # Create metadata for storage
            metadata = {
                'file_path': file_display_name,
                'node_count': len(wn.node_name_list),
                'link_count': len(wn.link_name_list),
                'node_names': list(wn.node_name_list),
                'link_names': list(wn.link_name_list)
            }
            
            # Initialize monitoring with first few elements
            initial_nodes = list(wn.node_name_list)[:5]
            initial_links = list(wn.link_name_list)[:5]
            
            success_msg = dbc.Alert(f"✅ Network '{file_display_name}' loaded successfully!", 
                                  color="success", dismissable=True)
            
            return True, success_msg, metadata, initial_nodes, initial_links
        else:
            error_msg = dbc.Alert("❌ Failed to load network file", color="danger")
            return False, error_msg, {}, [], []
            
    except Exception as e:
        error_msg = dbc.Alert(f"❌ Error loading network: {str(e)}", color="danger")
        return False, error_msg, {}, [], []

# Network status display callback
@app.callback(
    Output('network-status-display', 'children'),
    [Input('network-loaded', 'data'),
     Input('network-metadata', 'data'),
     Input('current-element', 'data'),
     Input('scheduled-events', 'data'),
     Input('applied-events', 'data')],
    prevent_initial_call=True
)
def display_network_status(network_loaded, metadata, current_element, scheduled_events, applied_events):
    """Display network status dashboard."""
    if not network_loaded or not metadata:
        return ""
    
    return dbc.Card([
        dbc.CardHeader(html.H4("📊 Network Status")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5(str(metadata['node_count']), className="text-primary"),
                    html.P("Total Nodes", className="mb-0")
                ], width=3),
                dbc.Col([
                    html.H5(str(metadata['link_count']), className="text-primary"),
                    html.P("Total Links", className="mb-0")
                ], width=3),
                dbc.Col([
                    html.H5(current_element['name'] if current_element else "None", 
                           className="text-primary"),
                    html.P("Selected Element", className="mb-0"),
                    html.Small(current_element['type'] if current_element else "", 
                             className="text-muted")
                ], width=3),
                dbc.Col([
                    html.H5(str(len(scheduled_events)), className="text-primary"),
                    html.P("Scheduled Events", className="mb-0"),
                    html.Small(f"{len(applied_events)} applied", className="text-muted")
                ], width=3)
            ])
        ])
    ], className="mb-4")

# Interactive main area callback
@app.callback(
    [Output('interactive-main-area', 'style'),
     Output('interactive-main-area-info', 'style')],
    Input('network-loaded', 'data'),
    prevent_initial_call=True
)
def toggle_interactive_area(network_loaded):
    """Toggle visibility of the main interactive area."""
    if network_loaded:
        return {'display': 'block'}, {'display': 'none'}
    return {'display': 'none'}, {'display': 'block'}

# Update monitoring dropdowns when network loads
@app.callback(
    [Output('pressure-nodes-dropdown', 'options'),
     Output('pressure-nodes-dropdown', 'value'),
     Output('flow-links-dropdown', 'options'),
     Output('flow-links-dropdown', 'value')],
    [Input('network-loaded', 'data'),
     Input('network-metadata', 'data')],
    prevent_initial_call=True
)
def update_monitoring_options(network_loaded, metadata):
    """Update monitoring dropdown options when network is loaded."""
    if not network_loaded or not metadata:
        return [], [], [], []
    
    # Create options for dropdowns
    node_options = [{"label": f"{name} (Node)", "value": name} for name in metadata['node_names']]
    link_options = [{"label": f"{name} (Link)", "value": name} for name in metadata['link_names']]
    
    # Set initial values (first 5 elements) 
    initial_nodes = metadata['node_names'][:5]
    initial_links = metadata['link_names'][:5]
    
    return node_options, initial_nodes, link_options, initial_links

# Additional callback to ensure monitoring is set up when simulation starts
@app.callback(
    [Output('pressure-nodes-dropdown', 'value', allow_duplicate=True),
     Output('flow-links-dropdown', 'value', allow_duplicate=True)],
    Input('sim-initialized', 'data'),
    [State('pressure-nodes-dropdown', 'value'),
     State('flow-links-dropdown', 'value'),
     State('network-metadata', 'data')],
    prevent_initial_call=True
)
def ensure_monitoring_on_init(sim_initialized, current_nodes, current_links, metadata):
    """Ensure monitoring elements are selected when simulation is initialized."""
    if sim_initialized and metadata:
        # If no elements are currently selected, select defaults
        if not current_nodes:
            current_nodes = metadata['node_names'][:5]
        if not current_links:
            current_links = metadata['link_names'][:5]
    
    return current_nodes or [], current_links or []

# Network map callback (dual view)
@app.callback(
    [Output('network-map-container', 'children'),
     Output('color-legends-container', 'children'),
     Output('state-map-container', 'children')],
    [Input('network-loaded', 'data'),
     Input('map-height-select', 'value'),
     Input('node-size-select', 'value'),
     Input('selected-nodes', 'data'),
     Input('selected-links', 'data'),
     Input('sim-initialized', 'data'),
     Input('simulation-data', 'data'),
     Input('current-sim-time', 'data'),
     Input('show-labels-always', 'data')],
    prevent_initial_call=True
)
def update_network_maps(network_loaded, map_height, node_size, 
                      selected_nodes, selected_links, sim_initialized, simulation_data, current_time, show_labels_always):
    """Update both network map visualizations."""
    try:
        if not network_loaded or global_state['wn'] is None:
            return "", "", ""
        
        wn = global_state['wn']
        show_sim_data = sim_initialized or False
        
        # Create network plot using existing visualization module
        map_height_int = int(map_height) if map_height else 700
        node_size_float = float(node_size) if node_size else 1.0
        
        # Primary network plot (pressure/flow)
        fig = create_network_plot(
            wn,
            selected_nodes or [],
            selected_links or [],
            show_simulation_data=show_sim_data,
            sim_initialized=sim_initialized or False,
            height=map_height_int,
            node_size_scale=node_size_float,
            show_labels_always=show_labels_always
        )
        
        map_component = dcc.Graph(id="network-graph", figure=fig)
        
        # State visualization plot
        from modules.visualization import create_state_visualization_plot
        state_fig = create_state_visualization_plot(
            wn,
            height=map_height_int,
            node_size_scale=node_size_float,
            show_labels_always=show_labels_always
        )
        
        state_component = dcc.Graph(id="state-graph", figure=state_fig)
        
        # Create color legends if showing simulation data
        legends = ""
        if show_sim_data:
            # Get current data for legends (simplified for now)
            pressure_fig = create_pressure_colorbar(0, 100)  # Default range
            flow_fig = create_flow_colorbar(-10, 10)  # Default range
            
            legends = dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=pressure_fig, style={"height": "120px"})
                ], width=6),
                dbc.Col([
                    dcc.Graph(figure=flow_fig, style={"height": "120px"})
                ], width=6)
            ])
        
        return map_component, legends, state_component
    except Exception as e:
        print(f"Error in update_network_maps: {e}")
        error_alert = dbc.Alert(f"Error updating map: {str(e)}", color="danger")
        return error_alert, "", error_alert

# Element selection callback (from map clicks) - now opens modal
@app.callback(
    [Output('current-element', 'data'),
     Output('selected-nodes', 'data'),
     Output('selected-links', 'data'),
     Output('event-config-modal', 'is_open')],
    [Input('network-graph', 'clickData'),
     Input('state-graph', 'clickData')],
    [State('network-loaded', 'data'),
     State('network-metadata', 'data')],
    prevent_initial_call=True
)
def handle_map_clicks(primary_click_data, state_click_data, network_loaded, metadata):
    """Handle clicks on both network maps - opens modal for event configuration."""
    if not network_loaded:
        return None, [], [], False
    
    # Determine which graph was clicked
    ctx_triggered = ctx.triggered_id if hasattr(ctx, "triggered_id") else None
    click_data = None
    
    if ctx_triggered == 'network-graph' and primary_click_data:
        click_data = primary_click_data
    elif ctx_triggered == 'state-graph' and state_click_data:
        click_data = state_click_data
    
    if not click_data:
        return None, [], [], False
    
    # Extract element info from click data
    point = click_data['points'][0]
    
    # For state graph, we need to extract node names from the hovertext or text
    if ctx_triggered == 'state-graph':
        # Extract element name from hover text or other available data
        if 'hovertext' in point and point['hovertext']:
            # Parse hovertext to extract element name
            hover_text = point['hovertext']
            if '<b>' in hover_text and '</b>' in hover_text:
                element_name = hover_text.split('<b>')[1].split('</b>')[0]
                # Determine if it's a node or link based on available metadata
                if element_name in (metadata.get('node_names', []) if metadata else []):
                    element_category = 'node'
                    # Get node type from network
                    if global_state['wn'] and element_name in global_state['wn'].node_name_list:
                        node = global_state['wn'].get_node(element_name)
                        element_type = node.node_type
                    else:
                        element_type = 'Junction'  # Default
                else:
                    element_category = 'link'
                    # Get link type from network
                    if global_state['wn'] and element_name in global_state['wn'].link_name_list:
                        link = global_state['wn'].get_link(element_name)
                        element_type = link.link_type
                    else:
                        element_type = 'Pipe'  # Default
                
                current_element = {
                    'name': element_name,
                    'category': element_category,
                    'type': element_type
                }
                
                if element_category == 'node':
                    return current_element, [element_name], [], True  # Open modal
                else:  # link
                    return current_element, [], [element_name], True  # Open modal
    
    # Handle primary graph clicks (original logic)
    elif 'customdata' in point and point['customdata']:
        custom_data = point['customdata']
        element_name = custom_data[0]
        element_type = custom_data[1]
        element_category = custom_data[2]  # 'node' or 'link'
        
        current_element = {
            'name': element_name,
            'category': element_category,
            'type': element_type
        }
        
        if element_category == 'node':
            return current_element, [element_name], [], True  # Open modal
        else:  # link
            return current_element, [], [element_name], True  # Open modal
    
    return None, [], [], False

# Modal content callbacks
@app.callback(
    [Output('modal-title', 'children'),
     Output('modal-element-info', 'children'),
     Output('modal-element-properties', 'children'),
     Output('modal-event-configuration', 'children'),
     Output('modal-event-buttons', 'children')],
    Input('current-element', 'data'),
    State('network-loaded', 'data'),
    prevent_initial_call=True
)
def update_modal_content(current_element, network_loaded):
    """Update modal content when an element is selected."""
    if not network_loaded or not current_element or global_state['wn'] is None:
        return "⚡ Configure Event", "", "", "", ""
    
    try:
        from modules.dash_ui_components import (create_element_properties_display, 
                                              create_event_configuration_form)
        
        wn = global_state['wn']
        element_name = current_element['name']
        element_type = current_element['type']
        element_category = current_element['category']
        
        # Title
        title = f"⚡ Configure Event - {element_name} ({element_type})"
        
        # Element info
        element_info = dbc.Alert([
            html.Strong(f"Selected: {element_name}"),
            html.Br(),
            f"Type: {element_type} | Category: {element_category}"
        ], color="primary")
        
        # Element properties
        properties = create_element_properties_display(wn, element_name, element_type)
        
        # Event configuration form
        event_form = create_event_configuration_form(element_name, element_type, element_category)
        
        return title, element_info, properties, event_form, ""
        
    except Exception as e:
        error_content = dbc.Alert(f"Error loading element data: {str(e)}", color="danger")
        return "⚡ Configure Event", error_content, "", "", ""

# Modal close callback
@app.callback(
    Output('event-config-modal', 'is_open', allow_duplicate=True),
    Input('cancel-event-btn', 'n_clicks'),
    State('event-config-modal', 'is_open'),
    prevent_initial_call=True
)
def close_modal(cancel_clicks, is_open):
    """Close the modal when cancel button is clicked."""
    if cancel_clicks:
        return False
    return is_open

# Simulation control callbacks - Initialize
@app.callback(
    [Output('sim-initialized', 'data'),
     Output('sim-status', 'children'),
     Output('simulation-duration-hours', 'data'),
     Output('simulation-timestep-minutes', 'data')],
    Input('init-btn', 'n_clicks'),
    [State('duration-input', 'value'),
     State('timestep-select', 'value'),
     State('network-loaded', 'data'),
     State('sim-initialized', 'data')],
    prevent_initial_call=True
)
def handle_simulation_initialize(n_clicks, duration_hours, timestep_minutes, network_loaded, current_status):
    """Handle simulation initialize button."""
    if not n_clicks or not network_loaded or global_state['sim'] is None:
        return current_status or False, dbc.Alert("🟡 No network loaded", color="warning"), 24, 60
    
    try:
        sim = global_state['sim']
        
        # Initialize simulation
        duration_hours = int(duration_hours or 24)
        timestep_minutes = int(timestep_minutes or 60)
        
        # Validate duration before initialization
        max_reasonable_hours = 87600  # 10 years
        if duration_hours > max_reasonable_hours:
            error_status = dbc.Alert(
                [
                    html.H6("⚠️ Duration Too Large", className="alert-heading"),
                    html.P(f"Cannot initialize simulation with {duration_hours:,} hours."),
                    html.P(f"Please enter a duration less than {max_reasonable_hours:,} hours (10 years).", className="mb-0")
                ],
                color="warning"
            )
            return False, error_status, 24, 60
        
        duration_seconds = duration_hours * 3600
        timestep_seconds = timestep_minutes * 60
        
        print(f"Initializing simulation for {duration_seconds} hours with {timestep_seconds} minutes timestep.")
        
        sim.init_simulation(
            global_timestep=timestep_seconds,
            duration=duration_seconds
        )
        
        status = dbc.Alert("🟢 Simulation Initialized Successfully!", color="success")
        return True, status, duration_hours, timestep_minutes
            
    except Exception as e:
        error_status = dbc.Alert(f"❌ Initialization Error: {str(e)}", color="danger")
        print(f"Initialization error: {e}")
        print(duration_hours, timestep_minutes)
        return False, error_status, 24, 60

# Simulation control callbacks - Reset
@app.callback(
    [Output('sim-initialized', 'data', allow_duplicate=True),
     Output('sim-status', 'children', allow_duplicate=True),
     Output('current-sim-time', 'data', allow_duplicate=True),
     Output('simulation-data', 'data', allow_duplicate=True),
     Output('applied-events', 'data', allow_duplicate=True),
     Output('scheduled-events', 'data', allow_duplicate=True)],
    Input('reset-btn', 'n_clicks'),
    State('network-loaded', 'data'),
    prevent_initial_call=True
)
def handle_simulation_reset(n_clicks, network_loaded):
    """Handle simulation reset button."""
    if not n_clicks or not network_loaded or global_state['wn'] is None:
        return no_update, no_update, no_update, no_update, no_update, no_update
    
    try:
        # Reset simulation
        from modules.simulation import reset_simulation_state, initialize_simulation_data
        import copy
        # Restore the pristine network if available, otherwise reload from file if possible
        if 'original_wn' in global_state:
            global_state['wn'] = copy.deepcopy(global_state['original_wn'])
        elif 'network_file_path' in global_state:
            from modules.simulation import load_network_model
            try:
                global_state['wn'] = load_network_model(global_state['network_file_path'])
            except Exception:
                pass  # Fall back to existing wn if reload fails

        global_state['sim'] = reset_simulation_state(global_state['wn'])
        
        status = dbc.Alert("🔄 Simulation Reset Complete!", color="info")
        return False, status, 0, initialize_simulation_data(), [], []
            
    except Exception as e:
        error_status = dbc.Alert(f"❌ Reset Error: {str(e)}", color="danger")
        return False, error_status, no_update, no_update, no_update, no_update

# Simulation step callback
@app.callback(
    [Output('current-sim-time', 'data', allow_duplicate=True),
     Output('simulation-data', 'data', allow_duplicate=True),
     Output('applied-events', 'data', allow_duplicate=True),
     Output('scheduled-events', 'data', allow_duplicate=True),
     Output('status-messages-area', 'children', allow_duplicate=True)],
    [Input('step-btn', 'n_clicks'),
     Input('interactive-play-interval', 'n_intervals')],
    [State('sim-initialized', 'data'),
     State('current-sim-time', 'data'),
     State('simulation-data', 'data'),
     State('scheduled-events', 'data'),
     State('applied-events', 'data'),
     State('pressure-monitoring-nodes', 'data'),
     State('flow-monitoring-links', 'data')],
    prevent_initial_call=True
)
def handle_simulation_step(step_clicks, play_intervals, sim_initialized, current_time, sim_data, 
                           scheduled_events, applied_events, monitored_nodes, monitored_links):
    """Handle simulation step button."""
    triggered_id = ctx.triggered_id if hasattr(ctx, "triggered_id") else None

    # valid triggers: manual step or interval tick
    if triggered_id not in ["step-btn", "interactive-play-interval"] or not sim_initialized or global_state['sim'] is None:
        return no_update, no_update, no_update, no_update, no_update

    step_multiplier = 1  # always single step per trigger
 
    try:
        from modules.simulation import apply_event_to_simulator, run_simulation_step, collect_simulation_data, initialize_simulation_data
 
        sim = global_state['sim']
        wn = global_state['wn']
 
        # Initialize variables
        current_time = current_time or 0
        sim_data = sim_data or initialize_simulation_data()
        scheduled_events = scheduled_events or []
        applied_events = applied_events or []
        status_messages = []
        new_applied_events = list(applied_events)
        new_scheduled_events = list(scheduled_events)
        new_sim_data = dict(sim_data)

        # Execute the required number of steps
        for _ in range(step_multiplier):
            # Apply scheduled events that are due for the CURRENT time
            events_to_apply = [e for e in new_scheduled_events 
                              if e.get('scheduled_time', e.get('time', 0)) <= current_time]

            for event in events_to_apply:
                success, message = apply_event_to_simulator(sim, wn, event)
                if success:
                    new_applied_events.append(event)
                    new_scheduled_events.remove(event)
                    status_messages.append(dbc.Alert(f"⚡ Applied: {message}", color="success", dismissable=True))
                else:
                    status_messages.append(dbc.Alert(f"❌ Failed: {message}", color="danger", dismissable=True))

            # Run a single simulation step
            success, new_sim_time, message = run_simulation_step(sim, wn)

            if not success:
                status_messages.append(dbc.Alert(f"⚠️ Step warning: {message}", color="warning", dismissable=True))
                break

            # Update tracking variables
            current_time = new_sim_time
            new_sim_data['time'].append(new_sim_time)

            # Collect data for monitored elements
            monitored_nodes = monitored_nodes or list(wn.node_name_list)[:5]
            monitored_links = monitored_links or list(wn.link_name_list)[:5]
            collect_simulation_data(wn, monitored_nodes, monitored_links, new_sim_data)

        if not status_messages:
            status_messages.append(dbc.Alert("✅ Step completed successfully!", color="success", dismissable=True))

        return current_time, new_sim_data, new_applied_events, new_scheduled_events, status_messages
 
    except Exception as e:
        error_msg = dbc.Alert(f"❌ Step error: {str(e)}", color="danger", dismissable=True)
        return no_update, no_update, no_update, no_update, [error_msg]

# Update step button state
@app.callback(
    [Output('step-btn', 'disabled'),
     Output('play-step-btn', 'disabled'),
     Output('pause-step-btn', 'disabled')],
    Input('sim-initialized', 'data')
)
def enable_manual_controls(sim_initialized):
    """Enable Step and Play when simulation ready."""
    disabled_state = not (sim_initialized or False)
    return disabled_state, disabled_state, disabled_state

# Play / Pause toggle callback
@app.callback(
    [Output('interactive-play-interval', 'disabled'),
     Output('play-step-btn', 'disabled', allow_duplicate=True),
     Output('pause-step-btn', 'disabled', allow_duplicate=True)],
    [Input('play-step-btn', 'n_clicks'),
     Input('pause-step-btn', 'n_clicks')],
    State('interactive-play-interval', 'disabled'),
    prevent_initial_call=True
)
def control_auto_stepping(play_clicks, pause_clicks, interval_disabled):
    ctx_id = ctx.triggered_id if hasattr(ctx, 'triggered_id') else None
    if ctx_id == 'play-step-btn':
        # start auto stepping
        return False, True, False  # interval enabled, play disabled, pause enabled
    elif ctx_id == 'pause-step-btn':
        return True, False, True   # interval disabled, play enabled, pause disabled
    # no change
    return interval_disabled, dash.no_update, dash.no_update

# Duration input validation callback
@app.callback(
    [Output('duration-input', 'value'),
     Output('status-messages-area', 'children', allow_duplicate=True)],
    Input('duration-input', 'value'),
    prevent_initial_call=True
)
def validate_duration_input(duration_value):
    """Validate duration input and enforce maximum limit."""
    if duration_value is None:
        return 24, dash.no_update  # Default value
    
    max_hours = 87600  # 10 years
    
    # If user enters more than max, reset to max value
    if duration_value > max_hours:
        warning_msg = dbc.Alert(
            f"⚠️ Duration capped at maximum: {max_hours:,} hours (10 years)",
            color="warning",
            dismissable=True,
            duration=4000
        )
        return max_hours, warning_msg
    
    # If user enters less than min, reset to min value  
    if duration_value < 1:
        warning_msg = dbc.Alert(
            "⚠️ Duration set to minimum: 1 hour",
            color="warning", 
            dismissable=True,
            duration=4000
        )
        return 1, warning_msg
        
    return duration_value, dash.no_update

# Simulation progress display callback
@app.callback(
    Output('simulation-progress-display', 'children'),
    [Input('sim-initialized', 'data'),
     Input('current-sim-time', 'data'),
     Input('simulation-data', 'data'),
     Input('simulation-duration-hours', 'data'),
     Input('simulation-timestep-minutes', 'data')],
    prevent_initial_call=True
)
def update_simulation_progress(sim_initialized, current_time, sim_data, duration_hours, timestep_minutes):
    """Display professional simulation progress bar."""
    if not sim_initialized:
        return dbc.Alert("🎯 Initialize simulation to see progress", color="info", className="text-center")
    
    try:
        import datetime
        
        # Calculate simulation parameters using stored values from initialization
        total_duration_seconds = (duration_hours or 24) * 3600
        timestep_seconds = (timestep_minutes or 60) * 60
        current_time = current_time or 0
        
        # Check if duration exceeds datetime.timedelta limits
        # timedelta max value is about 999,999,999 days (around 2.7 million years)
        # But practically, let's limit to something reasonable like 10 years
        max_reasonable_seconds = 10 * 365 * 24 * 3600  # 10 years
        
        if total_duration_seconds > max_reasonable_seconds:
            return dbc.Alert(
                [
                    html.H6("⚠️ Duration Too Large", className="alert-heading"),
                    html.P(f"The entered duration ({duration_hours:,} hours) exceeds reasonable limits."),
                    html.P("Please enter a duration less than 87,600 hours (10 years)."),
                    html.P("For very long simulations, consider using batch mode with shorter time segments.", className="mb-0")
                ],
                color="warning"
            )
        
        # Try to create timedelta objects to ensure they're valid
        try:
            total_time_display = str(datetime.timedelta(seconds=int(total_duration_seconds)))
            current_time_display = str(datetime.timedelta(seconds=int(current_time)))
            remaining_time = max(0, total_duration_seconds - current_time)
            remaining_time_display = str(datetime.timedelta(seconds=int(remaining_time)))
        except (ValueError, OverflowError):
            return dbc.Alert(
                [
                    html.H6("⚠️ Duration Invalid", className="alert-heading"),
                    html.P("The entered duration cannot be processed by the system."),
                    html.P("Please enter a smaller duration value.", className="mb-0")
                ],
                color="danger"
            )
        
        # Calculate progress
        time_progress = min(current_time / total_duration_seconds, 1.0)
        completed_steps = len((sim_data or {}).get('time', []))
        planned_total_steps = int(total_duration_seconds / timestep_seconds)
        step_progress = min(completed_steps / planned_total_steps, 1.0) if planned_total_steps > 0 else 0
        
        # Progress components
        progress_components = [
            html.H6("📊 Simulation Progress", className="text-primary mb-3"),
            
            # Progress metrics row
            dbc.Row([
                dbc.Col([
                    html.H6(f"{time_progress*100:.1f}%", className="text-success mb-0"),
                    html.Small("Complete", className="text-muted")
                ], width=3),
                dbc.Col([
                    html.H6(f"{completed_steps}", className="text-primary mb-0"),
                    html.Small("Steps Done", className="text-muted")
                ], width=3),
                dbc.Col([
                    html.H6(current_time_display, className="text-info mb-0"),
                    html.Small("Current Time", className="text-muted")
                ], width=3),
                dbc.Col([
                    html.H6(remaining_time_display, className="text-warning mb-0"),
                    html.Small("Remaining", className="text-muted")
                ], width=3)
            ], className="mb-3"),
            
            # Progress bars
            html.Div([
                html.Small(f"Time Progress: {current_time_display} / {total_time_display}", className="text-muted"),
                dbc.Progress(
                    value=time_progress*100, 
                    label=f"{time_progress*100:.1f}%",
                    color="success" if time_progress > 0.8 else "warning" if time_progress > 0.5 else "primary",
                    className="mb-2"
                ),
                html.Small(f"Step Progress: {completed_steps} / {planned_total_steps}", className="text-muted"),
                dbc.Progress(
                    value=step_progress*100,
                    label=f"{completed_steps} steps",
                    color="info",
                    className="mb-2"
                )
            ])
        ]
        
        # Add completion message if done
        if time_progress >= 1.0:
            progress_components.append(
                dbc.Alert("🎉 Simulation Complete!", color="success", className="mt-2")
            )
        
        return progress_components
        
    except Exception as e:
        return dbc.Alert(f"Error calculating progress: {str(e)}", color="danger")



# Real-time monitoring charts display callback - SIMPLIFIED WITH MULTI-SELECT
@app.callback(
    Output('simulation-results-container', 'children'),
    [Input('simulation-data', 'data'),
     Input('current-sim-time', 'data'),
     Input('sim-initialized', 'data'),
     Input('pressure-monitoring-nodes', 'data'),
     Input('flow-monitoring-links', 'data'),
     Input('network-metadata', 'data')],
    prevent_initial_call=True
)
def update_simulation_results(sim_data, current_time, sim_initialized, monitored_nodes, monitored_links, metadata):
    """Update real-time monitoring charts display with simple multi-select controls."""
    if not sim_initialized:
        return dbc.Alert("🚀 Initialize simulation to see real-time monitoring charts", color="info")
    
    # Use default selections if none provided
    if not monitored_nodes and global_state['wn'] is not None:
        monitored_nodes = list(global_state['wn'].node_name_list)[:3]
    if not monitored_links and global_state['wn'] is not None:
        monitored_links = list(global_state['wn'].link_name_list)[:3]
    
    try:
        # Create simple monitoring charts with multi-select functionality
        available_nodes = metadata.get('node_names', []) if metadata else []
        available_links = metadata.get('link_names', []) if metadata else []
        
        fig_pressure = create_simple_monitoring_chart(available_nodes, monitored_nodes or [], "pressure", sim_data)
        fig_flow = create_simple_monitoring_chart(available_links, monitored_links or [], "flow", sim_data)
        
        # Current time display
        time_display = f"⏱️ Current Time: {datetime.timedelta(seconds=int(current_time or 0))}" if current_time else "⏱️ Time: 00:00:00"
        
        results_content = [
            dbc.Row([
                dbc.Col([
                    html.H4("📊 Interactive Real-Time Monitoring", className="text-primary"),
                ], width=8),
                dbc.Col([
                    html.H6(time_display, className="text-muted text-end")
                ], width=4)
            ], className="mb-3"),
            
            # SIMPLIFIED Multi-Select Controls (From plotly_restyle_test.py)
            dbc.Card([
                dbc.CardHeader("🎛️ Multi-Select Monitoring Controls"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("🔵 Select Nodes for Pressure Monitoring"),
                            dcc.Dropdown(
                                id="interactive-pressure-multiselect",
                                options=[{"label": node, "value": node} for node in available_nodes],
                                value=monitored_nodes or [],
                                multi=True,
                                placeholder="Select nodes to monitor...",
                                style={"marginBottom": "10px"}
                            )
                        ], width=6),
                        dbc.Col([
                            html.H5("🔗 Select Links for Flow Monitoring"),
                            dcc.Dropdown(
                                id="interactive-flow-multiselect",
                                options=[{"label": link, "value": link} for link in available_links],
                                value=monitored_links or [],
                                multi=True,
                                placeholder="Select links to monitor...",
                                style={"marginBottom": "10px"}
                            )
                        ], width=6)
                    ])
                ])
            ], className="mb-4"),
            
            # Monitoring charts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("📈 Pressure Monitoring Chart", className="mb-0 text-primary")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(
                                id="pressure-monitoring-chart",
                                figure=fig_pressure, 
                                style={"height": "350px"},
                                config={'displayModeBar': True, 'displaylogo': False}
                            ) if fig_pressure else dbc.Alert("📍 Select nodes above to monitor pressure", color="info")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("📈 Flow Monitoring Chart", className="mb-0 text-primary")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(
                                id="flow-monitoring-chart",
                                figure=fig_flow,
                                style={"height": "350px"},
                                config={'displayModeBar': True, 'displaylogo': False}
                            ) if fig_flow else dbc.Alert("🔗 Select links above to monitor flow", color="info")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
                            
            # Summary statistics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("📈 Statistics", className="card-title"),
                            html.P(f"Steps Completed: {len((sim_data or {}).get('time', []))}", className="mb-1"),
                            html.P(f"Nodes Monitored: {len(monitored_nodes or [])}", className="mb-1"),
                            html.P(f"Links Monitored: {len(monitored_links or [])}", className="mb-0")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("🎯 Current Values", className="card-title"),
                            html.Div(id="current-values-display")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Event History Section
            dbc.Row([
                dbc.Col([
                    html.Div(id="event-history-container")
                ], width=12)
            ])
        ]
        
        return results_content
        
    except Exception as e:
        return dbc.Alert(f"Error displaying monitoring charts: {str(e)}", color="danger")

# Enhanced dynamic event parameters callback
# NEW CALLBACK: Handle multi-select dropdowns for interactive mode
@app.callback(
    [Output('pressure-monitoring-nodes', 'data', allow_duplicate=True),
     Output('flow-monitoring-links', 'data', allow_duplicate=True)],
    [Input('interactive-pressure-multiselect', 'value'),
     Input('interactive-flow-multiselect', 'value')],
    prevent_initial_call=True
)
def update_monitoring_from_multiselect(selected_nodes, selected_links):
    """Update monitoring stores based on multi-select dropdowns."""
    return selected_nodes or [], selected_links or []


@app.callback(
    Output({'type': 'event-params-container', 'element': MATCH}, 'children'),
    Input({'type': 'event-type-select', 'element': MATCH}, 'value'),
    State('current-element', 'data'),
    prevent_initial_call=True
)
def update_event_parameters(event_type, current_element):
    """Generate dynamic parameter inputs based on selected event type with all possible parameters."""
    if not event_type or not current_element:
        return []
    
    try:
        # Complete parameter definitions for ALL event types from MWNTR codebase
        event_parameters = {
            # Node Events - Junction Events
            'start_leak': {
                'params': ['leak_area', 'leak_discharge_coefficient'],
                'defaults': [0.01, 0.75],
                'descriptions': [
                    'Leak area in square meters (m²) - physical size of hole/crack',
                    'Discharge coefficient (0-1) - orifice efficiency, 0.75 is standard for sharp-edged orifices'
                ],
                'types': ['number', 'number'],
                'steps': [0.001, 0.01]
            },
            'stop_leak': {
                'params': [],
                'defaults': [],
                'descriptions': [],
                'types': [],
                'steps': []
            },
            'add_demand': {
                'params': ['base_demand', 'pattern_name', 'category'],
                'defaults': [0.1, 'default_pattern', 'user_added'],
                'descriptions': [
                    'Base water demand in cubic meters per second (m³/s) - baseline consumption rate',
                    'Pattern name for demand multipliers over time - references existing or creates new pattern',
                    'Demand category for grouping (e.g., "residential", "industrial") - used for organization'
                ],
                'types': ['number', 'text', 'text'],
                'steps': [0.001, None, None]
            },
            'remove_demand': {
                'params': ['name'],
                'defaults': ['user_added'],
                'descriptions': [
                    'Pattern name or category of demand to remove - must match existing demand entry'
                ],
                'types': ['text'],
                'steps': [None]
            },
            'add_fire_fighting_demand': {
                'params': ['fire_flow_demand', 'fire_start', 'fire_end'],
                'defaults': [0.5, 300, 1800],
                'descriptions': [
                    'Fire flow demand in m³/s - high water consumption for fire suppression (typically 0.1-1.0)',
                    'Fire start time in seconds from simulation start - when fire fighting begins',
                    'Fire end time in seconds from simulation start - when fire fighting ends'
                ],
                'types': ['number', 'number', 'number'],
                'steps': [0.01, 60, 60]
            },
            
            # Node Events - Tank Events  
            'set_tank_head': {
                'params': ['head'],
                'defaults': [50.0],
                'descriptions': [
                    'Tank water level in meters (m) - absolute height from tank bottom'
                ],
                'types': ['number'],
                'steps': [1.0]
            },
            
            # Link Events - Pipe Events
            'close_pipe': {
                'params': [],
                'defaults': [],
                'descriptions': [],
                'types': [],
                'steps': []
            },
            'open_pipe': {
                'params': [],
                'defaults': [],
                'descriptions': [],
                'types': [],
                'steps': []
            },
            'set_pipe_diameter': {
                'params': ['diameter'],
                'defaults': [0.3],
                'descriptions': [
                    'Internal pipe diameter in meters (m) - affects flow capacity and pressure loss'
                ],
                'types': ['number'],
                'steps': [0.01]
            },
            
            # Link Events - Pump Events
            'close_pump': {
                'params': [],
                'defaults': [],
                'descriptions': [],
                'types': [],
                'steps': []
            },
            'open_pump': {
                'params': [],
                'defaults': [],
                'descriptions': [],
                'types': [],
                'steps': []
            },
            'set_pump_speed': {
                'params': ['speed'],
                'defaults': [1.0],
                'descriptions': [
                    'Pump speed multiplier - 1.0 = normal speed, 0.5 = half speed, 1.5 = 150% speed'
                ],
                'types': ['number'],
                'steps': [0.1]
            },
            'set_pump_head_curve': {
                'params': ['head_curve'],
                'defaults': ['pump_head_curve_1'],
                'descriptions': [
                    'Head curve name defining pump performance - must exist in network or be created'
                ],
                'types': ['text'],
                'steps': [None]
            },
            
            # Link Events - Valve Events
            'close_valve': {
                'params': [],
                'defaults': [],
                'descriptions': [],
                'types': [],
                'steps': []
            },
            'open_valve': {
                'params': [],
                'defaults': [],
                'descriptions': [],
                'types': [],
                'steps': []
            }
        }
        
        if event_type not in event_parameters:
            return [dbc.Alert(f"Event type {event_type} not configured", color="warning")]
        
        event_config = event_parameters[event_type]
        params = event_config['params']
        defaults = event_config['defaults']
        descriptions = event_config['descriptions']
        types = event_config['types']
        steps = event_config['steps']
        
        if not params:
            return [dbc.Alert("✅ This event requires no additional parameters", color="success")]
        
        # Create enhanced input fields for each parameter
        param_inputs = [html.H6("📝 Event Parameters", className="text-primary mt-3 mb-2")]
        
        for i, param_name in enumerate(params):
            default_value = defaults[i] if i < len(defaults) else ""
            description = descriptions[i] if i < len(descriptions) else param_name.replace('_', ' ').title()
            input_type = types[i] if i < len(types) else 'text'
            step = steps[i] if i < len(steps) else None
            
            # Create parameter input with enhanced styling
            if input_type == 'number':
                param_input = dbc.Input(
                    type="number",
                    value=default_value,
                    step=step,
                    id={"type": f"param-{param_name}", "element": current_element['name']},
                    className="mb-2"
                )
            else:
                # Handle None values and provide proper default display
                display_value = ""
                if default_value is not None:
                    display_value = str(default_value)
                
                param_input = dbc.Input(
                    type="text",
                    value=display_value,
                    placeholder=f"Enter {param_name.replace('_', ' ').lower()}..." if not display_value else None,
                    id={"type": f"param-{param_name}", "element": current_element['name']},
                    className="mb-2"
                )
            
            # Enhanced parameter input with help text
            param_inputs.extend([
                dbc.Label([
                    html.Strong(param_name.replace('_', ' ').title()),
                    html.Br(),
                    html.Small(description, className="text-muted")
                ], className="fw-bold mt-2"),
                param_input
            ])
        
        # Add helpful information
        param_inputs.extend([
            html.Hr(),
            dbc.Alert([
                html.I(className="fas fa-info-circle me-2"),
                f"Event will be applied immediately to {current_element['name']} ({current_element['type']})"
            ], color="info", className="mt-3")
        ])
        
        return param_inputs
        
    except Exception as e:
        return [dbc.Alert(f"Error loading parameters: {str(e)}", color="danger")]

# Event scheduling callback
@app.callback(
    [Output('scheduled-events', 'data', allow_duplicate=True),
     Output('status-messages-area', 'children', allow_duplicate=True),
     Output('event-config-modal', 'is_open', allow_duplicate=True),
     Output('event-notification-store', 'data')],
    Input({'type': 'apply-event-btn', 'element': ALL}, 'n_clicks'),
    [State({'type': 'event-type-select', 'element': ALL}, 'value'),
     # All possible event parameters from our enhanced configuration
     State({'type': 'param-leak_area', 'element': ALL}, 'value'),
     State({'type': 'param-leak_discharge_coefficient', 'element': ALL}, 'value'),
     State({'type': 'param-base_demand', 'element': ALL}, 'value'),
     State({'type': 'param-pattern_name', 'element': ALL}, 'value'),
     State({'type': 'param-category', 'element': ALL}, 'value'),
     State({'type': 'param-fire_flow_demand', 'element': ALL}, 'value'),
     State({'type': 'param-fire_start', 'element': ALL}, 'value'),
     State({'type': 'param-fire_end', 'element': ALL}, 'value'),
     State({'type': 'param-name', 'element': ALL}, 'value'),
     State({'type': 'param-head', 'element': ALL}, 'value'),
     State({'type': 'param-diameter', 'element': ALL}, 'value'),
     State({'type': 'param-speed', 'element': ALL}, 'value'),
     State({'type': 'param-head_curve', 'element': ALL}, 'value'),
     State('current-element', 'data'),
     State('scheduled-events', 'data')],
    prevent_initial_call=True
)
def handle_event_application(btn_clicks, event_types, 
                           leak_area, leak_discharge_coeff, base_demand, pattern_name, category,
                           fire_flow_demand, fire_start, fire_end, name, head, diameter, speed, 
                           head_curve, current_element, scheduled_events):
    """Handle immediate event application from the event configuration form."""
    if not any(btn_clicks) or not current_element:
        return scheduled_events or [], dash.no_update, dash.no_update, dash.no_update
    
    try:
        # Find which button was clicked
        triggered_idx = None
        for i, clicks in enumerate(btn_clicks):
            if clicks:
                triggered_idx = i
                break
        
        if triggered_idx is None or not event_types[triggered_idx]:
            return scheduled_events or [], dash.no_update, dash.no_update, dash.no_update
        
        # Collect parameters for this specific element/event
        parameters = {}
        param_lists = {
            'leak_area': leak_area,
            'leak_discharge_coefficient': leak_discharge_coeff,
            'base_demand': base_demand,
            'pattern_name': pattern_name,
            'category': category,
            'fire_flow_demand': fire_flow_demand,
            'fire_start': fire_start,
            'fire_end': fire_end,
            'head': head,
            'diameter': diameter,
            'speed': speed,
            'head_curve': head_curve,
            'name': name
        }
        
        # Extract parameters for the triggered element (index triggered_idx)
        for param_name, param_values in param_lists.items():
            if param_values and triggered_idx < len(param_values) and param_values[triggered_idx] is not None:
                parameters[param_name] = param_values[triggered_idx]
        
        # Create and apply the event immediately
        from modules.simulation import create_event, apply_event_to_simulator
        
        # Use current simulation time (or 0 if not initialized)
        current_sim_time = global_state.get('current_sim_time', 0)
        
        event = create_event(
            element_name=current_element['name'],
            element_type=current_element['type'],
            element_category=current_element['category'],
            event_type=event_types[triggered_idx],
            scheduled_time=current_sim_time,
            parameters=parameters
        )
        
        if event:
            param_str = ", ".join([f"{k}={v}" for k, v in parameters.items()]) if parameters else "no parameters"
            # If simulation is running, apply immediately, otherwise add to scheduled events
            if global_state.get('sim') and global_state.get('sim_initialized'):
                # Apply event immediately
                sim = global_state['sim']
                wn = global_state['wn']
                success, message = apply_event_to_simulator(sim, wn, event)
                
                if success:
                    # Add to applied events instead of scheduled
                    new_applied_events = global_state.get('applied_events', []) + [event]
                    global_state['applied_events'] = new_applied_events
                    notification = {
                        "title": "Event Applied",
                        "message": f"Applied {event['event_type']} to {event['element_name']} immediately! ({param_str})",
                        "status": "success"
                    }
                    return scheduled_events or [], dash.no_update, False, notification
                else:
                    notification = {
                        "title": "Application Failed",
                        "message": f"Failed to apply event: {message}",
                        "status": "danger"
                    }
                    return scheduled_events or [], dash.no_update, dash.no_update, notification
            else:
                # Add to scheduled events for later application
                new_scheduled_events = (scheduled_events or []) + [event]
                notification = {
                    "title": "Event Scheduled",
                    "message": f"Event scheduled: {event['event_type']} on {event['element_name']} ({param_str})",
                    "status": "success"
                }
                return new_scheduled_events, dash.no_update, False, notification
        
        return scheduled_events or [], dash.no_update, dash.no_update, dash.no_update
        
    except Exception as e:
        notification = {
            "title": "Error",
            "message": f"Error scheduling event: {str(e)}",
            "status": "danger"
        }
        return scheduled_events or [], dash.no_update, dash.no_update, notification


# Callback to display toasts for event notifications
@app.callback(
    Output('toast-container', 'children'),
    Input('event-notification-store', 'data'),
    prevent_initial_call=True
)
def show_event_notification(notification):
    if not notification:
        return dash.no_update
    
    message = notification.get('message', '')
    title = notification.get('title', 'Notification')
    status = notification.get('status', 'info') # success, danger, warning, info
    
    toast = dbc.Toast(
        message,
        header=title,
        is_open=True,
        dismissable=True,
        duration=5000,
        icon=status,
    )
    return toast

# Update simulation status display with current time
@app.callback(
    Output('sim-status', 'children', allow_duplicate=True),
    [Input('current-sim-time', 'data'),
     Input('sim-initialized', 'data')],
    prevent_initial_call=True
)
def update_simulation_status_display(current_time, sim_initialized):
    """Update simulation status display with current time."""
    try:
        from modules.dash_ui_components import create_simulation_status_display
        return create_simulation_status_display(sim_initialized or False, current_time or 0)
    except Exception as e:
        return dbc.Alert(f"Status error: {str(e)}", color="warning")

# Event history display callback
@app.callback(
    Output('event-history-container', 'children'),
    [Input('scheduled-events', 'data'),
     Input('applied-events', 'data'),
     Input('current-sim-time', 'data'),
     Input('sim-initialized', 'data')],
    prevent_initial_call=True
)
def update_event_history_display(scheduled_events, applied_events, current_time, sim_initialized):
    """Create professional event history display."""
    if not sim_initialized:
        return []
    
    try:
        from modules.dash_ui_components import create_professional_event_history
        return create_professional_event_history(scheduled_events or [], applied_events or [], current_time or 0)
    except Exception as e:
        return [dbc.Alert(f"Error displaying event history: {str(e)}", color="danger")]

# Current values display callback
@app.callback(
    Output('current-values-display', 'children'),
    [Input('current-sim-time', 'data'),
     Input('sim-initialized', 'data'),
     Input('pressure-monitoring-nodes', 'data'),
     Input('flow-monitoring-links', 'data')],
    prevent_initial_call=True
)
def update_current_values_display(current_time, sim_initialized, monitored_nodes, monitored_links):
    """Update current values display for monitored elements."""
    if not sim_initialized or global_state['wn'] is None:
        return html.P("No data available", className="text-muted")
    
    try:
        from modules.simulation import get_current_element_values
        wn = global_state['wn']
        
        # Get current values
        current_pressures, current_flows = get_current_element_values(
            wn, monitored_nodes or [], monitored_links or []
        )
        
        elements = []
        
        # Show current pressures
        if monitored_nodes and current_pressures:
            for node_name in (monitored_nodes or [])[:3]:  # Show max 3 for space
                if node_name in current_pressures:
                    press_val = current_pressures[node_name]
                    try:
                        press_text = f"{float(press_val):.2f} m"
                    except (TypeError, ValueError):
                        press_text = "N/A"
                    elements.append(
                        html.P(f"🔵 {node_name}: {press_text}", className="mb-1 small")
                    )
        
        # Show current flows
        if monitored_links and current_flows:
            for link_name in (monitored_links or [])[:3]:  # Show max 3 for space
                if link_name in current_flows:
                    flow_val = current_flows[link_name]
                    try:
                        flow_text = f"{float(flow_val):.4f} m³/s"
                    except (TypeError, ValueError):
                        flow_text = "N/A"
                    elements.append(
                        html.P(f"🔗 {link_name}: {flow_text}", className="mb-1 small")
                    )
        
        if not elements:
            return html.P("Select elements to monitor", className="text-muted small")
        
        return elements
        
    except Exception as e:
        return html.P(f"Error: {str(e)}", className="text-danger small")

# Simplified batch events loading callback
@app.callback(
    [Output('batch-events', 'data'),
     Output('batch-metadata', 'data'),
     Output('status-messages-area', 'children', allow_duplicate=True)],
    Input('load-events-btn', 'n_clicks'),
    [State('example-events-select', 'value'),
     State('upload-events-file', 'contents'),
     State('upload-events-file', 'filename')],
    prevent_initial_call=True
)
def load_batch_events(n_clicks, example_file_value, upload_contents, upload_filename):
    """Load events from example JSON file."""
    if not n_clicks:
        return [], {}, ""
    
    import json, os
    try:
        path_to_load = example_file_value or "generated_events_NET_4_20250629_155628.json"
        
        with open(path_to_load, 'r') as f:
            data = json.load(f)
        
        if 'events' in data and 'metadata' in data:
            events = data['events']
            metadata = data['metadata']
        else:
            events = data if isinstance(data, list) else []
            metadata = {}
        
        if events:
            success_msg = dbc.Alert(
                f"✅ Loaded {len(events)} events from {path_to_load}! Duration: {metadata.get('duration_hours', 'N/A')} hours",
                color="success", dismissable=True
            )
            return events, metadata, success_msg
        else:
            error_msg = dbc.Alert("❌ No events found in file", color="danger")
            return [], {}, error_msg
    except Exception as e:
        error_msg = dbc.Alert(f"❌ Error loading events: {str(e)}", color="danger")
        return [], {}, error_msg

# Download example events file callback
@app.callback(
    Output('download-example-events', 'data'),
    Input('download-events-btn', 'n_clicks'),
    State('example-events-select', 'value'),
    prevent_initial_call=True
)
def download_example_events(n_clicks, selected_example):
    if not n_clicks or not selected_example:
        return dash.no_update
    import os
    from dash import dcc
    file_path = selected_example
    if not os.path.exists(file_path):
        return dash.no_update
    return dcc.send_file(file_path)

# Batch simulation main area callback
@app.callback(
    Output('batch-main-area', 'children'),
    [Input('network-loaded', 'data'),
     Input('batch-events', 'data'),
     Input('batch-metadata', 'data')],
    prevent_initial_call=True
)
def display_batch_main_area(network_loaded, batch_events, batch_metadata):
    """Display the main batch simulation area when network and events are loaded."""
    if not network_loaded:
        return dbc.Alert("👆 Please load a network file first", color="info")
    
    if not batch_events:
        return dbc.Alert("📋 Please load event file to begin batch simulation", color="info")
    
    # Calculate total simulation duration from events
    max_event_time = max([event.get('time', 0) for event in batch_events]) if batch_events else 3600
    total_duration_hours = max_event_time / 3600 + 1  # Add 1 hour buffer
    
    return [
        dbc.Row([
            # Animation controls column
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("🎬 Animation Controls")),
                    dbc.CardBody([
                        # Animation status
                        html.Div(id="batch-animation-status", className="mb-3"),
                        
                        # Progress bar
                        html.Div(id="batch-progress-display", className="mb-3"),
                        
                        # Animation settings
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Speed", className="fw-bold"),
                                dbc.Select([
                                    {"label": "0.5x", "value": 0.5},
                                    {"label": "1x", "value": 1.0},
                                    {"label": "2x", "value": 2.0},
                                    {"label": "5x", "value": 5.0},
                                    {"label": "10x", "value": 10.0}
                                ], value=1.0, id="batch-speed-select")
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Step Size (min)", className="fw-bold"),
                                dbc.Select([
                                    {"label": "15 min", "value": 15},
                                    {"label": "30 min", "value": 30},
                                    {"label": "60 min", "value": 60},
                                    {"label": "120 min", "value": 120}
                                ], value=60, id="batch-timestep-select")
                            ], width=6)
                        ], className="mb-3"),
                        
                        # Control buttons
                        dbc.ButtonGroup([
                            dbc.Button("▶️ Play", id="batch-play-btn", color="success"),
                            dbc.Button("⏸️ Pause", id="batch-pause-btn", color="warning"),
                            dbc.Button("🔄 Reset", id="batch-reset-btn", color="secondary")
                        ], className="w-100 mb-3"),
                        
                        # Quick jump removed (hidden to avoid callback errors)
                        html.Div([
                            dbc.Input(id="batch-jump-time-input", style={"display":"none"}),
                            dbc.Button(id="batch-jump-btn", style={"display":"none"})
                        ], style={"display":"none"})
                    ])
                ], className="mb-3"),
                
                # Event timeline
                dbc.Card([
                    dbc.CardHeader(html.H5("📅 Event Timeline")),
                    dbc.CardBody([
                        html.Div(id="batch-event-timeline")
                    ])
                ]),
                # Event history list as collapsible accordion
                dbc.Accordion([
                    dbc.AccordionItem(
                        html.Div(id="batch-event-history-container"),
                        title="🕑 Event History"
                    )
                ], start_collapsed=True)
            ], width=4),
            
            # Map and monitoring column
            dbc.Col([
                # Map controls
                dbc.Card([
                    dbc.CardBody([
                        dbc.Alert("🎬 Batch simulation shows animated network changes over time", 
                                color="info", className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Map Height"),
                                dbc.Select([
                                    {"label": "600px", "value": 600},
                                    {"label": "700px", "value": 700},
                                    {"label": "800px", "value": 800}
                                ], value=700, id="batch-map-height-select")
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Node Size"),
                                dbc.Select([
                                    {"label": "80%", "value": 0.8},
                                    {"label": "100%", "value": 1.0},
                                    {"label": "120%", "value": 1.2}
                                ], value=1.0, id="batch-node-size-select")
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Labels"),
                                dbc.ButtonGroup([
                                    dbc.Button("Always", id="batch-labels-always-btn", color="primary", size="sm"),
                                    dbc.Button("Hover", id="batch-labels-hover-btn", color="outline-primary", size="sm")
                                ], className="w-100")
                            ], width=3),
                            dbc.Col([
                                dbc.Label("View"),
                                dbc.ButtonGroup([
                                    dbc.Button("Dual", id="batch-dual-view-btn", color="success", size="sm"),
                                    dbc.Button("Single", id="batch-single-view-btn", color="outline-success", size="sm")
                                ], className="w-100")
                            ], width=3)
                        ])
                    ])
                ], className="mb-3"),
                
                # Primary Network Map (Pressure/Flow) - Always shown
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H6("🌊 Pressure & Flow", className="mb-0")),
                            dbc.CardBody([
                                html.Div(id="batch-network-map-container"),
                                html.Div(id="batch-color-legends-container", className="mt-2")
                            ])
                        ])
                    ], width=12, id="batch-primary-map-col")
                ], className="mb-3"),
                
                # Secondary State Map - Shown below in dual view
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.H6("🎯 Operational States", className="mb-0")),
                            dbc.CardBody([
                                html.Div(id="batch-state-map-container"),
                                dbc.Alert("🔵 Demands | 🔴 Leaks | ❌ Closed", 
                                        color="light", className="mt-2 small")
                            ])
                        ])
                    ], width=12, id="batch-state-map-col")
                ], id="batch-state-map-row")
            ], width=8)
        ]),
        
        # Results section
        html.Hr(),
        html.Div(id="batch-simulation-results-container"),
        
        # Animation interval component
        dcc.Interval(
            id='batch-animation-interval',
            interval=2000,  # 2 seconds default
            n_intervals=0,
            disabled=True
        )
    ]

# Batch animation control callbacks
@app.callback(
    [Output('batch-animation-running', 'data'),
     Output('batch-animation-interval', 'disabled')],
    [Input('batch-play-btn', 'n_clicks'),
     Input('batch-pause-btn', 'n_clicks')],
    State('batch-animation-running', 'data'),
    prevent_initial_call=True
)
def control_batch_animation(play_clicks, pause_clicks, is_running):
    """Control batch animation play/pause."""
    ctx_triggered = ctx.triggered[0]['prop_id'] if ctx.triggered else ""
    
    print(f"Batch animation control triggered: {ctx_triggered}, is_running: {is_running}")
    
    if 'batch-play-btn' in ctx_triggered:
        print("Play button clicked - starting animation")
        return True, False  # Start animation
    elif 'batch-pause-btn' in ctx_triggered:
        print("Pause button clicked - pausing animation")
        return False, True  # Pause animation
    
    # Default: keep animation stopped and interval disabled
    return False, True

# Batch animation reset callback
@app.callback(
    [Output('batch-current-time', 'data'),
     Output('batch-simulation-data', 'data', allow_duplicate=True),
     Output('batch-applied-events', 'data', allow_duplicate=True),
     Output('batch-animation-running', 'data', allow_duplicate=True),
     Output('batch-animation-interval', 'disabled', allow_duplicate=True)],
    Input('batch-reset-btn', 'n_clicks'),
    State('network-loaded', 'data'),
    prevent_initial_call=True
)
def reset_batch_animation(n_clicks, network_loaded):
    """Reset batch animation to start."""
    if not n_clicks or not network_loaded:
        return no_update, no_update, no_update, no_update, no_update
    
    try:
        # Reset simulation state and re-initialize for batch mode
        if global_state['wn'] is not None:
            from modules.simulation import reset_simulation_state, initialize_simulation_data
            from mwntr.sim.interactive_network_simulator import MWNTRInteractiveSimulator
            
            import copy
            if 'original_wn' in global_state:
                global_state['wn'] = copy.deepcopy(global_state['original_wn'])
            elif 'network_file_path' in global_state:
                from modules.simulation import load_network_model
                try:
                    global_state['wn'] = load_network_model(global_state['network_file_path'])
                except Exception:
                    pass

            # Create fresh simulator instance
            global_state['sim'] = MWNTRInteractiveSimulator(global_state['wn'])
            
            # Initialize the simulator with proper settings for batch mode
            duration_seconds = 24 * 3600  # 24 hours default
            timestep_seconds = 60 * 60   # 1 hour timestep
            
            global_state['sim'].init_simulation(
                global_timestep=timestep_seconds,
                duration=duration_seconds
            )
        
        return 0, initialize_simulation_data(), [], False, True
    except Exception as e:
        print(f"Error resetting batch animation: {e}")
        return no_update, no_update, no_update, no_update, no_update

# Batch animation step callback (triggered by interval)
@app.callback(
    [Output('batch-current-time', 'data', allow_duplicate=True),
     Output('batch-simulation-data', 'data', allow_duplicate=True),
     Output('batch-applied-events', 'data', allow_duplicate=True),
     Output('status-messages-area', 'children', allow_duplicate=True)],
    Input('batch-animation-interval', 'n_intervals'),
    [State('batch-animation-running', 'data'),
     State('batch-current-time', 'data'),
     State('batch-simulation-data', 'data'),
     State('batch-events', 'data'),
     State('batch-applied-events', 'data'),
     State('batch-timestep-select', 'value'),
     State('network-loaded', 'data'),
     State('pressure-monitoring-nodes', 'data'),
     State('flow-monitoring-links', 'data')],
    prevent_initial_call=True
)
def batch_animation_step(n_intervals, is_running, current_time, sim_data, batch_events, 
                        applied_events, timestep_minutes, network_loaded, monitored_nodes, monitored_links):
    """Process one step of batch animation."""
    print(f"Batch animation step: n_intervals={n_intervals}, is_running={is_running}, current_time={current_time}")
    
    if not is_running:
        return no_update, no_update, no_update, no_update
    
    if not network_loaded or not batch_events:
        error_msg = dbc.Alert("❌ Network or events not loaded", color="danger", dismissable=True)
        return no_update, no_update, no_update, [error_msg]
    
    if global_state['sim'] is None or global_state['wn'] is None:
        error_msg = dbc.Alert("❌ Simulation not initialized", color="danger", dismissable=True)
        return no_update, no_update, no_update, [error_msg]
    
    try:
        from modules.simulation import apply_event_to_simulator, run_simulation_step, collect_simulation_data, initialize_simulation_data
        
        sim = global_state['sim']
        wn = global_state['wn']
        
        # Initialize variables
        current_time = current_time or 0
        sim_data = sim_data or initialize_simulation_data()
        applied_events = applied_events or []
        timestep_seconds = (timestep_minutes or 60) * 60
        status_messages = []
        
        # Apply events that are due at current time
        events_to_apply = [e for e in batch_events 
                          if e.get('time', 0) <= current_time and e not in applied_events]
        
        new_applied_events = list(applied_events)
        
        for event in events_to_apply:
            success, message = apply_event_to_simulator(sim, wn, event)
            if success:
                new_applied_events.append(event)
                status_messages.append(dbc.Alert(f"⚡ Applied: {event['description']}", color="success", dismissable=True))
            else:
                status_messages.append(dbc.Alert(f"❌ Failed: {message}", color="danger", dismissable=True))
        
        # Run simulation step
        success, new_sim_time, message = run_simulation_step(sim, wn)
        
        if success:
            # Update simulation data
            new_sim_data = dict(sim_data)
            new_sim_data['time'].append(new_sim_time)
            
            # Collect data for monitored elements
            monitored_nodes = monitored_nodes or list(wn.node_name_list)[:5]
            monitored_links = monitored_links or list(wn.link_name_list)[:5]
            collect_simulation_data(wn, monitored_nodes, monitored_links, new_sim_data)
            
            return new_sim_time, new_sim_data, new_applied_events, status_messages
        else:
            return current_time, sim_data, new_applied_events, status_messages
            
    except Exception as e:
        error_msg = dbc.Alert(f"❌ Animation step error: {str(e)}", color="danger", dismissable=True)
        return no_update, no_update, no_update, [error_msg]

# Batch animation speed control callback
@app.callback(
    Output('batch-animation-interval', 'interval'),
    Input('batch-speed-select', 'value'),
    prevent_initial_call=True
)
def update_animation_speed(speed):
    """Update animation interval based on speed selection."""
    base_interval = 2000  # 2 seconds base
    speed_value = float(speed) if speed else 1.0
    return int(base_interval / speed_value)

# Batch animation status callback
@app.callback(
    Output('batch-animation-status', 'children'),
    [Input('batch-animation-running', 'data'),
     Input('batch-current-time', 'data')],
    prevent_initial_call=True
)
def update_batch_animation_status(is_running, current_time):
    """Update animation status display."""
    time_display = str(datetime.timedelta(seconds=int(current_time or 0)))
    
    if is_running:
        return dbc.Alert(f"▶️ Playing | Time: {time_display}", color="success")
    else:
        return dbc.Alert(f"⏸️ Paused | Time: {time_display}", color="warning")

# Batch progress display callback
@app.callback(
    Output('batch-progress-display', 'children'),
    [Input('batch-current-time', 'data'),
     Input('batch-events', 'data'),
     Input('batch-applied-events', 'data')],
    prevent_initial_call=True
)
def update_batch_progress(current_time, batch_events, applied_events):
    """Update batch simulation progress display."""
    if not batch_events:
        return ""
    
    # Calculate progress
    max_time = max([event.get('time', 0) for event in batch_events]) if batch_events else 3600
    time_progress = min((current_time or 0) / max_time, 1.0) if max_time > 0 else 0
    events_progress = len(applied_events or []) / len(batch_events) if batch_events else 0
    
    current_time_display = str(datetime.timedelta(seconds=int(current_time or 0)))
    max_time_display = str(datetime.timedelta(seconds=int(max_time)))
    
    return [
        html.Small(f"Time: {current_time_display} / {max_time_display}", className="text-muted"),
        dbc.Progress(
            value=time_progress*100,
            label=f"{time_progress*100:.1f}%",
            color="primary",
            className="mb-2"
        ),
        html.Small(f"Events: {len(applied_events or [])} / {len(batch_events)}", className="text-muted"),
        dbc.Progress(
            value=events_progress*100,
            label=f"{len(applied_events or [])} events",
            color="info"
        )
    ]

# Batch event timeline callback
@app.callback(
    Output('batch-event-timeline', 'children'),
    [Input('batch-events', 'data'),
     Input('batch-current-time', 'data'),
     Input('batch-applied-events', 'data')],
    prevent_initial_call=True
)
def update_batch_event_timeline(batch_events, current_time, applied_events):
    """Create event timeline visualization for batch mode."""
    if not batch_events:
        return "No events loaded"
    
    try:
        from modules.visualization import display_event_timeline
        
        # Create timeline figure
        fig = go.Figure()
        
        # Group events by type for color coding
        event_types = {}
        applied_event_ids = [str(e.get('time', 0)) + e.get('element_name', '') for e in (applied_events or [])]
        
        for i, event in enumerate(batch_events):
            event_type = event['event_type']
            if event_type not in event_types:
                event_types[event_type] = {'times': [], 'names': [], 'applied': []}
            
            event_id = str(event.get('time', 0)) + event.get('element_name', '')
            event_types[event_type]['times'].append(event['time'])
            event_types[event_type]['names'].append(event['element_name'])
            event_types[event_type]['applied'].append(event_id in applied_event_ids)
        
        # Color map for different event types
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (event_type, data) in enumerate(event_types.items()):
            # Applied events (filled markers)
            applied_times = [t for t, applied in zip(data['times'], data['applied']) if applied]
            applied_names = [n for n, applied in zip(data['names'], data['applied']) if applied]
            
            if applied_times:
                fig.add_trace(go.Scatter(
                    x=applied_times,
                    y=[event_type] * len(applied_times),
                    mode='markers',
                    marker=dict(size=10, color=colors[i % len(colors)], symbol='circle'),
                    text=[f"{event_type}<br>Element: {name}<br>Time: {t}s (APPLIED)" 
                          for t, name in zip(applied_times, applied_names)],
                    hovertemplate='%{text}<extra></extra>',
                    name=f"{event_type} (Applied)",
                    showlegend=False
                ))
            
            # Pending events (outlined markers)
            pending_times = [t for t, applied in zip(data['times'], data['applied']) if not applied]
            pending_names = [n for n, applied in zip(data['names'], data['applied']) if not applied]
            
            if pending_times:
                fig.add_trace(go.Scatter(
                    x=pending_times,
                    y=[event_type] * len(pending_times),
                    mode='markers',
                    marker=dict(size=8, color='white', line=dict(color=colors[i % len(colors)], width=2)),
                    text=[f"{event_type}<br>Element: {name}<br>Time: {t}s (PENDING)" 
                          for t, name in zip(pending_times, pending_names)],
                    hovertemplate='%{text}<extra></extra>',
                    name=event_type,
                    showlegend=True
                ))
        
        # Add current time line
        if current_time and current_time > 0:
            fig.add_vline(
                x=current_time,
                line=dict(color="red", width=3, dash="dash"),
                annotation_text=f"Current: {current_time}s"
            )
        
        fig.update_layout(
            title="Event Timeline",
            xaxis_title="Time (seconds)",
            yaxis_title="Event Type",
            height=300,
            hovermode='closest',
            margin=dict(l=10, r=10, t=30, b=10)
        )
        
        return dcc.Graph(figure=fig, style={"height": "300px"})
        
    except Exception as e:
        return dbc.Alert(f"Error creating timeline: {str(e)}", color="danger")

# Batch network map callback (dual view)
@app.callback(
    [Output('batch-network-map-container', 'children'),
     Output('batch-color-legends-container', 'children'),
     Output('batch-state-map-container', 'children')],
    [Input('network-loaded', 'data'),
     Input('batch-map-height-select', 'value'),
     Input('batch-node-size-select', 'value'),
     Input('batch-current-time', 'data'),
     Input('batch-simulation-data', 'data'),
     Input('show-labels-always', 'data')],
    prevent_initial_call=True
)
def update_batch_network_maps(network_loaded, map_height, node_size, current_time, simulation_data, show_labels_always):
    """Update both batch simulation network maps."""
    try:
        if not network_loaded or global_state['wn'] is None:
            return "", "", ""
        
        wn = global_state['wn']
        
        # Create network plot using existing visualization module
        map_height_int = int(map_height) if map_height else 700
        node_size_float = float(node_size) if node_size else 1.0
        
        fig = create_network_plot(
            wn,
            [],  # No manual selections in batch mode
            [],
            show_simulation_data=True,
            sim_initialized=True,
            height=map_height_int,
            node_size_scale=node_size_float,
            show_labels_always=show_labels_always
        )
        
        map_component = dcc.Graph(id="batch-network-graph", figure=fig)
        
        # Create color legends
        pressure_fig = create_pressure_colorbar(0, 100)  # Default range
        flow_fig = create_flow_colorbar(-10, 10)  # Default range
        
        legends = dbc.Row([
            dbc.Col([
                dcc.Graph(figure=pressure_fig, style={"height": "120px"})
            ], width=6),
            dbc.Col([
                dcc.Graph(figure=flow_fig, style={"height": "120px"})
            ], width=6)
        ])
        
        # State visualization plot
        from modules.visualization import create_state_visualization_plot
        state_fig = create_state_visualization_plot(
            wn,
            height=map_height_int,
            node_size_scale=node_size_float,
            show_labels_always=show_labels_always
        )
        
        state_component = dcc.Graph(id="batch-state-graph", figure=state_fig)
        
        return map_component, legends, state_component
    except Exception as e:
        print(f"Error in update_batch_network_maps: {e}")
        error_alert = dbc.Alert(f"Error updating batch maps: {str(e)}", color="danger")
        return error_alert, "", error_alert

# Batch simulation results callback
@app.callback(
    Output('batch-simulation-results-container', 'children'),
    [Input('batch-simulation-data', 'data'),
     Input('batch-current-time', 'data'),
     Input('pressure-monitoring-nodes', 'data'),
     Input('flow-monitoring-links', 'data'),
     Input('network-metadata', 'data')],
    prevent_initial_call=True
)
def update_batch_simulation_results(sim_data, current_time, monitored_nodes, monitored_links, metadata):
    """Update batch simulation monitoring charts with simple multi-select controls."""
    if not monitored_nodes and not monitored_links:
        return dbc.Alert("🚀 Select elements to monitor using the multi-select dropdowns below", color="info")
    
    try:
        # Create simple monitoring charts with multi-select functionality
        available_nodes = metadata.get('node_names', []) if metadata else []
        available_links = metadata.get('link_names', []) if metadata else []
        
        fig_pressure = create_simple_monitoring_chart(available_nodes, monitored_nodes or [], "pressure", sim_data)
        fig_flow = create_simple_monitoring_chart(available_links, monitored_links or [], "flow", sim_data)
        
        # Current time display
        time_display = f"⏱️ Current Time: {datetime.timedelta(seconds=int(current_time or 0))}"
        
        return [
            dbc.Row([
                dbc.Col([
                    html.H4("📊 Batch Simulation Monitoring", className="text-primary"),
                ], width=8),
                dbc.Col([
                    html.H6(time_display, className="text-muted text-end")
                ], width=4)
            ], className="mb-3"),
            
            # SIMPLIFIED Multi-Select Controls for Batch Mode
            dbc.Card([
                dbc.CardHeader("🎛️ Batch Multi-Select Monitoring Controls"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("🔵 Select Nodes for Pressure Monitoring"),
                            dcc.Dropdown(
                                id="batch-pressure-multiselect",
                                options=[{"label": node, "value": node} for node in available_nodes],
                                value=monitored_nodes or [],
                                multi=True,
                                placeholder="Select nodes to monitor...",
                                style={"marginBottom": "10px"}
                            )
                        ], width=6),
                        dbc.Col([
                            html.H5("🔗 Select Links for Flow Monitoring"),
                            dcc.Dropdown(
                                id="batch-flow-multiselect",
                                options=[{"label": link, "value": link} for link in available_links],
                                value=monitored_links or [],
                                multi=True,
                                placeholder="Select links to monitor...",
                                style={"marginBottom": "10px"}
                            )
                        ], width=6)
                    ])
                ])
            ], className="mb-4"),
            
            # Monitoring charts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("📈 Pressure Monitoring Chart", className="mb-0 text-primary")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(
                                id="batch-pressure-chart",
                                figure=fig_pressure, 
                                style={"height": "350px"}
                            ) if fig_pressure else dbc.Alert("📍 Select nodes above to monitor pressure", color="info")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("📈 Flow Monitoring Chart", className="mb-0 text-primary")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(
                                id="batch-flow-chart",
                                figure=fig_flow,
                                style={"height": "350px"}
                            ) if fig_flow else dbc.Alert("🔗 Select links above to monitor flow", color="info")
                        ])
                    ])
                ], width=6)
            ])
        ]
        
    except Exception as e:
        return dbc.Alert(f"Error displaying charts: {str(e)}", color="danger")


# NEW CALLBACK: Handle multi-select dropdowns for batch mode
@app.callback(
    [Output('pressure-monitoring-nodes', 'data', allow_duplicate=True),
     Output('flow-monitoring-links', 'data', allow_duplicate=True)],
    [Input('batch-pressure-multiselect', 'value'),
     Input('batch-flow-multiselect', 'value')],
    prevent_initial_call=True
)
def update_batch_monitoring_from_multiselect(selected_nodes, selected_links):
    """Update monitoring stores based on batch multi-select dropdowns."""
    return selected_nodes or [], selected_links or []


# Batch network status display callback
@app.callback(
    Output('batch-network-status-display', 'children'),
    [Input('network-loaded', 'data'),
     Input('network-metadata', 'data'),
     Input('batch-events', 'data'),
     Input('batch-applied-events', 'data'),
     Input('batch-current-time', 'data')],
    prevent_initial_call=True
)
def display_batch_network_status(network_loaded, metadata, batch_events, applied_events, current_time):
    """Display network status for batch mode."""
    if not network_loaded or not metadata:
        return ""
    
    return dbc.Card([
        dbc.CardHeader(html.H4("📊 Batch Simulation Status")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5(str(metadata['node_count']), className="text-primary"),
                    html.P("Total Nodes", className="mb-0")
                ], width=3),
                dbc.Col([
                    html.H5(str(metadata['link_count']), className="text-primary"),
                    html.P("Total Links", className="mb-0")
                ], width=3),
                dbc.Col([
                    html.H5(str(len(batch_events or [])), className="text-primary"),
                    html.P("Total Events", className="mb-0"),
                    html.Small(f"{len(applied_events or [])} applied", className="text-muted")
                ], width=3),
                dbc.Col([
                    html.H5(str(datetime.timedelta(seconds=int(current_time or 0))), className="text-primary"),
                    html.P("Current Time", className="mb-0")
                ], width=3)
            ])
        ])
    ], className="mb-4")

# Batch simulation initialization callback
@app.callback(
    [Output('batch-simulation-data', 'data', allow_duplicate=True),
     Output('pressure-monitoring-nodes', 'data', allow_duplicate=True),
     Output('flow-monitoring-links', 'data', allow_duplicate=True),
     Output('status-messages-area', 'children', allow_duplicate=True)],
    [Input('network-loaded', 'data'),
     Input('batch-events', 'data')],
    State('network-metadata', 'data'),
    prevent_initial_call=True
)
def initialize_batch_simulation(network_loaded, batch_events, metadata):
    """Initialize batch simulation when network and events are loaded."""
    if not network_loaded or not batch_events or not metadata:
        return no_update, no_update, no_update, no_update
    
    try:
        # Initialize simulation if needed
        if global_state['wn'] is not None:
            from mwntr.sim.interactive_network_simulator import MWNTRInteractiveSimulator
            global_state['sim'] = MWNTRInteractiveSimulator(global_state['wn'])
            
            # Initialize the simulator with proper settings
            duration_seconds = 24 * 3600  # 24 hours default
            timestep_seconds = 60 * 60   # 1 hour timestep
            
            # Get max event time to set proper duration
            if batch_events:
                max_event_time = max([event.get('time', 0) for event in batch_events])
                duration_seconds = max(max_event_time + 3600, duration_seconds)  # Add 1 hour buffer
            
            global_state['sim'].init_simulation(
                global_timestep=timestep_seconds,
                duration=duration_seconds
            )
        
        # Initialize monitoring with first few elements
        initial_nodes = metadata.get('node_names', [])[:5]
        initial_links = metadata.get('link_names', [])[:5]
        
        # Success message
        success_msg = dbc.Alert(
            "🚀 Batch simulation initialized! Click Play to start animation.",
            color="success", dismissable=True
        )
        
        return initialize_simulation_data(), initial_nodes, initial_links, success_msg
        
    except Exception as e:
        error_msg = dbc.Alert(f"❌ Error initializing batch simulation: {str(e)}", color="danger")
        print(f"Error initializing batch simulation: {e}")
        return no_update, no_update, no_update, error_msg

# Add Jump functionality
@app.callback(
    [Output('batch-current-time', 'data', allow_duplicate=True),
     Output('batch-simulation-data', 'data', allow_duplicate=True),
     Output('batch-applied-events', 'data', allow_duplicate=True)],
    Input('batch-jump-btn', 'n_clicks'),
    [State('batch-jump-time-input', 'value'),
     State('batch-events', 'data'),
     State('network-loaded', 'data')],
    prevent_initial_call=True
)
def handle_batch_jump(n_clicks, jump_hours, batch_events, network_loaded):
    """Handle jump to specific time in batch simulation."""
    if not n_clicks or not jump_hours or not batch_events or not network_loaded:
        return no_update, no_update, no_update
    
    try:
        jump_seconds = jump_hours * 3600
        
        # Reset simulation to start
        if global_state['wn'] is not None:
            from modules.simulation import reset_simulation_state, initialize_simulation_data
        global_state['sim'] = reset_simulation_state(global_state['wn'])
        
        # Apply all events up to jump time without running simulation steps
        events_to_apply = [e for e in batch_events if e.get('time', 0) <= jump_seconds]
        applied_events = []
        
        if global_state['sim'] and global_state['wn']:
            from modules.simulation import apply_event_to_simulator
            sim = global_state['sim']
            wn = global_state['wn']
            
            for event in events_to_apply:
                success, message = apply_event_to_simulator(sim, wn, event)
                if success:
                    applied_events.append(event)
        
        return jump_seconds, initialize_simulation_data(), applied_events
        
    except Exception as e:
        print(f"Error jumping to time: {e}")
        return no_update, no_update, no_update

# Toggle event source UI
@app.callback(
    [Output('example-events-div', 'style'),
     Output('upload-events-div', 'style'),
     Output('example-events-buttons-div', 'style')],
    Input('event-file-source-radio', 'value'),
    prevent_initial_call=True
)
def toggle_event_source(src):
    if src == 'example':
        return {"display":"block"}, {"display":"none"}, {"display":"block"}
    return {"display":"none"}, {"display":"block"}, {"display":"none"}

# Manual load for uploaded events
@app.callback(
    [Output('batch-events', 'data', allow_duplicate=True),
     Output('batch-metadata', 'data', allow_duplicate=True),
     Output('status-messages-area', 'children', allow_duplicate=True)],
    Input('load-uploaded-events-btn', 'n_clicks'),
    [State('upload-events-file', 'contents'),
     State('upload-events-file', 'filename'),
     State('event-file-source-radio', 'value')],
    prevent_initial_call=True
)
def load_uploaded_events_btn(n_clicks, contents, filename, src_value):
    if not n_clicks or src_value != 'upload' or contents is None or filename is None:
        return no_update, no_update, no_update
    import json, base64, tempfile, os
    try:
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.json') as tmp:
            tmp.write(decoded)
            tmp_path = tmp.name
        with open(tmp_path, 'r') as f:
            data = json.load(f)
        if 'events' in data and 'metadata' in data:
            events = data['events']
            metadata = data['metadata']
        else:
            events = data if isinstance(data, list) else []
            metadata = {}
        if events:
            msg = dbc.Alert(f"✅ Loaded {len(events)} events from {filename}!", color="success", dismissable=True)
            return events, metadata, msg
        else:
            msg = dbc.Alert("❌ No events found in uploaded file", color="danger", dismissable=True)
            return [], {}, msg
    except Exception as e:
        msg = dbc.Alert(f"❌ Error loading uploaded events: {str(e)}", color="danger", dismissable=True)
        return [], {}, msg
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Show selected filename after upload (does NOT load automatically)
@app.callback(
    Output('upload-events-status', 'children'),
    Input('upload-events-file', 'filename'),
    prevent_initial_call=True
)
def show_upload_selected(filename):
    if filename:
        return dbc.Alert(f"Selected file: {filename}. Click 'Load Uploaded Events' to import.", color="info")
    return ""

# Batch event history callback
@app.callback(
    Output('batch-event-history-container', 'children'),
    [Input('batch-events', 'data'),
     Input('batch-applied-events', 'data'),
     Input('batch-current-time', 'data')],
    prevent_initial_call=True
)
def update_batch_event_history(batch_events, applied_events, current_time):
    """Display event history for batch simulation."""
    if not batch_events:
        return dbc.Alert("No events loaded", color="info")
    try:
        from modules.dash_ui_components import create_professional_event_history
        return create_professional_event_history(batch_events or [], applied_events or [], current_time or 0)
    except Exception as e:
        return dbc.Alert(f"Error displaying event history: {str(e)}", color="danger")


if __name__ == '__main__':
    print("🚀 Starting Water Network Simulator with Simple Monitoring")
    print("✅ Simple dropdowns now work immediately!")
    print("📊 Clean charts update when you select elements")
    print("🗑️ Easy add/remove functionality")
    app.run(debug=True, host="0.0.0.0")
        