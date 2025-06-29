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
from dash import dcc, html, Input, Output, State, callback, ALL, MATCH, ctx, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import time
import datetime
import json
import uuid
import os
import tempfile
from typing import Dict, Any, List, Optional
import pandas as pd

# Import our modularized components
from modules.config import INP_FILE, SIMULATION_DURATION_SECONDS, HYDRAULIC_TIMESTEP_SECONDS
from debug_helpers import debug_session_state, debug_network_state, debug_event_processing
from modules.simulation import (
    load_network_model, apply_event_to_simulator, run_simulation_step,
    load_events_from_json, collect_simulation_data, 
    initialize_simulation_data, reset_simulation_state
)
from modules.visualization import (
    create_network_plot, create_pressure_colorbar, create_flow_colorbar,
    display_event_timeline, create_monitoring_charts
)
from modules.dash_ui_components import create_event_configuration_modal

# Global variables to store actual WNTR objects (can't serialize these)
global_state = {
    'wn': None,
    'sim': None,
    'batch_wn': None,
    'batch_sim': None
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
        dcc.Store(id='batch-sim-initialized', data=False),
        
        # User interface selections
        dcc.Store(id='selected-nodes', data=[]),
        dcc.Store(id='selected-links', data=[]),
        dcc.Store(id='current-element', data=None),
        
        # Event management
        dcc.Store(id='scheduled-events', data=[]),
        dcc.Store(id='applied-events', data=[]),
        dcc.Store(id='current-sim-time', data=0),
        dcc.Store(id='simulation-data', data=initialize_simulation_data()),
        
        # Batch mode state
        dcc.Store(id='loaded-events', data=[]),
        dcc.Store(id='event-metadata', data={}),
        dcc.Store(id='batch-current-sim-time', data=0),
        dcc.Store(id='batch-scheduled-events', data=[]),
        dcc.Store(id='batch-applied-events', data=[]),
        dcc.Store(id='batch-simulation-data', data=initialize_simulation_data()),
        
        # Monitoring selections
        dcc.Store(id='pressure-monitoring-nodes', data=[]),
        dcc.Store(id='flow-monitoring-links', data=[]),
        
        # Simulation planning
        dcc.Store(id='simulation-duration-hours', data=24),
        dcc.Store(id='simulation-timestep-minutes', data=60),
        
        # Network metadata
        dcc.Store(id='network-metadata', data={}),
    ]

def create_layout():
    """Create the main application layout."""
    return dbc.Container([
        # Store components
        html.Div(create_stores()),
        
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
                    dbc.Tab(label="🎮 Interactive Mode", tab_id="interactive"),
                    dbc.Tab(label="📋 Batch Simulator", tab_id="batch"),
                ], id="main-tabs", active_tab="interactive")
            ])
        ], className="mb-4"),
        
        # Main content area
        html.Div(id="main-content"),
        
        # Footer
        html.Hr(),
        html.Footer([
            html.P("Interactive Network Simulator | Built with Dash & Plotly", 
                  className="text-center text-muted small")
        ])
        
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
                    {"label": "NET_4.inp - Sample Distribution Network", "value": "NET_4.inp"}
                ], value="NET_2.inp", id="example-network-select")
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

# Set the app layout
app.layout = create_layout()

# Callback for tab switching
@app.callback(
    Output('main-content', 'children'),
    Input('main-tabs', 'active_tab')
)
def display_tab_content(active_tab):
    """Display content based on selected tab."""
    if active_tab == "interactive":
        return [
            dbc.Row([dbc.Col([create_network_file_selector()])], className="mb-4"),
            html.Div(id="network-status-display"),
            html.Div(id="interactive-main-area")
        ]
    elif active_tab == "batch":
        return [
            dbc.Row([dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4("📋 Batch Event Simulator")),
                    dbc.CardBody([
                        dbc.Alert("Load a JSON file with pre-defined events to run automated simulations", 
                                color="info"),
                        "Batch functionality will be implemented here..."
                    ])
                ])
            ])], className="mb-4")
        ]
    return html.Div("Select a tab")

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

# Callback for loading network
@app.callback(
    [Output('network-loaded', 'data'),
     Output('status-messages-area', 'children'),
     Output('network-metadata', 'data'),
     Output('pressure-monitoring-nodes', 'data'),
     Output('flow-monitoring-links', 'data')],
    Input('load-network-btn', 'n_clicks'),
    [State('file-source-radio', 'value'),
     State('example-network-select', 'value')],
    prevent_initial_call=True
)
def load_network_callback(n_clicks, source, example_file):
    """Load network model based on user selection."""
    if n_clicks is None or n_clicks == 0:
        return False, "", {}, [], []
    
    try:
        # Handle source parameter safely
        source = source or "example"
        file_path = example_file if source == "example" else None
        
        if file_path is None:
            return False, dbc.Alert("❌ Please select a file", color="danger"), {}, [], []
        
        # Load the network model
        wn = load_network_model(file_path)
        
        if wn:
            # Store network in global state
            global_state['wn'] = wn
            
            # Create simulator instance
            from mwntr.sim.interactive_network_simulator import MWNTRInteractiveSimulator
            global_state['sim'] = MWNTRInteractiveSimulator(wn)
            
            # Create metadata for storage
            metadata = {
                'file_path': file_path,
                'node_count': len(wn.node_name_list),
                'link_count': len(wn.link_name_list),
                'node_names': list(wn.node_name_list),
                'link_names': list(wn.link_name_list)
            }
            
            # Initialize monitoring with first few elements
            initial_nodes = list(wn.node_name_list)[:5]
            initial_links = list(wn.link_name_list)[:5]
            
            filename = os.path.basename(file_path)
            success_msg = dbc.Alert(f"✅ Network '{filename}' loaded successfully!", 
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
    Output('interactive-main-area', 'children'),
    [Input('network-loaded', 'data'),
     Input('network-metadata', 'data')],
    prevent_initial_call=True
)
def display_interactive_main(network_loaded, metadata):
    """Display the main interactive area when network is loaded."""
    if not network_loaded:
        return dbc.Alert("👆 Please load a network file to begin simulation", color="info")
    
    return [
        dbc.Row([
            # Map column
            dbc.Col([
                # Map controls
                dbc.Card([
                    dbc.CardBody([
                        dbc.Alert("🖱️ Click on nodes (circles) or links (squares) to select and configure events", 
                                color="info", className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Map Height"),
                                dbc.Select([
                                    {"label": "700px", "value": 700},
                                    {"label": "800px", "value": 800},
                                    {"label": "900px", "value": 900},
                                    {"label": "1000px", "value": 1000}
                                ], value=800, id="map-height-select")
                            ], width=4),
                            dbc.Col([
                                dbc.Label("Node Size"),
                                dbc.Select([
                                    {"label": "60%", "value": 0.6},
                                    {"label": "80%", "value": 0.8},
                                    {"label": "100%", "value": 1.0},
                                    {"label": "120%", "value": 1.2}
                                ], value=1.0, id="node-size-select")
                            ], width=4),
                            dbc.Col([
                                dbc.Button("🧭 Legend", id="legend-btn", color="info", size="sm")
                            ], width=4)
                        ])
                    ])
                ], className="mb-3"),
                
                # Network map
                html.Div(id="network-map-container"),
                
                # Color legends
                html.Div(id="color-legends-container")
            ], width=8),
            
            # Control panel column (simplified)
            dbc.Col([
                # Simulation controls
                dbc.Card([
                    dbc.CardHeader(html.H5("⚡ Simulation Controls")),
                    dbc.CardBody([
                        html.Div(id="sim-status"),
                        
                        # Simulation progress
                        html.Div(id="simulation-progress-display", className="mb-3"),
                        
                        # Planning controls
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Duration (hours)"),
                                dbc.Input(id="duration-input", type="number", 
                                        value=24, min=1, max=168, step=1)
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
                            dbc.Button("🔄 Reset", id="reset-btn", color="secondary")
                        ], className="w-100")
                    ])
                ], className="mb-3")
            ], width=4)
        ]),
        
        # Results section
        html.Hr(),
        html.Div(id="simulation-results-container"),
        
        # Event configuration modal
        create_event_configuration_modal()
    ]

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

# Network map callback
@app.callback(
    [Output('network-map-container', 'children'),
     Output('color-legends-container', 'children')],
    [Input('network-loaded', 'data'),
     Input('map-height-select', 'value'),
     Input('node-size-select', 'value'),
     Input('selected-nodes', 'data'),
     Input('selected-links', 'data'),
     Input('sim-initialized', 'data'),
     Input('simulation-data', 'data'),
     Input('current-sim-time', 'data')],
    prevent_initial_call=True
)
def update_network_map(network_loaded, map_height, node_size, 
                      selected_nodes, selected_links, sim_initialized, simulation_data, current_time):
    """Update the network map visualization."""
    try:
        if not network_loaded or global_state['wn'] is None:
            return "", ""
        
        wn = global_state['wn']
        show_sim_data = sim_initialized or False
        
        # Create network plot using existing visualization module
        fig = create_network_plot(
            wn,
            selected_nodes or [],
            selected_links or [],
            show_simulation_data=show_sim_data,
            sim_initialized=sim_initialized or False,
            height=map_height or 800,
            node_size_scale=node_size or 1.0
        )
        
        map_component = dcc.Graph(id="network-graph", figure=fig)
        
        # Create color legends if showing simulation data
        legends = ""
        if show_sim_data:
            # Get current data for legends (simplified for now)
            pressure_fig = create_pressure_colorbar(0, 100)  # Default range
            flow_fig = create_flow_colorbar(-10, 10)  # Default range
            
            legends = dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=pressure_fig, style={"height": "150px"})
                ], width=6),
                dbc.Col([
                    dcc.Graph(figure=flow_fig, style={"height": "150px"})
                ], width=6)
            ])
        
        return map_component, legends
    except Exception as e:
        print(f"Error in update_network_map: {e}")
        return dbc.Alert(f"Error updating map: {str(e)}", color="danger"), ""

# Element selection callback (from map clicks) - now opens modal
@app.callback(
    [Output('current-element', 'data'),
     Output('selected-nodes', 'data'),
     Output('selected-links', 'data'),
     Output('event-config-modal', 'is_open')],
    Input('network-graph', 'clickData'),
    [State('network-loaded', 'data'),
     State('network-metadata', 'data')],
    prevent_initial_call=True
)
def handle_map_click(click_data, network_loaded, metadata):
    """Handle clicks on the network map - opens modal for event configuration."""
    if not click_data or not network_loaded:
        return None, [], [], False
    
    # Extract element info from click data
    point = click_data['points'][0]
    if 'customdata' in point and point['customdata']:
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
    [Input('close-event-modal', 'n_clicks'),
     Input('cancel-event-btn', 'n_clicks')],
    State('event-config-modal', 'is_open'),
    prevent_initial_call=True
)
def close_modal(close_clicks, cancel_clicks, is_open):
    """Close the modal when close or cancel buttons are clicked."""
    if close_clicks or cancel_clicks:
        return False
    return is_open

# Simulation control callbacks - Initialize
@app.callback(
    [Output('sim-initialized', 'data'),
     Output('sim-status', 'children')],
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
        return current_status or False, dbc.Alert("🟡 No network loaded", color="warning")
    
    try:
        sim = global_state['sim']
        
        # Initialize simulation
        duration_seconds = (duration_hours or 24) * 3600
        timestep_seconds = (timestep_minutes or 60) * 60
        
        sim.init_simulation(
            global_timestep=timestep_seconds,
            duration=duration_seconds
        )
        
        status = dbc.Alert("🟢 Simulation Initialized Successfully!", color="success")
        return True, status
            
    except Exception as e:
        error_status = dbc.Alert(f"❌ Initialization Error: {str(e)}", color="danger")
        return False, error_status

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
    Input('step-btn', 'n_clicks'),
    [State('sim-initialized', 'data'),
     State('current-sim-time', 'data'),
     State('simulation-data', 'data'),
     State('scheduled-events', 'data'),
     State('applied-events', 'data'),
     State('pressure-nodes-dropdown', 'value'),
     State('flow-links-dropdown', 'value')],
    prevent_initial_call=True
)
def handle_simulation_step(step_clicks, sim_initialized, current_time, sim_data, 
                          scheduled_events, applied_events, monitored_nodes, monitored_links):
    """Handle simulation step button."""
    if not step_clicks or not sim_initialized or global_state['sim'] is None:
        return no_update, no_update, no_update, no_update, no_update
    
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
        
        # Apply scheduled events that are due
        events_to_apply = [e for e in scheduled_events 
                          if e.get('scheduled_time', e.get('time', 0)) <= current_time]
        
        new_applied_events = list(applied_events)
        new_scheduled_events = list(scheduled_events)
        
        for event in events_to_apply:
            success, message = apply_event_to_simulator(sim, wn, event)
            if success:
                new_applied_events.append(event)
                new_scheduled_events.remove(event)
                status_messages.append(dbc.Alert(f"⚡ Applied: {message}", color="success", dismissable=True))
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
            
            status_messages.append(dbc.Alert("✅ Step completed successfully!", color="success", dismissable=True))
            
            return new_sim_time, new_sim_data, new_applied_events, new_scheduled_events, status_messages
        else:
            status_messages.append(dbc.Alert(f"⚠️ Step warning: {message}", color="warning", dismissable=True))
            return current_time, sim_data, new_applied_events, new_scheduled_events, status_messages
            
    except Exception as e:
        error_msg = dbc.Alert(f"❌ Step error: {str(e)}", color="danger", dismissable=True)
        return no_update, no_update, no_update, no_update, [error_msg]

# Update step button state
@app.callback(
    Output('step-btn', 'disabled'),
    Input('sim-initialized', 'data')
)
def update_step_button(sim_initialized):
    """Enable/disable step button based on simulation state."""
    return not (sim_initialized or False)

# Simulation progress display callback
@app.callback(
    Output('simulation-progress-display', 'children'),
    [Input('sim-initialized', 'data'),
     Input('current-sim-time', 'data'),
     Input('simulation-data', 'data')],
    [State('duration-input', 'value'),
     State('timestep-select', 'value')],
    prevent_initial_call=True
)
def update_simulation_progress(sim_initialized, current_time, sim_data, duration_hours, timestep_minutes):
    """Display professional simulation progress bar."""
    if not sim_initialized:
        return dbc.Alert("🎯 Initialize simulation to see progress", color="info", className="text-center")
    
    try:
        import datetime
        
        # Calculate simulation parameters
        total_duration_seconds = (duration_hours or 24) * 3600
        timestep_seconds = (timestep_minutes or 60) * 60
        current_time = current_time or 0
        
        # Calculate progress
        time_progress = min(current_time / total_duration_seconds, 1.0)
        completed_steps = len((sim_data or {}).get('time', []))
        planned_total_steps = int(total_duration_seconds / timestep_seconds)
        step_progress = min(completed_steps / planned_total_steps, 1.0) if planned_total_steps > 0 else 0
        
        # Time displays
        current_time_display = str(datetime.timedelta(seconds=int(current_time)))
        total_time_display = str(datetime.timedelta(seconds=int(total_duration_seconds)))
        remaining_time = max(0, total_duration_seconds - current_time)
        remaining_time_display = str(datetime.timedelta(seconds=int(remaining_time)))
        
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



# Real-time monitoring charts display callback
@app.callback(
    Output('simulation-results-container', 'children'),
    [Input('simulation-data', 'data'),
     Input('current-sim-time', 'data'),
     Input('sim-initialized', 'data'),
     Input('pressure-nodes-dropdown', 'value'),
     Input('flow-links-dropdown', 'value')],
    prevent_initial_call=True
)
def update_simulation_results(sim_data, current_time, sim_initialized, monitored_nodes, monitored_links):
    """Update real-time monitoring charts display."""
    if not sim_initialized:
        return dbc.Alert("🚀 Initialize simulation to see real-time monitoring charts", color="info")
    
    # Use default selections if none provided
    if not monitored_nodes and global_state['wn'] is not None:
        monitored_nodes = list(global_state['wn'].node_name_list)[:5]
    if not monitored_links and global_state['wn'] is not None:
        monitored_links = list(global_state['wn'].link_name_list)[:5]
    
    try:
        # Create monitoring charts (always show charts, even if no data yet)
        fig_pressure, fig_flow = create_monitoring_charts(sim_data or initialize_simulation_data(), monitored_nodes or [], monitored_links or [])
        
        # Current time display
        time_display = f"⏱️ Current Time: {datetime.timedelta(seconds=int(current_time or 0))}" if current_time else "⏱️ Time: 00:00:00"
        
        results_content = [
            dbc.Row([
                dbc.Col([
                    html.H4("📊 Real-Time Monitoring", className="text-primary"),
                ], width=8),
                dbc.Col([
                    html.H6(time_display, className="text-muted text-end")
                ], width=4)
            ], className="mb-3"),
            
            # Monitoring charts in full width rows for better visibility
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=fig_pressure,
                        style={"height": "400px"},
                        config={'displayModeBar': True, 'displaylogo': False}
                    ) if fig_pressure else 
                    dbc.Alert("📍 Select nodes to monitor pressure", color="info")
                ], width=12)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=fig_flow,
                        style={"height": "400px"},
                        config={'displayModeBar': True, 'displaylogo': False}
                    ) if fig_flow else 
                    dbc.Alert("🔗 Select links to monitor flow", color="info")
                ], width=12)
            ], className="mb-3"),
            
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
     Output('event-config-modal', 'is_open', allow_duplicate=True)],
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
        return scheduled_events or [], "", dash.no_update
    
    try:
        # Find which button was clicked
        triggered_idx = None
        for i, clicks in enumerate(btn_clicks):
            if clicks:
                triggered_idx = i
                break
        
        if triggered_idx is None or not event_types[triggered_idx]:
            return scheduled_events or [], ""
        
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
                    param_str = ", ".join([f"{k}={v}" for k, v in parameters.items()]) if parameters else "no parameters"
                    success_msg = dbc.Alert(
                        f"⚡ Applied {event['event_type']} to {event['element_name']} immediately! ({param_str})",
                        color="success",
                        dismissable=True
                    )
                    return scheduled_events or [], success_msg, False  # Close modal
                else:
                    error_msg = dbc.Alert(f"❌ Failed to apply event: {message}", color="danger")
                    return scheduled_events or [], error_msg, dash.no_update
            else:
                # Add to scheduled events for later application
                new_scheduled_events = (scheduled_events or []) + [event]
                param_str = ", ".join([f"{k}={v}" for k, v in parameters.items()]) if parameters else "no parameters"
                success_msg = dbc.Alert(
                    f"✅ Event scheduled: {event['event_type']} on {event['element_name']} ({param_str})",
                    color="success",
                    dismissable=True
                )
                return new_scheduled_events, success_msg, False  # Close modal
        
        return scheduled_events or [], "", dash.no_update
        
    except Exception as e:
        error_msg = dbc.Alert(f"❌ Error scheduling event: {str(e)}", color="danger")
        return scheduled_events or [], error_msg, dash.no_update

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
     Input('pressure-nodes-dropdown', 'value'),
     Input('flow-links-dropdown', 'value')],
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
                    pressure = current_pressures[node_name]
                    elements.append(
                        html.P(f"🔵 {node_name}: {pressure:.2f} m", className="mb-1 small")
                    )
        
        # Show current flows
        if monitored_links and current_flows:
            for link_name in (monitored_links or [])[:3]:  # Show max 3 for space
                if link_name in current_flows:
                    flow = current_flows[link_name]
                    elements.append(
                        html.P(f"🔗 {link_name}: {flow:.4f} m³/s", className="mb-1 small")
                    )
        
        if not elements:
            return html.P("Select elements to monitor", className="text-muted small")
        
        return elements
        
    except Exception as e:
        return html.P(f"Error: {str(e)}", className="text-danger small")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)