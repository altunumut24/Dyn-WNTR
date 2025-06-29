"""
Dash UI Components for Interactive Network Simulator.

This module contains all the reusable Dash interface components that make up
the user interface. It's organized as a collection of functions that create
specific UI elements like buttons, forms, displays, and controls.

Key responsibilities:
- Creating consistent UI components across the application
- Handling user input and form validation
- Displaying network information and simulation status
- Managing event configuration interfaces
- Providing monitoring and control panels

UI Component categories:
1. Status displays (network info, simulation status)
2. Control panels (simulation controls, event configuration)
3. Data displays (current values, event history)
4. Input forms (event parameters, monitoring selection)
5. Utility components (legends, help text, file upload)

Design principles:
- Each component is self-contained and reusable
- Components return Dash components rather than modifying global state
- Consistent styling using Bootstrap components
- Clear labeling and help text for user guidance
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import datetime
import pandas as pd
from typing import List, Dict, Any, Optional
import plotly.graph_objects as go

# Import WNTR components
from mwntr.network import WaterNetworkModel
from mwntr.sim.interactive_network_simulator import MWNTRInteractiveSimulator

# Import our configuration and simulation functions
from .config import NODE_EVENTS, LINK_EVENTS, DEFAULT_MONITORED_NODES_COUNT, DEFAULT_MONITORED_LINKS_COUNT
from .simulation import get_available_events, create_event


def create_simulation_status_display(sim_initialized: bool, current_sim_time: float) -> dbc.Alert:
    """
    Create simulation status display component.
    
    Args:
        sim_initialized (bool): Whether simulation is initialized
        current_sim_time (float): Current simulation time in seconds
        
    Returns:
        dbc.Alert: Bootstrap alert component showing simulation status
    """
    if sim_initialized:
        current_time = datetime.timedelta(seconds=int(current_sim_time))
        return dbc.Alert(f"🟢 Active | Time: {current_time}", color="success")
    else:
        return dbc.Alert("🟡 Not Initialized", color="warning")


def create_element_properties_display(wn: WaterNetworkModel, element_name: str, element_type: str) -> List:
    """
    Create element properties display component.
    
    Args:
        wn (WaterNetworkModel): The water network model
        element_name (str): Name/ID of the selected element
        element_type (str): Type of element (Junction, Tank, Reservoir, Pipe, etc.)
        
    Returns:
        List: List of Dash components displaying element properties
    """
    try:
        properties = []
        
        if element_type in ['Junction', 'Tank', 'Reservoir']:
            # Handle node elements
            element = wn.get_node(element_name)
            coords = element.coordinates
            properties.append(html.P(f"Coordinates: ({coords[0]:.2f}, {coords[1]:.2f})"))
            
            # Show node-specific properties
            if hasattr(element, 'base_demand'):
                properties.append(html.P(f"Base Demand: {element.base_demand:.4f} m³/s"))
            if hasattr(element, 'elevation'):
                properties.append(html.P(f"Elevation: {element.elevation:.2f} m"))
            if hasattr(element, 'pressure') and element.pressure is not None:
                properties.append(html.P(f"Current Pressure: {element.pressure:.2f} m"))
                
        else:  # Handle link elements
            element = wn.get_link(element_name)
            properties.append(html.P(f"Start Node: {element.start_node_name}"))
            properties.append(html.P(f"End Node: {element.end_node_name}"))
            
            # Show link-specific properties
            if hasattr(element, 'length'):
                properties.append(html.P(f"Length: {element.length:.2f} m"))
            if hasattr(element, 'diameter'):
                properties.append(html.P(f"Diameter: {element.diameter:.3f} m"))
            if hasattr(element, 'flow') and element.flow is not None:
                properties.append(html.P(f"Current Flow: {element.flow:.4f} m³/s"))
        
        return properties
        
    except Exception as e:
        return [dbc.Alert(f"Error getting element properties: {e}", color="danger")]


def create_event_configuration_form(element_name: str, element_type: str, element_category: str) -> List:
    """
    Create event configuration form for the selected element.
    
    Args:
        element_name (str): Name of the selected element
        element_type (str): Type of element
        element_category (str): Category ('node' or 'link')
        
    Returns:
        List: List of Dash components for event configuration
    """
    available_events = get_available_events(element_type, element_category)
    
    if not available_events:
        return [dbc.Alert(f"No events available for {element_type}", color="info")]
    
    # Event type selection
    event_options = [{"label": event, "value": event} for event in available_events.keys()]
    
    form_components = [
        dbc.Label("Event Type:", className="fw-bold"),
        dbc.Select(
            options=event_options,
            id={"type": "event-type-select", "element": element_name},
            placeholder="Select event type..."
        ),
        html.Br(),
        
        dbc.Label("Schedule Time (seconds):", className="fw-bold"),
        dbc.Input(
            type="number",
            value=0,
            min=0,
            step=60,
            id={"type": "event-time-input", "element": element_name}
        ),
        html.Br(),
        
        # Dynamic parameter inputs will be added via callback
        html.Div(id={"type": "event-params-container", "element": element_name}),
        
        dbc.Button(
            "⚡ Schedule Event",
            id={"type": "schedule-event-btn", "element": element_name},
            color="primary",
            className="mt-2"
        )
    ]
    
    return form_components


def create_current_values_display(wn: WaterNetworkModel, monitored_nodes: List[str], 
                                monitored_links: List[str], title_prefix: str = "") -> List:
    """
    Create current values display for monitored elements.
    
    Args:
        wn (WaterNetworkModel): The water network model
        monitored_nodes (List[str]): List of node names to monitor
        monitored_links (List[str]): List of link names to monitor
        title_prefix (str): Prefix for the title
        
    Returns:
        List: List of Dash components displaying current values
    """
    if not monitored_nodes and not monitored_links:
        return []
    
    components = [html.H6(f"📈 {title_prefix}Current Values for Monitored Elements")]
    
    if monitored_nodes:
        node_components = [html.H6("Current Node Pressures:", className="fw-bold")]
        for node_name in monitored_nodes[:6]:  # Show max 6 for space
            if node_name in wn.node_name_list:
                node = wn.get_node(node_name)
                pressure = getattr(node, 'pressure', 0) if hasattr(node, 'pressure') else 0
                node_components.append(
                    dbc.Row([
                        dbc.Col(html.P(node_name), width=6),
                        dbc.Col(html.P(f"{pressure:.2f} m"), width=6)
                    ])
                )
        components.extend(node_components)
    
    if monitored_links:
        link_components = [html.H6("Current Link Flow Rates:", className="fw-bold")]
        for link_name in monitored_links[:6]:  # Show max 6 for space
            if link_name in wn.link_name_list:
                link = wn.get_link(link_name)
                flow = getattr(link, 'flow', 0) if hasattr(link, 'flow') else 0
                link_components.append(
                    dbc.Row([
                        dbc.Col(html.P(link_name), width=6),
                        dbc.Col(html.P(f"{flow:.4f} m³/s"), width=6)
                    ])
                )
        components.extend(link_components)
    
    return components


def create_events_summary_display(scheduled_events: List[Dict], applied_events: List[Dict]) -> List:
    """
    Create events summary display component.
    
    Args:
        scheduled_events (List[Dict]): List of scheduled events
        applied_events (List[Dict]): List of applied events
        
    Returns:
        List: List of Dash components for events summary
    """
    components = []
    
    # Events metrics
    components.append(
        dbc.Row([
            dbc.Col([
                html.H5(str(len(scheduled_events)), className="text-primary"),
                html.P("Scheduled", className="mb-0")
            ], width=6),
            dbc.Col([
                html.H5(str(len(applied_events)), className="text-primary"),
                html.P("Applied", className="mb-0")
            ], width=6)
        ])
    )
    
    # Upcoming events
    if scheduled_events:
        upcoming_items = [html.H6("⏰ Upcoming Events", className="fw-bold")]
        for idx, event in enumerate(scheduled_events[-5:]):  # Show last 5
            upcoming_items.append(
                dbc.Row([
                    dbc.Col(
                        html.P(f"- {event['event_type']} → {event['element_name']}"),
                        width=10
                    ),
                    dbc.Col(
                        dbc.Button("🗑️", id=f"remove-event-{idx}", size="sm", color="danger"),
                        width=2
                    )
                ])
            )
        components.extend(upcoming_items)
    
    # Recent applied events
    if applied_events:
        recent_items = [html.H6("✅ Recent Events", className="fw-bold")]
        for event in applied_events[-3:]:  # Show last 3
            recent_items.append(
                html.P(f"- {event['event_type']} → {event['element_name']}")
            )
        components.extend(recent_items)
    
    return components


def create_applied_events_history_display(applied_events: List[Dict], title_prefix: str = "") -> List:
    """
    Create applied events history display component.
    
    Args:
        applied_events (List[Dict]): List of applied events
        title_prefix (str): Prefix for the title
        
    Returns:
        List: List of Dash components for events history
    """
    if not applied_events:
        return [dbc.Alert("📝 No events applied yet. Events will appear here once the simulation processes them.", color="info")]
    
    components = [html.H5(f"📋 {title_prefix}Applied Events History")]
    
    # Recent events display
    recent_events = applied_events[-5:]  # Show last 5
    for event in reversed(recent_events):
        event_time = datetime.timedelta(seconds=int(event['time']))
        components.append(
            dbc.Alert(f"✅ {event_time} - {event['description']}", color="success")
        )
    
    # Full history in collapsible section
    if len(applied_events) > 5:
        # Create table data
        table_data = []
        for event in applied_events:
            table_data.append({
                'Time (s)': event['time'],
                'Time': str(datetime.timedelta(seconds=int(event['time']))),
                'Element': event['element_name'],
                'Event Type': event['event_type'],
                'Description': event['description']
            })
        
        df = pd.DataFrame(table_data)
        
        # Create table component
        table = dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, size="sm")
        
        components.append(
            dbc.Collapse([
                dbc.Card([
                    dbc.CardHeader("📜 Complete Event History"),
                    dbc.CardBody([table])
                ])
            ], id="event-history-collapse")
        )
        
        components.append(
            dbc.Button("📜 Show Complete History", id="toggle-history-btn", size="sm", className="mt-2")
        )
    
    return components


def create_legend_display() -> dbc.Modal:
    """
    Create network legend modal component.
    
    Returns:
        dbc.Modal: Bootstrap modal containing network legend
    """
    return dbc.Modal([
        dbc.ModalHeader("🎨 Network Legend"),
        dbc.ModalBody([
            html.P("Nodes:", className="fw-bold"),
            html.Ul([
                html.Li("🔵 Junctions"),
                html.Li("🔷 Tanks"),
                html.Li("🟢 Reservoirs")
            ]),
            html.P("Links:", className="fw-bold"),
            html.Ul([
                html.Li("▬ Pipes"),
                html.Li("▬ Pumps"),
                html.Li("▬ Valves")
            ]),
            html.P("Selection:", className="fw-bold"),
            html.Ul([
                html.Li("🔴 Red highlights selected elements")
            ])
        ]),
        dbc.ModalFooter([
            dbc.Button("Close", id="close-legend-modal", className="ms-auto")
        ])
    ], id="legend-modal", is_open=False)


def create_simulation_progress_display(simulation_data: Dict, planning_data: Dict) -> List:
    """
    Create simulation progress display component.
    
    Args:
        simulation_data (Dict): Current simulation data
        planning_data (Dict): Simulation planning parameters
        
    Returns:
        List: List of Dash components for progress display
    """
    if not simulation_data['time']:
        return [dbc.Alert("No simulation data yet - start simulation to see progress", color="info")]
    
    # Get planning data
    total_duration_seconds = planning_data.get('duration_hours', 24) * 3600
    timestep_seconds = planning_data.get('timestep_minutes', 60) * 60
    planned_total_steps = int(total_duration_seconds / timestep_seconds)
    
    # Current progress
    current_time = simulation_data['time'][-1]
    completed_steps = len(simulation_data['time'])
    
    # Calculate progress
    time_progress = min(current_time / total_duration_seconds, 1.0)
    
    components = []
    
    # Progress metrics
    components.append(
        dbc.Row([
            dbc.Col([
                html.H6(f"{completed_steps}/{planned_total_steps}", className="text-primary"),
                html.P("Completed Steps", className="mb-0")
            ], width=3),
            dbc.Col([
                html.H6(str(datetime.timedelta(seconds=int(current_time))), className="text-primary"),
                html.P("Simulation Time", className="mb-0")
            ], width=3),
            dbc.Col([
                html.H6(str(datetime.timedelta(seconds=int(max(0, total_duration_seconds - current_time)))), className="text-primary"),
                html.P("Time Remaining", className="mb-0")
            ], width=3),
            dbc.Col([
                html.H6(f"{time_progress*100:.1f}%", className="text-primary"),
                html.P("Progress", className="mb-0")
            ], width=3)
        ])
    )
    
    # Progress bar
    components.append(
        dbc.Progress(value=time_progress*100, label=f"{time_progress*100:.1f}%", className="mb-3")
    )
    
    return components


def create_batch_upload_interface() -> List:
    """
    Create batch event upload interface component.
    
    Returns:
        List: List of Dash components for batch upload
    """
    return [
        dbc.Label("Upload Event Batch:", className="fw-bold"),
        dcc.Upload([
            html.Div([
                "Drag and Drop or ",
                html.A("Select JSON File", className="text-primary")
            ], className="text-center p-4 border border-dashed rounded")
        ], id="upload-batch-file", multiple=False, accept=".json"),
        
        html.Div(id="batch-upload-status"),
        
        dbc.Collapse([
            dbc.Card([
                dbc.CardHeader("📋 Example Event Format"),
                dbc.CardBody([
                    html.Pre('''
{
    "metadata": {
        "generated_at": "2024-01-01 12:00:00",
        "total_events": 10,
        "duration_seconds": 3600
    },
    "events": [
        {
            "time": 300,
            "element_name": "J1",
            "element_type": "Junction",
            "element_category": "node",
            "event_type": "start_leak",
            "parameters": {
                "leak_area": 0.01,
                "leak_discharge_coefficient": 0.75
            },
            "description": "Leak started on J1"
        }
    ]
}
                    ''', style={"fontSize": "12px", "backgroundColor": "#f8f9fa"})
                ])
            ])
        ], id="example-format-collapse", is_open=False),
        
        dbc.Button("📋 Show Example Format", id="show-example-btn", 
                  color="info", size="sm", className="mt-2")
    ]


def validate_inp_file_content(content: str) -> Dict[str, bool]:
    """
    Validate basic structure of INP file content.
    
    Args:
        content (str): Content of the INP file
        
    Returns:
        Dict[str, bool]: Validation results for different sections
    """
    content_upper = content.upper()
    
    return {
        "Contains [JUNCTIONS] section": "[JUNCTIONS]" in content_upper,
        "Contains [PIPES] section": "[PIPES]" in content_upper,
        "Contains [RESERVOIRS] or [TANKS] section": 
            "[RESERVOIRS]" in content_upper or "[TANKS]" in content_upper,
        "Contains [COORDINATES] section": "[COORDINATES]" in content_upper,
        "File size is reasonable (< 10MB)": len(content.encode('utf-8')) < 10 * 1024 * 1024,
        "Contains [END] marker": "[END]" in content_upper
    }


def create_network_info_display(file_path: str, node_count: int, link_count: int) -> dbc.Card:
    """
    Create network information display component.
    
    Args:
        file_path (str): Path to the network file
        node_count (int): Number of nodes in the network
        link_count (int): Number of links in the network
        
    Returns:
        dbc.Card: Bootstrap card containing network information
    """
    import os
    filename = os.path.basename(file_path)
    
    return dbc.Card([
        dbc.CardHeader("ℹ️ Network Information"),
        dbc.CardBody([
            html.P(f"File: {filename}"),
            html.P(f"Nodes: {node_count}"),
            html.P(f"Links: {link_count}")
        ])
    ]) 