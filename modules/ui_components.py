"""
UI Components for Interactive Network Simulator.

This module contains all the reusable Streamlit interface components that make up
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
- Components return data rather than modifying global state
- Consistent styling using Streamlit's built-in components
- Clear labeling and help text for user guidance
"""

import streamlit as st
import datetime
import pandas as pd
from typing import List, Dict, Any, Optional

# Import WNTR components
from mwntr.network import WaterNetworkModel
from mwntr.sim.interactive_network_simulator import MWNTRInteractiveSimulator

# Import our configuration and simulation functions
from .config import NODE_EVENTS, LINK_EVENTS, DEFAULT_MONITORED_NODES_COUNT, DEFAULT_MONITORED_LINKS_COUNT
from .simulation import get_available_events, create_event


def display_network_status(wn: WaterNetworkModel, current_element: Optional[Dict], 
                          scheduled_events: List[Dict], applied_events: List[Dict]):
    """
    Display the network status dashboard with key metrics.
    
    This component shows an overview of the network and current simulation state
    in a clean, metrics-based layout. It provides quick insights into:
    - Network size (nodes and links)
    - Current selection state
    - Event status (scheduled vs applied)
    
    Args:
        wn (WaterNetworkModel): The loaded water network model
        current_element (Optional[Dict]): Currently selected element info
        scheduled_events (List[Dict]): Events waiting to be applied
        applied_events (List[Dict]): Events that have been applied
        
    Interface design:
    - Creates the top status bar with key metrics
    - Provides quick overview without cluttering the interface
    - Delta values show additional context (like element type)
    - Helps users understand current state at a glance
    """
    st.markdown('<h3 class="section-header">📊 Network Status</h3>', unsafe_allow_html=True)
    
    # Create four columns for different metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Show total number of nodes in the network
        st.metric("Total Nodes", len(wn.node_name_list), help="Junction, Tank, and Reservoir nodes")
        
    with col2:
        # Show total number of links in the network
        st.metric("Total Links", len(wn.link_name_list), help="Pipes, Pumps, and Valves")
        
    with col3:
        # Show currently selected element (if any)
        if current_element:
            st.metric("Selected Element", current_element['name'], 
                     delta=current_element['type'], delta_color="normal")
        else:
            st.metric("Selected Element", "None", help="Click on network elements to select")
            
    with col4:
        # Show event status (scheduled events with applied events as delta)
        st.metric("Scheduled Events", len(scheduled_events), 
                 delta=len(applied_events), delta_color="normal")

    # Add visual separator
    st.markdown("---")


def display_simulation_controls(sim: MWNTRInteractiveSimulator, current_sim_time: float, key_prefix: str = "") -> str:
    """
    Display simulation control buttons and status indicator.
    
    This component provides the main simulation controls that allow users to:
    - Initialize the simulation with current settings
    - Step through simulation one time step at a time
    - Reset simulation back to initial state
    
    Args:
        sim (MWNTRInteractiveSimulator): The WNTR simulator instance
        current_sim_time (float): Current simulation time in seconds
        key_prefix (str): Prefix for button keys (for multiple instances)
        
    Returns:
        str: Action taken by user ("initialize", "step", "reset", or None)
        
    Control logic:
    - Creates the main simulation control panel
    - Button states change based on simulation status
    - Returns the action so main app can handle the simulation logic
    - Status indicator shows current simulation state
    """
    st.markdown('<h4 class="section-header">⚡ Simulation</h4>', unsafe_allow_html=True)
    
    # Display current simulation status
    if sim.initialized_simulation:
        # Simulation is active - show current time
        current_time = datetime.timedelta(seconds=int(current_sim_time))
        st.success(f"🟢 **Active** | Time: {current_time}")
    else:
        # Simulation not started yet
        st.warning("🟡 **Not Initialized**")
    
    # Create control button layout
    sim_col1, sim_col2 = st.columns(2)
    
    action = None  # Track which action user took
    
    with sim_col1:
        # Initialize button - disabled if already initialized
        if st.button("🚀 Initialize", disabled=sim.initialized_simulation, use_container_width=True, key=f"{key_prefix}initialize"):
            action = "initialize"
    
    with sim_col2:
        # Step button - disabled if not initialized or simulation ended
        step_disabled = not sim.initialized_simulation or sim.is_terminated()
        if st.button("⏭️ Step", disabled=step_disabled, use_container_width=True, key=f"{key_prefix}step"):
            action = "step"
    
    # Reset button - always available to start over
    if st.button("🔄 Reset", use_container_width=True, key=f"{key_prefix}reset"):
        action = "reset"
    
    return action


def display_element_properties(wn: WaterNetworkModel, element_name: str, element_type: str):
    """
    Display detailed properties of the selected network element.
    
    This component shows technical details about the currently selected
    element, including both static properties (from the network model)
    and dynamic properties (from the current simulation state).
    
    Args:
        wn (WaterNetworkModel): The water network model
        element_name (str): Name/ID of the selected element
        element_type (str): Type of element (Junction, Tank, Reservoir, Pipe, etc.)
        
    Display features:
    - Creates the detailed info panel when elements are selected
    - Shows both design parameters and current simulation values
    - Helps users understand element characteristics and current state
    - Different properties shown for nodes vs links
    """
    st.markdown(f"**Element Details:**")
    
    try:
        if element_type in ['Junction', 'Tank', 'Reservoir']:
            # Handle node elements (junctions, tanks, reservoirs)
            element = wn.get_node(element_name)
            coords = element.coordinates
            st.write(f"- **Coordinates:** ({coords[0]:.2f}, {coords[1]:.2f})")
            
            # Show total current demand instead of just base demand
            if hasattr(element, 'demand_timeseries_list'):
                try:
                    current_time = wn.sim_time if hasattr(wn, 'sim_time') else 0
                    total_demand = element.demand_timeseries_list.at(current_time)
                    st.write(f"- **Total Current Demand:** {total_demand:.4f} m³/s")
                    
                    # Also show breakdown if multiple demand entries exist
                    if len(element.demand_timeseries_list) > 1:
                        st.write(f"- **Demand Entries:** {len(element.demand_timeseries_list)} total")
                        # Show first few entries
                        for i, demand_entry in enumerate(element.demand_timeseries_list._list[:3]):
                            category = demand_entry.category or "base"
                            st.write(f"  • {category}: {demand_entry.base_value:.4f} m³/s")
                        if len(element.demand_timeseries_list) > 3:
                            st.write(f"  • ... and {len(element.demand_timeseries_list) - 3} more")
                except:
                    # Fallback to base demand if calculation fails
                    if hasattr(element, 'base_demand'):
                        st.write(f"- **Base Demand:** {element.base_demand:.4f} m³/s")
                        
            if hasattr(element, 'elevation'):
                st.write(f"- **Elevation:** {element.elevation:.2f} m")
            if hasattr(element, 'pressure') and element.pressure is not None:
                st.write(f"- **Current Pressure:** {element.pressure:.2f} m")
                
        else:  # Handle link elements (pipes, pumps, valves)
            element = wn.get_link(element_name)
            st.write(f"- **Start Node:** {element.start_node_name}")
            st.write(f"- **End Node:** {element.end_node_name}")
            
            # Show link-specific properties
            if hasattr(element, 'length'):
                st.write(f"- **Length:** {element.length:.2f} m")
            if hasattr(element, 'diameter'):
                st.write(f"- **Diameter:** {element.diameter:.3f} m")
            if hasattr(element, 'flow') and element.flow is not None:
                st.write(f"- **Current Flow:** {element.flow:.4f} m³/s")
                
    except Exception as e:
        st.error(f"Error getting element properties: {e}")


def display_event_interface(element_name: str, element_type: str, element_category: str) -> Optional[Dict]:
    """Display event configuration interface and return created event."""
    available_events = get_available_events(element_type, element_category)
    
    if not available_events:
        st.info(f"No events available for {element_type}")
        return None
    
    # Event selection
    event_type = st.selectbox(
        "Event Type:",
        list(available_events.keys()),
        help="Select the type of event to apply"
    )
    
    if not event_type:
        return None
    
    event_config = available_events[event_type]
    params = event_config.get('params', [])
    defaults = event_config.get('defaults', [])
    
    # Scheduling time
    scheduled_time = st.number_input(
        "Schedule Time (seconds):",
        min_value=0.0,
        value=0.0,
        step=60.0,
        help="When to apply this event (in simulation seconds)"
    )
    
    # Parameter inputs
    parameters = {}
    for i, param in enumerate(params):
        default_value = defaults[i] if i < len(defaults) else None
        
        if param in ['leak_area', 'leak_discharge_coefficient', 'base_demand', 'diameter', 'speed', 'head']:
            parameters[param] = st.number_input(
                f"{param.replace('_', ' ').title()}:",
                value=float(default_value) if default_value is not None else 0.0,
                step=0.01,
                format="%.4f"
            )
        elif param in ['fire_flow_demand']:
            parameters[param] = st.number_input(
                f"{param.replace('_', ' ').title()}:",
                value=float(default_value) if default_value is not None else 0.0,
                step=0.1,
                format="%.2f"
            )
        elif param in ['fire_start', 'fire_end']:
            parameters[param] = st.number_input(
                f"{param.replace('_', ' ').title()} (seconds):",
                value=int(default_value) if default_value is not None else 0,
                step=60
            )
        elif param == 'pattern_name':
            parameters[param] = st.text_input(
                "Pattern Name:",
                value=str(default_value) if default_value is not None else ""
            )
        elif param in ['category', 'name']:
            parameters[param] = st.text_input(
                f"{param.title()}:",
                value=str(default_value) if default_value is not None else ""
            )
        else:
            # Generic parameter input
            if default_value is not None:
                if isinstance(default_value, (int, float)):
                    parameters[param] = st.number_input(
                        f"{param.replace('_', ' ').title()}:",
                        value=float(default_value)
                    )
                else:
                    parameters[param] = st.text_input(
                        f"{param.replace('_', ' ').title()}:",
                        value=str(default_value)
                    )
            else:
                parameters[param] = st.text_input(f"{param.replace('_', ' ').title()}:")
    
    # Create event button
    if st.button("⚡ Schedule Event", use_container_width=True):
        return create_event(
            element_name, element_type, element_category,
            event_type, scheduled_time, parameters
        )
    
    return None


def display_monitoring_controls(wn: WaterNetworkModel, session_key_prefix: str = "") -> tuple:
    """Display monitoring controls for selecting nodes and links to monitor."""
    monitoring_col1, monitoring_col2 = st.columns(2)
    
    with monitoring_col1:
        st.markdown("**📍 Select Nodes to Monitor:**")
        available_nodes = list(wn.node_name_list)
        
        # Use the widget key as the primary session state key to avoid mismatches
        widget_key = f"{session_key_prefix}pressure_monitoring_nodes"
        
        # Initialize or validate existing session state
        if widget_key not in st.session_state:
            # First time - use first few nodes
            st.session_state[widget_key] = available_nodes[:DEFAULT_MONITORED_NODES_COUNT]
        else:
            # Validate that existing selections still exist in current network
            current_selections = st.session_state[widget_key]
            valid_selections = [node for node in current_selections if node in available_nodes]
            
            # If none of the previous selections are valid, reset to defaults
            if not valid_selections:
                st.session_state[widget_key] = available_nodes[:DEFAULT_MONITORED_NODES_COUNT]
            else:
                st.session_state[widget_key] = valid_selections
        
        monitored_nodes = st.multiselect(
            "Choose nodes for pressure monitoring:",
            available_nodes,
            default=st.session_state[widget_key],
            key=widget_key
        )
        
        # Also maintain backward compatibility with the old session state key
        old_session_key = f"{session_key_prefix}monitored_nodes"
        st.session_state[old_session_key] = monitored_nodes
        
        # Initialize data structure for newly selected nodes
        simulation_data_key = f"{session_key_prefix}simulation_data" if session_key_prefix else "simulation_data"
        if simulation_data_key in st.session_state:
            simulation_data = st.session_state[simulation_data_key]
            for node_name in monitored_nodes:
                if node_name not in simulation_data['pressures']:
                    # Initialize with current data if available, or empty list
                    simulation_data['pressures'][node_name] = []
                    # If we have time data, backfill with current pressure value
                    if simulation_data['time'] and node_name in wn.node_name_list:
                        try:
                            current_pressure = getattr(wn.get_node(node_name), 'pressure', 0) or 0
                            current_pressure = float(current_pressure)
                            # Backfill with current value
                            simulation_data['pressures'][node_name] = [current_pressure] * len(simulation_data['time'])
                        except:
                            # Fallback to zeros if we can't get current value
                            simulation_data['pressures'][node_name] = [0.0] * len(simulation_data['time'])
    
    with monitoring_col2:
        st.markdown("**🔗 Select Links to Monitor:**")
        available_links = list(wn.link_name_list)
        
        # Use the widget key as the primary session state key to avoid mismatches
        widget_key = f"{session_key_prefix}flow_monitoring_links"
        
        # Initialize or validate existing session state
        if widget_key not in st.session_state:
            # First time - use first few links
            st.session_state[widget_key] = available_links[:DEFAULT_MONITORED_LINKS_COUNT]
        else:
            # Validate that existing selections still exist in current network
            current_selections = st.session_state[widget_key]
            valid_selections = [link for link in current_selections if link in available_links]
            
            # If none of the previous selections are valid, reset to defaults
            if not valid_selections:
                st.session_state[widget_key] = available_links[:DEFAULT_MONITORED_LINKS_COUNT]
            else:
                st.session_state[widget_key] = valid_selections
        
        monitored_links = st.multiselect(
            "Choose links for flow monitoring:",
            available_links,
            default=st.session_state[widget_key],
            key=widget_key
        )
        
        # Also maintain backward compatibility with the old session state key
        old_session_key = f"{session_key_prefix}monitored_links"
        st.session_state[old_session_key] = monitored_links
        
        # Initialize data structure for newly selected links
        simulation_data_key = f"{session_key_prefix}simulation_data" if session_key_prefix else "simulation_data"
        if simulation_data_key in st.session_state:
            simulation_data = st.session_state[simulation_data_key]
            for link_name in monitored_links:
                if link_name not in simulation_data['flows']:
                    # Initialize with current data if available, or empty list
                    simulation_data['flows'][link_name] = []
                    # If we have time data, backfill with current flow value
                    if simulation_data['time'] and link_name in wn.link_name_list:
                        try:
                            current_flow = getattr(wn.get_link(link_name), 'flow', 0) or 0
                            current_flow = float(current_flow)
                            # Backfill with current value
                            simulation_data['flows'][link_name] = [current_flow] * len(simulation_data['time'])
                        except:
                            # Fallback to zeros if we can't get current value
                            simulation_data['flows'][link_name] = [0.0] * len(simulation_data['time'])
    
    return monitored_nodes, monitored_links


def display_current_values(wn: WaterNetworkModel, monitored_nodes: List[str], 
                          monitored_links: List[str], title_prefix: str = ""):
    """Display current values for monitored elements."""
    if not monitored_nodes and not monitored_links:
        return
        
    st.markdown(f'<h6 class="section-header">📈 {title_prefix}Current Values for Monitored Elements</h6>', 
                unsafe_allow_html=True)
    
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


def display_events_summary(scheduled_events: List[Dict], applied_events: List[Dict]):
    """Display events summary and recent events."""
    st.markdown("---")
    st.markdown('<h4 class="section-header">📅 Events</h4>', unsafe_allow_html=True)
    
    event_col1, event_col2 = st.columns(2)
    with event_col1:
        st.metric("Scheduled", len(scheduled_events))
    with event_col2:
        st.metric("Applied", len(applied_events))
    
    # Upcoming events with ability to remove
    if scheduled_events:
        with st.expander("⏰ Upcoming Events", expanded=False):
            # Show up to 5 upcoming events (last added first)
            for idx, event in enumerate(reversed(scheduled_events[-5:])):
                col_desc, col_btn = st.columns([4, 1])
                with col_desc:
                    st.write(f"- {event['event_type']} → {event['element_name']}")
                with col_btn:
                    # Unique key per button so Streamlit keeps state correctly
                    if st.button("🗑️ Remove", key=f"remove_evt_{event.get('time', idx)}_{idx}"):
                        # Remove the event from the original list held in session_state
                        try:
                            scheduled_events.remove(event)
                            st.experimental_rerun()
                        except ValueError:
                            pass
            st.markdown("---")
            if st.button("🧹 Clear All Pending", key="clear_all_events"):
                scheduled_events.clear()
                st.experimental_rerun()
    
    if applied_events:
        with st.expander("✅ Recent Events", expanded=False):
            for event in applied_events[-3:]:  # Show last 3
                st.write(f"- {event['event_type']} → {event['element_name']}")


def display_applied_events_history(applied_events: List[Dict], title_prefix: str = ""):
    """Display applied events history."""
    st.markdown(f'<h5 class="section-header">📋 {title_prefix}Applied Events History</h5>', 
                unsafe_allow_html=True)
    
    if not applied_events:
        st.info("📝 No events applied yet. Events will appear here once the simulation processes them.")
        return
    
    # Recent events display
    recent_events = applied_events[-5:]  # Show last 5
    for event in reversed(recent_events):
        event_time = datetime.timedelta(seconds=int(event['time']))
        st.success(f"✅ **{event_time}** - {event['description']}")
    
    # Full history in expandable section
    with st.expander("📜 Complete Event History", expanded=False):
        applied_df = pd.DataFrame([
            {
                'Time (s)': event['time'],
                'Time': str(datetime.timedelta(seconds=int(event['time']))),
                'Element': event['element_name'],
                'Event Type': event['event_type'],
                'Description': event['description']
            }
            for event in applied_events
        ])
        st.dataframe(applied_df, use_container_width=True)


def display_legend():
    """Display network legend."""
    if st.button("🧭 Legend", help="Show network element types"):
        with st.expander("🎨 Network Legend", expanded=True):
            st.markdown("""
            **Nodes:** 🔵 Junctions | 🔷 Tanks | 🟢 Reservoirs  
            **Links:** ▬ Pipes | ▬ Pumps | ▬ Valves  
            **Selected:** 🔴 Red highlights
            """)


def display_manual_selection(wn: WaterNetworkModel) -> Optional[Dict]:
    """Display manual element selection interface."""
    with st.expander("🔍 Manual Selection", expanded=False):
        node_options = ["None"] + list(wn.node_name_list)
        selected_node = st.selectbox("Node:", node_options, key="node_selector")
        
        link_options = ["None"] + list(wn.link_name_list)
        selected_link = st.selectbox("Link:", link_options, key="link_selector")
        
        if selected_node != "None":
            return {
                'name': selected_node,
                'category': 'node',
                'type': wn.get_node(selected_node).node_type
            }
            
        if selected_link != "None":
            return {
                'name': selected_link,
                'category': 'link',
                'type': wn.get_link(selected_link).link_type
            }
    
    return None


def display_batch_upload_interface():
    """Display batch event upload interface."""
    st.markdown('<h4 class="section-header">📂 Upload Event Batch</h4>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a JSON file with events:",
        type=['json'],
        help="Upload a JSON file containing a batch of events to simulate"
    )
    
    return uploaded_file


def display_example_event_format():
    """Display example event format."""
    with st.expander("📋 Example Event Format", expanded=False):
        st.markdown("""
        **Expected JSON format:**
        ```json
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
        ```
        """)


def display_footer(inp_file: str, duration_hours: int):
    """Display application footer with current network information."""
    import os
    
    st.markdown("---")
    
    # Get current network file from session state if available
    current_file = st.session_state.get('current_inp_file', inp_file)
    filename = os.path.basename(current_file)
    
    st.caption(f"Interactive Network Simulator | Network: {filename} | Duration: {duration_hours}h")


def display_network_file_selector() -> Optional[str]:
    """
    Display a professional network file selection interface.
    
    This component allows users to either use example network files
    or upload their own EPANET INP file for simulation.
    
    Returns:
        Optional[str]: Path to the selected INP file, or None if no valid file
        
    Interface design:
    - Shows available example networks
    - Provides professional file upload option
    - Validates uploaded files
    - Handles file naming and storage
    """
    st.markdown('<h4 class="section-header">📁 Network Configuration</h4>', unsafe_allow_html=True)
    
    # Network file selection options
    file_option = st.radio(
        "Network Source:",
        ["📊 Use Example Networks", "📤 Upload Custom Network"],
        help="Choose to use example networks or upload your own EPANET INP file"
    )
    
    if file_option == "📊 Use Example Networks":
        # Example network selection
        st.markdown("**Choose an example network:**")
        example_network = st.selectbox(
            "Example Networks:",
            [
                "NET_2.inp - Large Distribution Network",
                "NET_3.inp - Medium Distribution Network", 
                "NET_4.inp - Sample Distribution Network"
            ],
            help="Select from available example networks"
        )
        
        # Extract filename from selection
        selected_file = example_network.split(" - ")[0]
        
        # Show information about selected network
        with st.expander(f"ℹ️ {selected_file} Information", expanded=False):
            if selected_file == "NET_2.inp":
                st.info("""
                **NET_2.inp - Large Distribution Network**
                - Complex water distribution network with multiple zones
                - Contains numerous junctions, pipes, pumps, and tanks
                - Suitable for advanced simulation and analysis
                - Larger network for comprehensive testing
                """)
            elif selected_file == "NET_3.inp":
                st.info("""
                **NET_3.inp - Medium Distribution Network**
                - Moderate-sized water distribution network
                - Good balance of complexity and performance
                - Contains junctions, pipes, pumps, and tanks
                - Suitable for intermediate analysis and learning
                """)
            else:  # NET_4.inp
                st.info("""
                **NET_4.inp - Sample Distribution Network**
                - Sample water distribution network
                - Pre-configured for demonstration
                - Contains junctions, pipes, pumps, and tanks
                - Suitable for testing and learning
                """)
        
        return selected_file
    
    else:
        # Custom file upload
        st.markdown("**Upload Network File:**")
        uploaded_file = st.file_uploader(
            "Choose an EPANET INP file",
            type=['inp'],
            help="Upload a valid EPANET 2.0+ INP file for simulation",
            accept_multiple_files=False
        )
        
        if uploaded_file is not None:
            # Validate and save the uploaded file
            try:
                # Create a temporary filename
                import tempfile
                import os
                
                # Create temporary file with .inp extension
                with tempfile.NamedTemporaryFile(mode='w', suffix='.inp', delete=False) as tmp_file:
                    # Write uploaded content to temporary file
                    content = uploaded_file.getvalue().decode('utf-8')
                    tmp_file.write(content)
                    temp_path = tmp_file.name
                
                # Display file information
                file_size = len(content.encode('utf-8'))
                st.success(f"✅ **{uploaded_file.name}** uploaded successfully")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("File Size", f"{file_size:,} bytes")
                with col2:
                    st.metric("Format", "EPANET INP")
                
                # Show file validation status
                with st.expander("📋 File Validation", expanded=False):
                    # Basic validation - check for required sections
                    validation_results = validate_inp_file_content(content)
                    
                    for check, status in validation_results.items():
                        if status:
                            st.success(f"✅ {check}")
                        else:
                            st.warning(f"⚠️ {check}")
                
                return temp_path
                
            except Exception as e:
                st.error(f"❌ Error processing uploaded file: {str(e)}")
                st.info("Please ensure the file is a valid EPANET INP format.")
                return None
        else:
            st.info("👆 Please upload an EPANET INP file to proceed")
            return None


def validate_inp_file_content(content: str) -> Dict[str, bool]:
    """
    Validate basic structure of INP file content.
    
    Args:
        content (str): Content of the INP file
        
    Returns:
        Dict[str, bool]: Validation results for different sections
    """
    content_upper = content.upper()
    
    validation_results = {
        "Contains [JUNCTIONS] section": "[JUNCTIONS]" in content_upper,
        "Contains [PIPES] section": "[PIPES]" in content_upper,
        "Contains [RESERVOIRS] or [TANKS] section": 
            "[RESERVOIRS]" in content_upper or "[TANKS]" in content_upper,
        "Contains [COORDINATES] section": "[COORDINATES]" in content_upper,
        "File size is reasonable (< 10MB)": len(content.encode('utf-8')) < 10 * 1024 * 1024,
        "Contains [END] marker": "[END]" in content_upper
    }
    
    return validation_results 