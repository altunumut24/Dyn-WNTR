"""
UI Components for Interactive Network Simulator.
Contains reusable Streamlit interface components and functions.
"""

import streamlit as st
import datetime
import pandas as pd
from typing import List, Dict, Any, Optional

from mwntr.network import WaterNetworkModel
from mwntr.sim.interactive_network_simulator import MWNTRInteractiveSimulator
from config import NODE_EVENTS, LINK_EVENTS, DEFAULT_MONITORED_NODES_COUNT, DEFAULT_MONITORED_LINKS_COUNT
from simulation import get_available_events, create_event


def display_network_status(wn: WaterNetworkModel, current_element: Optional[Dict], 
                          scheduled_events: List[Dict], applied_events: List[Dict]):
    """Display network status dashboard."""
    st.markdown('<h3 class="section-header">📊 Network Status</h3>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Nodes", len(wn.node_name_list), help="Junction, Tank, and Reservoir nodes")
    with col2:
        st.metric("Total Links", len(wn.link_name_list), help="Pipes, Pumps, and Valves")
    with col3:
        if current_element:
            st.metric("Selected Element", current_element['name'], 
                     delta=current_element['type'], delta_color="normal")
        else:
            st.metric("Selected Element", "None", help="Click on network elements to select")
    with col4:
        st.metric("Scheduled Events", len(scheduled_events), 
                 delta=len(applied_events), delta_color="normal")

    st.markdown("---")


def display_simulation_controls(sim: MWNTRInteractiveSimulator, current_sim_time: float, key_prefix: str = "") -> str:
    """Display simulation control buttons and status. Returns action taken."""
    st.markdown('<h4 class="section-header">⚡ Simulation</h4>', unsafe_allow_html=True)
    
    # Status display
    if sim.initialized_simulation:
        current_time = datetime.timedelta(seconds=int(current_sim_time))
        st.success(f"🟢 **Active** | Time: {current_time}")
    else:
        st.warning("🟡 **Not Initialized**")
    
    # Control buttons
    sim_col1, sim_col2 = st.columns(2)
    
    action = None
    
    with sim_col1:
        if st.button("🚀 Initialize", disabled=sim.initialized_simulation, use_container_width=True, key=f"{key_prefix}initialize"):
            action = "initialize"
    
    with sim_col2:
        step_disabled = not sim.initialized_simulation or sim.is_terminated()
        if st.button("⏭️ Step", disabled=step_disabled, use_container_width=True, key=f"{key_prefix}step"):
            action = "step"
    
    # Reset button
    if st.button("🔄 Reset", use_container_width=True, key=f"{key_prefix}reset"):
        action = "reset"
    
    return action


def display_element_properties(wn: WaterNetworkModel, element_name: str, element_type: str):
    """Display properties of selected element."""
    st.markdown(f"**Element Details:**")
    
    try:
        if element_type in ['Junction', 'Tank', 'Reservoir']:
            element = wn.get_node(element_name)
            coords = element.coordinates
            st.write(f"- **Coordinates:** ({coords[0]:.2f}, {coords[1]:.2f})")
            
            if hasattr(element, 'base_demand'):
                st.write(f"- **Base Demand:** {element.base_demand:.4f} m³/s")
            if hasattr(element, 'elevation'):
                st.write(f"- **Elevation:** {element.elevation:.2f} m")
            if hasattr(element, 'pressure') and element.pressure is not None:
                st.write(f"- **Current Pressure:** {element.pressure:.2f} m")
                
        else:  # Link
            element = wn.get_link(element_name)
            st.write(f"- **Start Node:** {element.start_node_name}")
            st.write(f"- **End Node:** {element.end_node_name}")
            
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
        
        session_key = f"{session_key_prefix}monitored_nodes"
        if session_key not in st.session_state:
            st.session_state[session_key] = available_nodes[:DEFAULT_MONITORED_NODES_COUNT]
        
        monitored_nodes = st.multiselect(
            "Choose nodes for pressure monitoring:",
            available_nodes,
            default=st.session_state[session_key],
            key=f"{session_key_prefix}pressure_monitoring_nodes"
        )
        st.session_state[session_key] = monitored_nodes
    
    with monitoring_col2:
        st.markdown("**🔗 Select Links to Monitor:**")
        available_links = list(wn.link_name_list)
        
        session_key = f"{session_key_prefix}monitored_links"
        if session_key not in st.session_state:
            st.session_state[session_key] = available_links[:DEFAULT_MONITORED_LINKS_COUNT]
        
        monitored_links = st.multiselect(
            "Choose links for flow monitoring:",
            available_links,
            default=st.session_state[session_key],
            key=f"{session_key_prefix}flow_monitoring_links"
        )
        st.session_state[session_key] = monitored_links
    
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
    
    # Quick event list
    if scheduled_events:
        with st.expander("⏰ Upcoming Events", expanded=False):
            for event in scheduled_events[-3:]:  # Show last 3
                st.write(f"- {event['event_type']} → {event['element_name']}")
    
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
    """Display application footer."""
    st.markdown("---")
    st.caption(f"Interactive Network Simulator | Network: {inp_file} | Duration: {duration_hours}h") 