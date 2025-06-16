"""
Interactive Water Network Event Simulator - Main Application
A Streamlit-based application for interactive water network simulation with event scheduling.
"""

import streamlit as st
import time
import datetime
from typing import Dict, Any

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
from modules.ui_components import (
    display_network_status, display_simulation_controls, display_element_properties,
    display_event_interface, display_monitoring_controls, display_current_values,
    display_events_summary, display_applied_events_history, display_legend,
    display_manual_selection, display_batch_upload_interface, 
    display_example_event_format, display_footer
)

# Streamlit page configuration
st.set_page_config(
    page_title="Interactive Network Simulator",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better space utilization
st.markdown("""
<style>
    /* Reduce padding and margins for better space utilization */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* Ensure plotly charts use full container width */
    .js-plotly-plot, .plotly {
        width: 100% !important;
    }
    
    /* Optimize plotly text rendering for different DPI settings */
    .plotly .main-svg {
        font-family: "Arial", sans-serif;
    }
    
    /* Optimize column spacing */
    .element-container {
        margin-bottom: 0.5rem;
    }
    
    /* Make section headers more compact */
    .section-header {
        margin-bottom: 0.5rem !important;
        margin-top: 0.5rem !important;
    }
    
    /* Optimize for cloud deployment rendering */
    .stSelectbox > div > div {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize all session state variables."""
    # Interactive mode state
    session_vars = {
        'wn': None,
        'sim': None,
        'selected_nodes': [],
        'selected_links': [],
        'current_element': None,
        'scheduled_events': [],
        'applied_events': [],
        'current_sim_time': 0,
        'simulation_data': initialize_simulation_data(),
        
        # Batch mode state  
        'batch_wn': None,
        'batch_sim': None,
        'loaded_events': [],
        'event_metadata': {},
        'batch_current_sim_time': 0,
        'batch_scheduled_events': [],
        'batch_applied_events': [],
        'batch_simulation_data': initialize_simulation_data()
    }
    
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value


def load_network_models():
    """Load network models for both interactive and batch modes."""
    if st.session_state.wn is None:
        with st.spinner("🔄 Loading network model..."):
            # Load for interactive mode
            st.session_state.wn = load_network_model(INP_FILE)
            if st.session_state.wn:
                from mwntr.sim.interactive_network_simulator import MWNTRInteractiveSimulator
                st.session_state.sim = MWNTRInteractiveSimulator(st.session_state.wn)
                
                # Also load for batch mode
                st.session_state.batch_wn = load_network_model(INP_FILE)
                if st.session_state.batch_wn:
                    st.session_state.batch_sim = MWNTRInteractiveSimulator(st.session_state.batch_wn)
                
                st.success(f"✅ Network '{INP_FILE}' loaded successfully!")
                time.sleep(1)
            else:
                st.stop()


def handle_simulation_controls(sim, wn, simulation_data, session_prefix=""):
    """Handle simulation control actions."""
    action = display_simulation_controls(sim, st.session_state[f'{session_prefix}current_sim_time'], session_prefix)
    
    if action == "initialize":
        try:
            with st.spinner("Initializing..."):
                # Use user-provided planning values
                duration_seconds = st.session_state.get('simulation_duration_hours', 24) * 3600
                timestep_seconds = st.session_state.get('simulation_timestep_minutes', 60) * 60
                sim.init_simulation(
                    global_timestep=timestep_seconds,
                    duration=duration_seconds
                )
                # Remember what this run uses so all progress widgets stay consistent
                st.session_state.run_duration_seconds = duration_seconds
                st.session_state.run_timestep_seconds = timestep_seconds
                st.session_state[f'{session_prefix}current_sim_time'] = 0
            st.success("✅ Ready!")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed: {e}")
    
    elif action == "step":
        with st.spinner("Processing..."):
            handle_simulation_step(sim, wn, simulation_data, session_prefix)
        time.sleep(0.5)
        st.rerun()
    
    elif action == "reset":
        with st.spinner("Resetting..."):
            reset_simulation(wn, session_prefix)
        st.success("🔄 Reset complete!")
        time.sleep(1)
        st.rerun()


def handle_simulation_step(sim, wn, simulation_data, session_prefix=""):
    """Handle a single simulation step."""
    scheduled_events_key = f'{session_prefix}scheduled_events' if session_prefix else 'scheduled_events'
    applied_events_key = f'{session_prefix}applied_events' if session_prefix else 'applied_events'
    current_time_key = f'{session_prefix}current_sim_time'
    
    # Apply scheduled events
    events_to_apply = [e for e in st.session_state[scheduled_events_key] 
                      if e.get('scheduled_time', e.get('time', 0)) <= st.session_state[current_time_key]]
    
    for event in events_to_apply:
        success, message = apply_event_to_simulator(sim, wn, event)
        if success:
            st.session_state[applied_events_key].append(event)
            st.success(f"⚡ {message}")
        else:
            st.error(f"❌ {message}")
        st.session_state[scheduled_events_key].remove(event)
    
    # Run simulation step
    success, sim_time, message = run_simulation_step(sim, wn)
    st.session_state[current_time_key] = sim_time
    
    if success:
        # Collect data for monitored elements
        simulation_data['time'].append(sim_time)
        
        # Get monitored elements
        monitored_nodes_key = f'{session_prefix}monitored_nodes' if session_prefix else 'monitored_nodes'
        monitored_links_key = f'{session_prefix}monitored_links' if session_prefix else 'monitored_links'
        
        monitored_nodes = st.session_state.get(monitored_nodes_key, list(wn.node_name_list)[:5])
        monitored_links = st.session_state.get(monitored_links_key, list(wn.link_name_list)[:5])
        
        collect_simulation_data(wn, monitored_nodes, monitored_links, simulation_data)
        st.success("✅ Step completed!")
    else:
        st.warning(f"⚠️ {message}")


def reset_simulation(wn, session_prefix=""):
    """Reset simulation state."""
    if session_prefix:
        st.session_state[f'{session_prefix}sim'] = reset_simulation_state(wn)
        st.session_state[f'{session_prefix}current_sim_time'] = 0
        st.session_state[f'{session_prefix}applied_events'] = []
        st.session_state[f'{session_prefix}simulation_data'] = initialize_simulation_data()
        if session_prefix == 'batch_':
            st.session_state['loaded_events'] = []
    else:
        st.session_state.sim = reset_simulation_state(wn)
        st.session_state.current_sim_time = 0
        st.session_state.scheduled_events = []
        st.session_state.applied_events = []
        st.session_state.simulation_data = initialize_simulation_data()


def handle_network_click(chart_data):
    """Handle clicks on the network plot."""
    if chart_data and chart_data.get('selection') and chart_data['selection']['points']:
        selected_point = chart_data['selection']['points'][0]
        
        if selected_point.get('customdata'):
            custom_data = selected_point['customdata']
            element_name = custom_data[0]
            element_type = custom_data[1]
            element_category = custom_data[2]  # 'node' or 'link'
            
            # Update session state only if it's a new selection
            if (not st.session_state.current_element or 
                st.session_state.current_element['name'] != element_name):
                
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


def display_interactive_mode():
    """Display the interactive mode tab."""
    wn = st.session_state.wn
    sim = st.session_state.sim
    
    # Network Status Dashboard
    with st.container():
        display_network_status(
            wn, st.session_state.current_element,
            st.session_state.scheduled_events, st.session_state.applied_events
        )
    
    # Interactive Network Visualization with Side Panel
    st.markdown('<h3 class="section-header">🗺️ Interactive Network Map & Event Configuration</h3>', 
                unsafe_allow_html=True)
    
    # Layout controls
    fullscreen_map = st.toggle("🔍 Fullscreen Map", value=False, help="Show map in fullscreen mode")
    
    if fullscreen_map:
        # Fullscreen mode - no columns, use full width
        display_fullscreen_network_map(wn, sim)
        
        # Show controls below the map in fullscreen mode
        st.markdown("---")
        control_col1, control_col2 = st.columns([1, 1])
        with control_col1:
            display_simulation_controls_compact(sim, wn)
        with control_col2:
            display_element_configuration_compact(wn)
    else:
        # Normal column layout with optimized ratio for better map visibility
        map_col, control_col = st.columns([5, 1])  # Give more space to map
        
        with map_col:
            display_network_map_section(wn, sim)
        
        with control_col:
            display_control_panel(wn, sim)
    
    # Results visualization (always show below the main layout)
    if len(st.session_state.simulation_data['time']) > 1:
        st.markdown("---")
        display_simulation_results(wn, st.session_state.simulation_data)


def display_color_legends(wn):
    """Display color scale legends for pressure and flow."""
    # Get current data for legends
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
            min_pressure, max_pressure = min(pressure_values), max(pressure_values)
        else:
            min_pressure = max_pressure = 0
    else:
        min_pressure = max_pressure = 0
    
    if link_flows:
        flow_values = [float(f) for f in link_flows.values() if f is not None]
        if flow_values:
            min_flow, max_flow = min(flow_values), max(flow_values)
        else:
            min_flow = max_flow = 0
    else:
        min_flow = max_flow = 0
    
    # Display legends side by side
    legend_col1, legend_col2 = st.columns(2)
    with legend_col1:
        if min_pressure != max_pressure or min_pressure != 0:
            pressure_fig = create_pressure_colorbar(min_pressure, max_pressure)
            st.plotly_chart(pressure_fig, use_container_width=True)
        else:
            st.info("🔵 **Pressure**: No data yet")
    with legend_col2:
        if min_flow != max_flow or min_flow != 0:
            flow_fig = create_flow_colorbar(min_flow, max_flow)
            st.plotly_chart(flow_fig, use_container_width=True)
        else:
            st.info("🔗 **Flow**: No data yet")


def display_element_configuration(wn):
    """Display element selection and configuration interface."""
    st.markdown('<h4 class="section-header">🎯 Element Configuration</h4>', unsafe_allow_html=True)
    
    if st.session_state.current_element:
        element = st.session_state.current_element
        
        # Selected element display
        st.success(f"**Selected:** {element['name']}")
        st.info(f"**Type:** {element['type']} ({element['category']})")
        
        # Element properties
        display_element_properties(wn, element['name'], element['type'])
        
        # Clear selection button
        if st.button("🧹 Clear selection", use_container_width=True):
            st.session_state.selected_nodes = []
            st.session_state.selected_links = []
            st.session_state.current_element = None
            st.rerun()
        
        # Event interface
        st.markdown("**⚡ Schedule Event:**")
        new_event = display_event_interface(element['name'], element['type'], element['category'])
        if new_event:
            st.session_state.scheduled_events.append(new_event)
            sched_time = new_event.get('scheduled_time', 0)
            st.success(f"✅ Scheduled {new_event['event_type']} on {new_event['element_name']} at {sched_time}s")
            time.sleep(1)
            st.rerun()
    else:
        st.info("👆 **Select an element** from the network map to configure events")
        
        # Manual selection fallback
        manual_element = display_manual_selection(wn)
        if manual_element:
            st.session_state.current_element = manual_element
            if manual_element['category'] == 'node':
                st.session_state.selected_nodes = [manual_element['name']]
                st.session_state.selected_links = []
            else:
                st.session_state.selected_links = [manual_element['name']]
                st.session_state.selected_nodes = []
            st.rerun()


def display_simulation_results(wn, simulation_data):
    """Display simulation results and monitoring charts."""
    with st.container():
        st.markdown('<h3 class="section-header">📊 Simulation Results</h3>', unsafe_allow_html=True)
        
        # Monitoring controls
        monitored_nodes, monitored_links = display_monitoring_controls(wn)
        
        # Create and display charts
        fig_pressure, fig_flow = create_monitoring_charts(simulation_data, monitored_nodes, monitored_links)
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            if fig_pressure:
                st.plotly_chart(fig_pressure, use_container_width=True)
            else:
                st.info("📍 Select nodes above to see pressure monitoring")
        
        with chart_col2:
            if fig_flow:
                st.plotly_chart(fig_flow, use_container_width=True)
            else:
                st.info("🔗 Select links above to see flow monitoring")
        
        # Current values
        if st.session_state.current_sim_time > 0:
            display_current_values(wn, monitored_nodes, monitored_links)


def display_simulation_planning():
    """Display simulation planning controls for duration and timestep."""
    st.markdown('<h4 class="section-header">⏱️ Simulation Planning</h4>', unsafe_allow_html=True)
    
    # Initialize planning session state
    if 'simulation_duration_hours' not in st.session_state:
        st.session_state.simulation_duration_hours = 24
    if 'simulation_timestep_minutes' not in st.session_state:
        st.session_state.simulation_timestep_minutes = 60
    
    plan_col1, plan_col2 = st.columns(2)
    
    with plan_col1:
        duration_hours = st.number_input(
            "Simulation Duration (hours)",
            min_value=1,
            max_value=168,  # 1 week max
            value=st.session_state.simulation_duration_hours,
            step=1,
            help="Total time to simulate"
        )
        st.session_state.simulation_duration_hours = duration_hours
    
    with plan_col2:
        timestep_minutes = st.selectbox(
            "Timestep (minutes)",
            options=[15, 30, 60, 120, 180, 360],  # Common timesteps
            index=[15, 30, 60, 120, 180, 360].index(st.session_state.simulation_timestep_minutes),
            help="Time between simulation steps"
        )
        st.session_state.simulation_timestep_minutes = timestep_minutes
    
    # Calculate total steps
    total_duration_seconds = duration_hours * 3600
    timestep_seconds = timestep_minutes * 60
    total_steps = int(total_duration_seconds / timestep_seconds)
    
    # Display plan summary
    st.info(f"**Plan:** {total_steps} steps over {duration_hours} hours (every {timestep_minutes} minutes)")
    
    return total_duration_seconds, timestep_seconds, total_steps


def display_simulation_progress_compact(simulation_data):
    """Display compact simulation progress in the controls area."""
    st.markdown('<h4 class="section-header">⏱️ Progress</h4>', unsafe_allow_html=True)
    
    # Get planning data
    total_duration_seconds = st.session_state.get('run_duration_seconds', st.session_state.get('simulation_duration_hours', 24)*3600)
    timestep_seconds = st.session_state.get('run_timestep_seconds', st.session_state.get('simulation_timestep_minutes', 60)*60)
    planned_total_steps = int(total_duration_seconds / timestep_seconds)
    
    # Current progress
    current_time = simulation_data['time'][-1]
    completed_steps = len(simulation_data['time'])
    
    # Calculate progress
    time_progress = min(current_time / total_duration_seconds, 1.0)
    
    # Compact metrics
    prog_col1, prog_col2 = st.columns(2)
    
    with prog_col1:
        st.metric("Steps", f"{completed_steps}/{planned_total_steps}")
    
    with prog_col2:
        remaining_steps = max(0, planned_total_steps - completed_steps)
        st.metric("Remaining", f"{remaining_steps} steps")
    
    # Progress bar
    st.progress(time_progress, text=f"{time_progress*100:.1f}% Complete")
    
    # Time info
    remaining_time = max(0, total_duration_seconds - current_time)
    st.caption(f"⏰ {datetime.timedelta(seconds=int(current_time))} elapsed, {datetime.timedelta(seconds=int(remaining_time))} remaining")


def display_simulation_progress(simulation_data):
    """Display simulation progress timeline and statistics."""
    st.markdown('<h4 class="section-header">📊 Simulation Progress</h4>', unsafe_allow_html=True)
    
    if not simulation_data['time']:
        st.info("No simulation data yet - start simulation to see progress")
        return
    
    # Get planning data
    total_duration_seconds = st.session_state.get('run_duration_seconds', st.session_state.get('simulation_duration_hours', 24)*3600)
    timestep_seconds = st.session_state.get('run_timestep_seconds', st.session_state.get('simulation_timestep_minutes', 60)*60)
    planned_total_steps = int(total_duration_seconds / timestep_seconds)
    
    # Current progress
    current_time = simulation_data['time'][-1]
    completed_steps = len(simulation_data['time'])
    
    # Calculate progress
    time_progress = min(current_time / total_duration_seconds, 1.0)
    step_progress = min(completed_steps / planned_total_steps, 1.0)
    
    # Display metrics
    prog_col1, prog_col2, prog_col3, prog_col4 = st.columns(4)
    
    with prog_col1:
        st.metric("Completed Steps", f"{completed_steps}/{planned_total_steps}")
    
    with prog_col2:
        st.metric("Simulation Time", f"{datetime.timedelta(seconds=int(current_time))}")
    
    with prog_col3:
        remaining_time = max(0, total_duration_seconds - current_time)
        st.metric("Time Remaining", f"{datetime.timedelta(seconds=int(remaining_time))}")
    
    with prog_col4:
        st.metric("Progress", f"{time_progress*100:.1f}%")
    
    # Progress bar
    st.progress(time_progress, text=f"Simulation Progress: {time_progress*100:.1f}%")
    
    # Timeline visualization
    if completed_steps > 1:
        import plotly.graph_objects as go
        
        # Create timeline chart
        fig = go.Figure()
        
        # Add completed time line
        fig.add_trace(go.Scatter(
            x=simulation_data['time'],
            y=[1] * len(simulation_data['time']),
            mode='lines+markers',
            name='Completed',
            line=dict(color='#0072B2', width=4),  # blue
            marker=dict(size=8, color='#0072B2')
        ))
        
        # Add remaining time line
        if current_time < total_duration_seconds:
            remaining_times = list(range(int(current_time), int(total_duration_seconds) + timestep_seconds, timestep_seconds))
            fig.add_trace(go.Scatter(
                x=remaining_times,
                y=[1] * len(remaining_times),
                mode='lines+markers',
                name='Remaining',
                line=dict(color='#999999', width=2, dash='dash'),  # gray
                marker=dict(size=6, color='#999999')
            ))
        
        # Add current position marker
        fig.add_trace(go.Scatter(
            x=[current_time],
            y=[1],
            mode='markers',
            name='Current',
            marker=dict(size=15, color='#E69F00', symbol='diamond')  # orange
        ))
        
        fig.update_layout(
            title="🕐 Simulation Timeline",
            xaxis_title="Time (seconds)",
            yaxis=dict(showticklabels=False, range=[0.5, 1.5]),
            height=200,
            showlegend=True,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)


def display_batch_mode():
    """Display the batch simulator tab."""
    st.markdown('<h3 class="section-header">📋 Batch Event Simulator</h3>', unsafe_allow_html=True)
    st.info("Load a JSON file with pre-defined events to run automated simulations")
    
    uploaded_file = display_batch_upload_interface()
    
    if uploaded_file:
        # Load events
        if not st.session_state.loaded_events:
            events, metadata = load_events_from_json(uploaded_file)
            if events:
                st.session_state.loaded_events = events
                st.session_state.event_metadata = metadata
                # Copy events to scheduled events for processing
                st.session_state.batch_scheduled_events = events.copy()
                st.success(f"✅ Loaded {len(events)} events from {uploaded_file.name}")
        
        # Display loaded events info
        if st.session_state.loaded_events:
            st.markdown(f"**📄 Events loaded:** {len(st.session_state.loaded_events)}")
            
            # Event timeline
            display_event_timeline(st.session_state.loaded_events, st.session_state.batch_current_sim_time)
            
            # Batch simulation controls
            batch_wn = st.session_state.batch_wn
            batch_sim = st.session_state.batch_sim
            
            handle_simulation_controls(batch_sim, batch_wn, st.session_state.batch_simulation_data, "batch_")
            
            # Network visualization for batch mode
            show_batch_sim_data = st.toggle(
                "🌊 Live Batch Visualization",
                value=batch_sim.initialized_simulation,
                disabled=not batch_sim.initialized_simulation,
                help="Show real-time data during batch simulation"
            )
            
            batch_fig = create_network_plot(
                batch_wn,
                [],  # No selection in batch mode
                [],
                show_simulation_data=show_batch_sim_data,
                sim_initialized=batch_sim.initialized_simulation,
                height=800,  # Default height for batch mode
                node_size_scale=0.8  # Smaller nodes for batch mode
            )
            
            st.plotly_chart(batch_fig, use_container_width=True)
            
            # Color scale legends for batch mode
            if show_batch_sim_data and batch_sim.initialized_simulation:
                display_color_legends(batch_wn)
            
            # Applied events history
            display_applied_events_history(st.session_state.batch_applied_events, "Batch - ")
            
            # Batch simulation results
            if len(st.session_state.batch_simulation_data['time']) > 1:
                display_batch_results(batch_wn)
    else:
        st.info("👆 **Upload a JSON event file** to start batch simulation")
        display_example_event_format()


def display_batch_results(batch_wn):
    """Display batch simulation results."""
    st.markdown('<h5 class="section-header">📈 Simulation Data</h5>', unsafe_allow_html=True)
    
    # Monitoring controls for batch mode
    monitored_nodes, monitored_links = display_monitoring_controls(batch_wn, "batch_")
    
    # Create and display batch charts
    fig_pressure, fig_flow = create_monitoring_charts(
        st.session_state.batch_simulation_data, monitored_nodes, monitored_links
    )
    
    batch_chart_col1, batch_chart_col2 = st.columns(2)
    
    with batch_chart_col1:
        if fig_pressure:
            fig_pressure.update_layout(title="🔵 Batch - Node Pressure Over Time")
            st.plotly_chart(fig_pressure, use_container_width=True)
        else:
            st.info("📍 No pressure data available for selected nodes")
    
    with batch_chart_col2:
        if fig_flow:
            fig_flow.update_layout(title="🔗 Batch - Link Flow Over Time")
            st.plotly_chart(fig_flow, use_container_width=True)
        else:
            st.info("🔗 No flow data available for selected links")
    
    # Current values for batch mode
    if st.session_state.batch_current_sim_time > 0:
        display_current_values(batch_wn, monitored_nodes, monitored_links, "Batch - ")


def display_network_map_section(wn, sim):
    """Display the network map section with controls."""
    # Instructions and controls above the map
    st.info("🖱️ **Click on nodes (circles) or links (squares) to select and configure events**")
    
    # Visualization controls
    viz_col1, viz_col2, viz_col3, viz_col4 = st.columns([1, 1, 1, 1])
    with viz_col1:
        show_sim_data = st.toggle(
            "🌊 Live Visualization", 
            value=sim.initialized_simulation,
            disabled=not sim.initialized_simulation,
            help="Show real-time pressure and flow data"
        )
    with viz_col2:
        display_legend()
    with viz_col3:
        # Map size control
        map_height = st.selectbox("📏 Map Size", 
                                options=[700, 800, 900, 1000, 1200], 
                                index=2, 
                                help="Adjust map height")
    with viz_col4:
        # Node size control for different screen resolutions
        node_size_scale = st.selectbox("🔘 Node Size", 
                                     options=[0.6, 0.8, 1.0, 1.2, 1.4], 
                                     index=2, 
                                     format_func=lambda x: f"{int(x*100)}%",
                                     help="Adjust node size for your screen")
    
    # Network plot with dynamic height and node size
    fig = create_network_plot(
        wn, 
        st.session_state.selected_nodes, 
        st.session_state.selected_links,
        show_simulation_data=show_sim_data,
        sim_initialized=sim.initialized_simulation,
        height=map_height,
        node_size_scale=node_size_scale
    )
    
    # Display plot with click handling
    chart_data = st.plotly_chart(
        fig, 
        use_container_width=True, 
        key="network_plot_interaction",
        on_select="rerun",
        selection_mode="points"
    )
    
    # Handle click events
    handle_network_click(chart_data)
    
    # Color scale legends
    if show_sim_data and sim.initialized_simulation:
        display_color_legends(wn)


def display_fullscreen_network_map(wn, sim):
    """Display the network map in fullscreen mode."""
    st.markdown("### 🔍 Fullscreen Network Map")
    st.info("🖱️ **Click on nodes (circles) or links (squares) to select and configure events**")
    
    # Visualization controls in fullscreen
    viz_col1, viz_col2, viz_col3, viz_col4 = st.columns([1, 1, 1, 1])
    with viz_col1:
        show_sim_data = st.toggle(
            "🌊 Live Visualization", 
            value=sim.initialized_simulation,
            disabled=not sim.initialized_simulation,
            help="Show real-time pressure and flow data",
            key="fullscreen_sim_data"
        )
    with viz_col2:
        display_legend()
    with viz_col3:
        map_height = st.selectbox("📏 Map Size", 
                                options=[800, 900, 1000, 1200, 1400], 
                                index=2, 
                                help="Adjust map height",
                                key="fullscreen_map_height")
    with viz_col4:
        node_size_scale = st.selectbox("🔘 Node Size", 
                                     options=[0.6, 0.8, 1.0, 1.2, 1.4], 
                                     index=2, 
                                     format_func=lambda x: f"{int(x*100)}%",
                                     help="Adjust node size for your screen",
                                     key="fullscreen_node_size")
    
    # Network plot in fullscreen
    fig = create_network_plot(
        wn, 
        st.session_state.selected_nodes, 
        st.session_state.selected_links,
        show_simulation_data=show_sim_data,
        sim_initialized=sim.initialized_simulation,
        height=map_height,
        node_size_scale=node_size_scale
    )
    
    # Display plot with click handling
    chart_data = st.plotly_chart(
        fig, 
        use_container_width=True, 
        key="network_plot_interaction_fullscreen",
        on_select="rerun",
        selection_mode="points"
    )
    
    # Handle click events
    handle_network_click(chart_data)
    
    # Color scale legends
    if show_sim_data and sim.initialized_simulation:
        display_color_legends(wn)


def display_control_panel(wn, sim):
    """Display the control panel section."""
    st.markdown('<h3 class="section-header">🎮 Controls</h3>', unsafe_allow_html=True)
    
    # Simulation planning (before starting)
    if not sim.initialized_simulation:
        display_simulation_planning()
        st.markdown("---")
    
    # Simulation controls
    handle_simulation_controls(sim, wn, st.session_state.simulation_data)
    
    # Show progress during simulation
    if sim.initialized_simulation and len(st.session_state.simulation_data['time']) > 0:
        display_simulation_progress_compact(st.session_state.simulation_data)
    
    st.markdown("---")
    
    # Element configuration
    display_element_configuration(wn)
    
    # Events summary
    display_events_summary(st.session_state.scheduled_events, st.session_state.applied_events)
    
    # Applied events history
    display_applied_events_history(st.session_state.applied_events, "Interactive - ")


def display_simulation_controls_compact(sim, wn):
    """Display compact simulation controls for fullscreen mode."""
    st.markdown('<h4 class="section-header">🎮 Simulation Controls</h4>', unsafe_allow_html=True)
    
    # Simulation planning (before starting)
    if not sim.initialized_simulation:
        display_simulation_planning()
    
    # Simulation controls
    handle_simulation_controls(sim, wn, st.session_state.simulation_data)
    
    # Show progress during simulation
    if sim.initialized_simulation and len(st.session_state.simulation_data['time']) > 0:
        display_simulation_progress_compact(st.session_state.simulation_data)


def display_element_configuration_compact(wn):
    """Display compact element configuration for fullscreen mode."""
    st.markdown('<h4 class="section-header">🎯 Element Configuration</h4>', unsafe_allow_html=True)
    
    # Element configuration
    display_element_configuration(wn)
    
    # Events summary
    display_events_summary(st.session_state.scheduled_events, st.session_state.applied_events)


def main():
    """Main application function."""
    st.title("🌊 Interactive Water Network Event Simulator")
    
    # DEBUG TOGGLE - Add this to your sidebar
    debug_mode = st.sidebar.toggle("🐛 Debug Mode", value=False, help="Show debugging information")
    
    # Initialize session state
    initialize_session_state()
    
    # Load network models
    load_network_models()
    
    # Initialize active tab state
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "🎮 Interactive Mode"
    
    # Create pill-based tabs that maintain state
    tabs = ["🎮 Interactive Mode", "📋 Batch Simulator"]
    active_tab = st.pills("", tabs, selection_mode="single", default=st.session_state.active_tab)
    
    # Update session state when tab changes
    if active_tab != st.session_state.active_tab:
        st.session_state.active_tab = active_tab
    
    # DEBUG SECTION - Show debugging info if enabled
    if debug_mode:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🐛 Debug Info")
        with st.sidebar.expander("Session State", expanded=True):
            for key, value in st.session_state.items():
                st.sidebar.write(f"**{key}:** `{type(value).__name__}`")
                
                # Show actual values based on type
                if isinstance(value, (str, int, float, bool)):
                    st.sidebar.write(f"  Value: `{value}`")
                elif isinstance(value, list):
                    st.sidebar.write(f"  Length: `{len(value)}`")
                    if len(value) > 0 and len(value) < 5:
                        st.sidebar.write(f"  Items: `{value}`")
                    elif len(value) > 0:
                        st.sidebar.write(f"  First item: `{value[0]}`")
                elif isinstance(value, dict):
                    st.sidebar.write(f"  Keys: `{len(value)}`")
                    if len(value) > 0 and len(value) < 5:
                        for k, v in value.items():
                            st.sidebar.write(f"    {k}: `{v}`")
                elif value is None:
                    st.sidebar.write(f"  Value: `None`")
                else:
                    st.sidebar.write(f"  Object: `{str(value)[:50]}...`")
                st.sidebar.write("---")
    
    # Display content based on selected tab
    if st.session_state.active_tab == "🎮 Interactive Mode":
        display_interactive_mode()
        
        # Show debug info at bottom if enabled
        if debug_mode:
            st.markdown("---")
            st.markdown("## 🐛 Debug Information")
            col1, col2 = st.columns(2)
            with col1:
                debug_session_state()
            with col2:
                if st.session_state.wn and st.session_state.sim:
                    debug_network_state(st.session_state.wn, st.session_state.sim)
                debug_event_processing(st.session_state.scheduled_events, st.session_state.current_sim_time)
                    
    elif st.session_state.active_tab == "📋 Batch Simulator":
        display_batch_mode()
    
    # Footer
    display_footer(INP_FILE, SIMULATION_DURATION_SECONDS // 3600)


if __name__ == "__main__":
    main()
