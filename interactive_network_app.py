import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Import mwntr components
try:
    import mwntr
    from mwntr.sim.interactive_network_simulator import MWNTRInteractiveSimulator
    from mwntr.network import WaterNetworkModel, Node, Link, LinkStatus
    from mwntr.network.elements import Junction, Tank, Reservoir, Pipe, Pump, Valve
    from mwntr.graphics.network import plot_interactive_network
    MWNTR_AVAILABLE = True
except ImportError:
    MWNTR_AVAILABLE = False
    st.error("MWNTR library not available. Please install mwntr to use this application.")
    st.stop()

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
        wn = mwntr.network.WaterNetworkModel(inp_file)
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

def create_network_plot(wn: WaterNetworkModel, selected_nodes: List[str] = None, selected_links: List[str] = None):
    """Create interactive plotly network visualization with click handling."""
    node_positions, edge_list, node_info, edge_info = get_network_layout(wn)
    
    if selected_nodes is None:
        selected_nodes = []
    if selected_links is None:
        selected_links = []
    
    fig = go.Figure()
    
    # Add edges (connect nodes with lines)
    for link_name, info in edge_info.items():
        start_pos = node_positions[info['start']]
        end_pos = node_positions[info['end']]
        
        color = 'red' if link_name in selected_links else ('darkblue' if info['type'] == 'Pipe' else 'green' if info['type'] == 'Pump' else 'orange')
        width = 6 if link_name in selected_links else 3
        
        fig.add_trace(go.Scatter(
            x=[start_pos[0], end_pos[0]],
            y=[start_pos[1], end_pos[1]],
            mode='lines',
            line=dict(color=color, width=width),
            name=f"{info['type']}s",
            legendgroup=info['type'],
            showlegend=link_name == list(edge_info.keys())[0] or info['type'] not in [edge_info[k]['type'] for k in list(edge_info.keys())[:list(edge_info.keys()).index(link_name)]],
            hoverinfo='skip'  # Skip hover for lines
        ))
    
    # Add clickable markers at link midpoints for link selection
    link_x = []
    link_y = []
    link_names = []
    link_types = []
    link_customdata = []
    
    for link_name, info in edge_info.items():
        link_x.append(info['mid_pos'][0])
        link_y.append(info['mid_pos'][1])
        link_names.append(link_name)
        link_types.append(info['type'])
        # Store name and type as customdata for click detection
        link_customdata.append([link_name, info['type'], 'link'])
    
    fig.add_trace(go.Scatter(
        x=link_x,
        y=link_y,
        mode='markers+text',
        marker=dict(
            size=20,
            color=['red' if name in selected_links else 'white' for name in link_names],
            line=dict(width=2, color='black'),
            symbol='square'
        ),
        text=link_names,
        textfont=dict(size=8, color='black'),
        # CRITICAL: Store name, type, and category in customdata
        customdata=link_customdata,
        hovertemplate="<b>%{customdata[0]}</b><br>Type: %{customdata[1]}<br>Click to select<extra></extra>",
        name='Links (clickable)',
        showlegend=False
    ))
    
    # Add nodes with proper customdata
    node_x = []
    node_y = []
    node_names = []
    node_colors = []
    node_customdata = []
    
    for node_name, info in node_info.items():
        node_x.append(info['pos'][0])
        node_y.append(info['pos'][1])
        node_names.append(node_name)
        # Store name, type, and category as customdata for click detection
        node_customdata.append([node_name, info['type'], 'node'])
        
        if node_name in selected_nodes:
            color = 'red'
        elif info['type'] == 'Junction':
            color = 'lightblue'
        elif info['type'] == 'Tank':
            color = 'darkblue'
        elif info['type'] == 'Reservoir':
            color = 'darkgreen'
        else:
            color = 'gray'
        
        node_colors.append(color)
    
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(size=25, color=node_colors, line=dict(width=2, color='black')),
        text=node_names,
        textposition='middle center',
        textfont=dict(size=8, color='black'),
        # CRITICAL: Store name, type, and category in customdata
        customdata=node_customdata,
        hovertemplate="<b>%{customdata[0]}</b><br>Type: %{customdata[1]}<br>Click to select<extra></extra>",
        name='Nodes (clickable)',
        showlegend=False
    ))
    
    fig.update_layout(
        title="Interactive Water Network - Click on elements to select them",
        showlegend=True,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=600
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
    """Display event configuration interface for selected element."""
    st.subheader(f"Available Events for {element_name}")
    
    # Get available events based on element type
    if element_category == 'node':
        available_events = NODE_EVENTS.get(element_type, {})
    else:
        available_events = LINK_EVENTS.get(element_type, {})
    
    if not available_events:
        st.info(f"No events available for {element_type}")
        return None
    
    # Event selection
    event_names = list(available_events.keys())
    selected_event = st.selectbox("Select Event Type:", event_names, key=f"event_{element_name}")
    
    if selected_event:
        event_config = available_events[selected_event]
        params = event_config['params']
        defaults = event_config['defaults']
        
        st.write(f"**Event:** {selected_event}")
        
        # Parameter input
        param_values = {}
        for i, param in enumerate(params):
            default_val = defaults[i] if i < len(defaults) else None
            
            if param in ['leak_area', 'diameter', 'base_demand', 'fire_flow_demand', 'head', 'speed']:
                param_values[param] = st.number_input(
                    f"{param.replace('_', ' ').title()}:",
                    value=float(default_val) if default_val is not None else 0.0,
                    step=0.01,
                    key=f"{param}_{element_name}"
                )
            elif param in ['leak_discharge_coefficient']:
                param_values[param] = st.slider(
                    f"{param.replace('_', ' ').title()}:",
                    min_value=0.1, max_value=1.0, value=float(default_val) if default_val is not None else 0.75,
                    step=0.05,
                    key=f"{param}_{element_name}"
                )
            elif param in ['fire_start', 'fire_end']:
                param_values[param] = st.number_input(
                    f"{param.replace('_', ' ').title()} (seconds):",
                    value=int(default_val) if default_val is not None else 0,
                    step=60,
                    key=f"{param}_{element_name}"
                )
            else:
                param_values[param] = st.text_input(
                    f"{param.replace('_', ' ').title()}:",
                    value=str(default_val) if default_val is not None else "",
                    key=f"{param}_{element_name}"
                )
        
        # Add event button
        if st.button(f"Add {selected_event} to {element_name}", key=f"add_event_{element_name}"):
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
st.set_page_config(layout="wide", page_title="Interactive Network Event Simulator")
st.title("🔧 Interactive Water Network Event Simulator")

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

# Load network
if st.session_state.wn is None:
    with st.spinner("Loading network model..."):
        st.session_state.wn = load_network_model(INP_FILE)
        if st.session_state.wn:
            st.session_state.sim = MWNTRInteractiveSimulator(st.session_state.wn)
            st.success(f"Network '{INP_FILE}' loaded successfully!")
        else:
            st.stop()

wn = st.session_state.wn
sim = st.session_state.sim

# Main layout
st.subheader("🗺️ Interactive Network Map")
st.info("🖱️ **Click directly on nodes (circles) or links (squares) in the network map below to select them**")

# Create and display network plot (full width)
fig = create_network_plot(wn, st.session_state.selected_nodes, st.session_state.selected_links)

# Display plot with click handling
chart_data = st.plotly_chart(
    fig, 
    use_container_width=True, 
    key="network_plot_interaction",  # Unique key for event handling
    on_select="rerun",               # Rerun the app on selection
    selection_mode="points"          # Enable point selection
)

# Process click events after plot display
if chart_data and chart_data.get('selection') and chart_data['selection']['points']:
    # Get the first clicked point
    selected_point = chart_data['selection']['points'][0]
    
    # Debug information
    st.info(f"🔍 Debug: Detected click event. Point data: {selected_point}")
    
    # Get customdata [name, type, category]
    if selected_point.get('customdata'):
        custom_data = selected_point['customdata']
        element_name = custom_data[0]
        element_type = custom_data[1]
        element_category = custom_data[2]  # 'node' or 'link'
        
        st.success(f"🎯 Clicked on {element_category}: **{element_name}** ({element_type})")
        
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
            
            st.rerun()  # Rerun to update the display
    else:
        st.warning("⚠️ No customdata found in clicked point")
else:
    # Show current selection status
    if st.session_state.current_element:
        st.info(f"📌 Currently selected: **{st.session_state.current_element['name']}** ({st.session_state.current_element['type']})")

# Alternative selection method using selectboxes as backup
st.markdown("### 🎯 Alternative Selection (if clicking doesn't work)")
select_col1, select_col2 = st.columns(2)

with select_col1:
    st.markdown("**Select Node:**")
    node_options = ["None"] + list(wn.node_name_list)
    # Set default index based on current selection
    current_node_index = 0
    if st.session_state.current_element and st.session_state.current_element['category'] == 'node':
        try:
            current_node_index = node_options.index(st.session_state.current_element['name'])
        except ValueError:
            current_node_index = 0
    
    selected_node = st.selectbox(
        "Choose a node to configure events:",
        node_options,
        index=current_node_index,
        key="node_selector"
    )
    
    if selected_node != "None" and (not st.session_state.current_element or st.session_state.current_element.get('name') != selected_node):
        st.session_state.current_element = {
            'name': selected_node,
            'category': 'node',
            'type': wn.get_node(selected_node).node_type
        }
        st.session_state.selected_nodes = [selected_node]
        st.session_state.selected_links = []

with select_col2:
    st.markdown("**Select Link:**")
    link_options = ["None"] + list(wn.link_name_list)
    # Set default index based on current selection
    current_link_index = 0
    if st.session_state.current_element and st.session_state.current_element['category'] == 'link':
        try:
            current_link_index = link_options.index(st.session_state.current_element['name'])
        except ValueError:
            current_link_index = 0
    
    selected_link = st.selectbox(
        "Choose a link to configure events:",
        link_options,
        index=current_link_index,
        key="link_selector"
    )
    
    if selected_link != "None" and (not st.session_state.current_element or st.session_state.current_element.get('name') != selected_link):
        st.session_state.current_element = {
            'name': selected_link,
            'category': 'link',
            'type': wn.get_link(selected_link).link_type
        }
        st.session_state.selected_links = [selected_link]
        st.session_state.selected_nodes = []

# Clear selection button
if st.button("🔄 Clear All Selections"):
    st.session_state.selected_nodes = []
    st.session_state.selected_links = []
    st.session_state.current_element = None
    st.rerun()

st.markdown("---")

# Display selected element properties and events ONLY when something is selected
if st.session_state.current_element:
    element = st.session_state.current_element
    
    st.markdown("### 🎯 Selected Element Configuration")
    st.success(f"**{element['name']}** ({element['type']}) - Ready to configure events!")
    
    # Create columns for properties and events
    prop_col, event_col = st.columns([1, 1])
    
    with prop_col:
        st.markdown("#### 📋 Element Properties")
        display_element_properties(wn, element['name'], element['type'])
    
    with event_col:
        st.markdown("#### ⚡ Configure Events")
        new_event = display_event_interface(element['name'], element['type'], element['category'])
        if new_event:
            st.session_state.scheduled_events.append(new_event)
            st.success(f"✅ Event added: {new_event['event_type']} for {new_event['element_name']}")
            st.rerun()
else:
    st.info("📝 **Select a node or link above to view its properties and configure events**")

# Simulation Control
st.subheader("Simulation Control")

col3, col4, col5 = st.columns(3)

with col3:
    if st.button("🚀 Initialize Simulation", disabled=sim.initialized_simulation):
        try:
            sim.init_simulation(global_timestep=HYDRAULIC_TIMESTEP_SECONDS, duration=SIMULATION_DURATION_SECONDS)
            st.session_state.current_sim_time = 0
            st.success("Simulation initialized!")
        except Exception as e:
            st.error(f"Initialization failed: {e}")

with col4:
    if st.button("⏯️ Step Forward", disabled=not sim.initialized_simulation or sim.is_terminated()):
        # Apply any scheduled events for current time
        events_to_apply = [e for e in st.session_state.scheduled_events if e['scheduled_time'] <= st.session_state.current_sim_time]
        
        for event in events_to_apply:
            success, message = apply_event_to_simulator(sim, wn, event)
            if success:
                st.session_state.applied_events.append(event)
                st.success(message)
            else:
                st.error(message)
            st.session_state.scheduled_events.remove(event)
        
        # Run simulation step
        success, sim_time, message = run_simulation_step(sim, wn)
        st.session_state.current_sim_time = sim_time
        
        if success:
            # Collect data
            st.session_state.simulation_data['time'].append(sim_time)
            
            # Sample some pressure data
            for node_name in list(wn.node_name_list)[:5]:  # First 5 nodes
                node = wn.get_node(node_name)
                if node_name not in st.session_state.simulation_data['pressures']:
                    st.session_state.simulation_data['pressures'][node_name] = []
                pressure = getattr(node, 'pressure', 0) if hasattr(node, 'pressure') else 0
                st.session_state.simulation_data['pressures'][node_name].append(pressure)
            
            # Collect flow data for pipes
            for link_name in list(wn.link_name_list)[:5]:  # First 5 links
                link = wn.get_link(link_name)
                if link_name not in st.session_state.simulation_data['flows']:
                    st.session_state.simulation_data['flows'][link_name] = []
                flow = getattr(link, 'flow', 0) if hasattr(link, 'flow') else 0
                st.session_state.simulation_data['flows'][link_name].append(flow)
            
            st.success(f"⏰ Time: {datetime.timedelta(seconds=int(sim_time))} - {message}")
        else:
            st.warning(message)

with col5:
    if st.button("🔄 Reset Simulation"):
        st.session_state.sim = MWNTRInteractiveSimulator(wn)
        st.session_state.current_sim_time = 0
        st.session_state.scheduled_events = []
        st.session_state.applied_events = []
        st.session_state.simulation_data = {'time': [], 'pressures': {}, 'flows': {}}
        st.success("Simulation reset!")
        st.rerun()

# Display current simulation status
if sim.initialized_simulation:
    st.info(f"🕐 Current Simulation Time: {datetime.timedelta(seconds=int(st.session_state.current_sim_time))}")

# Events Panel
st.subheader("📅 Event Timeline")

col6, col7 = st.columns(2)

with col6:
    st.write("**Scheduled Events:**")
    if st.session_state.scheduled_events:
        for i, event in enumerate(st.session_state.scheduled_events):
            st.write(f"- {event['event_type']} → {event['element_name']} (t={event['scheduled_time']}s)")
    else:
        st.write("No scheduled events")

with col7:
    st.write("**Applied Events:**")
    if st.session_state.applied_events:
        for event in st.session_state.applied_events[-5:]:  # Show last 5
            st.write(f"- ✅ {event['event_type']} → {event['element_name']}")
    else:
        st.write("No events applied yet")

# Results Visualization
if len(st.session_state.simulation_data['time']) > 1:
    st.subheader("📊 Simulation Results")
    
    # Create two columns for side-by-side plots
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Create pressure plot
        pressure_data = st.session_state.simulation_data['pressures']
        if pressure_data:
            df_pressure = pd.DataFrame({
                'Time': st.session_state.simulation_data['time'],
                **pressure_data
            })
            
            fig_pressure = px.line(df_pressure, x='Time', y=df_pressure.columns[1:], 
                                  title="🔵 Node Pressures Over Time",
                                  labels={'value': 'Pressure (m)', 'Time': 'Time (s)'})
            fig_pressure.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_pressure, use_container_width=True)
    
    with chart_col2:
        # Create flowrate plot
        flow_data = st.session_state.simulation_data['flows']
        if flow_data:
            df_flow = pd.DataFrame({
                'Time': st.session_state.simulation_data['time'],
                **flow_data
            })
            
            fig_flow = px.line(df_flow, x='Time', y=df_flow.columns[1:], 
                              title="🔗 Link Flowrates Over Time",
                              labels={'value': 'Flow (m³/s)', 'Time': 'Time (s)'})
            fig_flow.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_flow, use_container_width=True)
    
    # Current values display
    if st.session_state.current_sim_time > 0:
        st.subheader("📈 Current Values")
        
        current_col1, current_col2 = st.columns(2)
        
        with current_col1:
            st.write("**Current Node Pressures:**")
            for node_name in list(wn.node_name_list)[:5]:
                node = wn.get_node(node_name)
                pressure = getattr(node, 'pressure', 0) if hasattr(node, 'pressure') else 0
                st.metric(f"{node_name}", f"{pressure:.2f} m", delta=None)
        
        with current_col2:
            st.write("**Current Link Flows:**")
            for link_name in list(wn.link_name_list)[:5]:
                link = wn.get_link(link_name)
                flow = getattr(link, 'flow', 0) if hasattr(link, 'flow') else 0
                st.metric(f"{link_name}", f"{flow:.4f} m³/s", delta=None)

# Footer
st.markdown("---")
st.caption(f"Interactive Network Simulator | Network: {INP_FILE} | Duration: {SIMULATION_DURATION_SECONDS//3600}h") 