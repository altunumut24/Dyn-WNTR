import streamlit as st
import pandas as pd
import time
import random
import datetime
from typing import List, Dict, Any, Generator, Tuple

# Assuming mwntr and its components are in the PYTHONPATH
# For a real deployment, ensure mwntr is properly installed in the environment
try:
    import mwntr  # type: ignore
    from mwntr.sim.interactive_network_simulator import MWNTRInteractiveSimulator  # type: ignore
    from mwntr.network import WaterNetworkModel, Node, Link, LinkStatus  # type: ignore
    from mwntr.network.elements import Pipe, Pump, Valve # type: ignore
    MWNTR_AVAILABLE = True
except ImportError:
    MWNTR_AVAILABLE = False
    # Mock classes for UI development if mwntr is not available
    class WaterNetworkModel:
        def __init__(self):
            self.node_name_list: List[str] = []
            self.link_name_list: List[str] = []
            self.pipe_name_list: List[str] = []
            self.pump_name_list: List[str] = []
            self.valve_name_list: List[str] = []
        def get_node(self, name: str): return _MockNode(name) # type: ignore
        def get_link(self, name: str): return _MockPipe(name) # type: ignore

    class _MockMWNTRInteractiveSimulator: # Renamed to avoid conflict
        def __init__(self, wn): self._wn = wn
        def get_node_name_list(self): return self._wn.node_name_list if self._wn else []
        def get_link_name_list(self): return self._wn.link_name_list if self._wn else []
        def step_sim(self, save_results=False): return True, 0 # Mock success
        def get_time(self): return str(datetime.timedelta(seconds=self._current_time_sec if hasattr(self, '_current_time_sec') else 0))
        def get_sim_time(self): return self._current_time_sec if hasattr(self, '_current_time_sec') else 0
        def initialize_sim(self, wn, patterns_path=None, curves_path=None, rules_path=None,
                           sim_type='PDD', hyd_step=5, qual_step=5, report_step=300,
                           report_start=0, sim_duration=24*3600, convergence_error=False,
                           results_path="results.csv"):
            self._current_time_sec = 0
            self._hyd_step = hyd_step
            pass
        def add_leak(self, wn, node_name, area, start_time, end_time): pass # Mock method
        def set_pipe_status(self, wn, pipe_name, status, start_time): pass # Mock method


    class _MockNode: # Renamed
        def __init__(self, name): self.name = name; self.pressure = random.uniform(10, 50); self.node_type = 'Junction'
    class _MockPipe: # Renamed
        def __init__(self, name): self.name = name; self.flow = random.uniform(1, 20); self.link_type = "Pipe"; self.status = "Open"
    class _MockPump: # Renamed
        def __init__(self, name): self.name = name; self.flow = random.uniform(50, 100); self.link_type = "Pump"; self.status = "Open"
    class _MockValve: # Renamed
        def __init__(self, name): self.name = name; self.flow = random.uniform(0, 5); self.link_type = "Valve"; self.status = "Open"

    class _MockLinkStatus: # Renamed
        Open = "OPEN"
        Closed = "CLOSED"
    
    if not MWNTR_AVAILABLE: # Assign mocks only if needed
        MWNTRInteractiveSimulator = _MockMWNTRInteractiveSimulator # type: ignore
        Node = _MockNode # type: ignore
        Pipe = _MockPipe # type: ignore
        Pump = _MockPump # type: ignore
        Valve = _MockValve # type: ignore
        LinkStatus = _MockLinkStatus # type: ignore

# --- Configuration & Constants ---
SIMULATION_DURATION_SECONDS = 1 * 3600  # Simulate for 1 hour for demo
HYDRAULIC_TIMESTEP_SECONDS = 5
REPORT_TIMESTEP_SECONDS = 5 # How often we update the plot / simulator steps by
INP_FILE = 'NET_4.inp'

# --- Helper Functions & Simulation Logic ---

def load_water_network_model(inp_file: str) -> WaterNetworkModel:
    """Loads the water network model from an INP file or creates a mock."""
    if not MWNTR_AVAILABLE:
        st.sidebar.warning("MWNTR library not available. Using a mock network model.")
        mock_wn = WaterNetworkModel()
        mock_wn.node_name_list = [f"J{i:02d}" for i in range(1, 11)] + [f"T{i}" for i in range(1,3)]
        mock_wn.pipe_name_list = [f"P{i:03d}" for i in range(1, 16)]
        mock_wn.pump_name_list = [f"PU1"]
        mock_wn.valve_name_list = [f"V1"]
        mock_wn.link_name_list = mock_wn.pipe_name_list + mock_wn.pump_name_list + mock_wn.valve_name_list
        # Ensure mock objects are instantiated for get_node/get_link
        _mock_nodes_internal = {name: _MockNode(name) for name in mock_wn.node_name_list}
        _mock_links_internal = {}
        for name in mock_wn.pipe_name_list: _mock_links_internal[name] = _MockPipe(name)
        for name in mock_wn.pump_name_list: _mock_links_internal[name] = _MockPump(name)
        for name in mock_wn.valve_name_list: _mock_links_internal[name] = _MockValve(name)

        def mock_get_node(name): return _mock_nodes_internal.get(name, _MockNode(name)) # type: ignore
        def mock_get_link(name): return _mock_links_internal.get(name, _MockPipe(name)) # type: ignore
        mock_wn.get_node = mock_get_node # type: ignore
        mock_wn.get_link = mock_get_link # type: ignore
        return mock_wn
    
    try:
        wn = mwntr.network.WaterNetworkModel(inp_file)
        wn.options.hydraulic.demand_model = 'PDD'
        wn.options.time.duration = SIMULATION_DURATION_SECONDS 
        wn.options.time.hydraulic_timestep = HYDRAULIC_TIMESTEP_SECONDS 
        wn.options.time.report_timestep = REPORT_TIMESTEP_SECONDS 
        return wn
    except FileNotFoundError:
        st.sidebar.error(f"Error: INP file '{inp_file}' not found. Place it in the same directory as the script or provide the full path.")
        return None
    except Exception as e:
        st.sidebar.error(f"Error loading INP file '{inp_file}': {e}")
        return None

def apply_random_controls(sim: MWNTRInteractiveSimulator, wn: WaterNetworkModel, current_time_seconds: float):
    """Applies random network changes during the simulation."""
    if not MWNTR_AVAILABLE or not hasattr(sim, 'add_leak'): # Check for actual sim methods for safety
        return

    # Example: Add a small leak
    if random.random() < 0.03: # 3% chance each control step period
        # Ensure nodes exist and are of type Junction if possible
        junction_nodes = [name for name in wn.node_name_list if hasattr(wn.get_node(name), 'node_type') and wn.get_node(name).node_type == 'Junction']
        if not junction_nodes: junction_nodes = wn.node_name_list # Fallback for simpler mocks

        if junction_nodes:
            node_to_leak = random.choice(junction_nodes)
            leak_area = random.uniform(0.001, 0.005) # Smaller leak area
            try:
                # Ensure times are integers if required by the API
                sim.add_leak(wn, node_to_leak, leak_area, 
                             start_time=int(current_time_seconds), 
                             end_time=int(current_time_seconds + random.randint(600, 1800)))
                st.session_state.log_messages.append(f"Leak added to {node_to_leak} at {datetime.timedelta(seconds=int(current_time_seconds))}")
            except Exception as e:
                st.session_state.log_messages.append(f"Leak add failed: {node_to_leak}, {e}")


    # Example: Change pipe status 
    if random.random() < 0.01 and wn.pipe_name_list: # 1% chance
        pipe_to_change = random.choice(wn.pipe_name_list)
        pipe_obj = wn.get_link(pipe_to_change)
        if pipe_obj and hasattr(pipe_obj, 'status'):
            current_status = pipe_obj.status
            new_status = LinkStatus.Closed if current_status != LinkStatus.Closed else LinkStatus.Open
            try:
                sim.set_pipe_status(wn, pipe_to_change, new_status, start_time=int(current_time_seconds))
                st.session_state.log_messages.append(f"Pipe {pipe_to_change} status to {new_status} at {datetime.timedelta(seconds=int(current_time_seconds))}")
            except Exception as e:
                st.session_state.log_messages.append(f"Pipe status change failed: {pipe_to_change}, {e}")

def run_simulation_step_by_step(sim: MWNTRInteractiveSimulator, wn: WaterNetworkModel) -> Generator[Dict[str, Any], None, None]:
    """Runs the simulation step-by-step, yielding data."""
    sim_control_interval = 60 * 5 # Apply random controls every 5 minutes of sim time
    next_control_time = 0
    
    if not MWNTR_AVAILABLE or not hasattr(sim, 'initialize_sim'): # Mock simulation
        st.sidebar.info("Using mock simulation data.")
        # Use session state for mock node/link names if available from mock WN
        mock_node_names = st.session_state.all_node_names
        mock_link_names = st.session_state.all_link_names

        current_mock_pressures = {name: random.uniform(20, 50) for name in mock_node_names}
        current_mock_flowrates = {name: random.uniform(5, 20) for name in mock_link_names}
        
        sim._current_time_sec = 0 # Initialize for mock simulator's get_time()
        for t_sec in range(0, SIMULATION_DURATION_SECONDS, REPORT_TIMESTEP_SECONDS):
            sim._current_time_sec = t_sec # Update mock sim time
            for name in mock_node_names:
                current_mock_pressures[name] = max(5, current_mock_pressures.get(name,30) + random.uniform(-1, 1))
            for name in mock_link_names:
                current_mock_flowrates[name] = max(0.1, current_mock_flowrates.get(name,10) + random.uniform(-0.5, 0.5))
            
            yield {
                'time': t_sec, 'time_str': get_time_str(sim),
                'pressures': current_mock_pressures.copy(),
                'flowrates': current_mock_flowrates.copy(),
                'converged': True
            }
            time.sleep(0.02) # Slow down mock for visibility
        return

    # Actual MWNTR simulation
    sim_time_seconds = 0
    try:
        # Use REPORT_TIMESTEP for hyd_step to get data at each plotted point
        sim.initialize_sim(wn, sim_duration=SIMULATION_DURATION_SECONDS, 
                           hyd_step=REPORT_TIMESTEP_SECONDS, 
                           report_step=REPORT_TIMESTEP_SECONDS) 
    except Exception as e:
        st.error(f"Error during simulation initialization: {e}")
        return

    while sim_time_seconds < SIMULATION_DURATION_SECONDS:
        try:
            if sim_time_seconds >= next_control_time:
                apply_random_controls(sim, wn, sim_time_seconds)
                next_control_time += sim_control_interval

            converged, _ = sim.step_sim(save_results=False)
            sim_time_seconds = int(sim.get_sim_time())
            current_time_str = get_time_str(sim)

            current_pressures: Dict[str, float] = {}
            for node_name in wn.node_name_list:
                node = wn.get_node(node_name)
                val = 0.0
                if node and hasattr(node, 'pressure') and node.pressure is not None:
                    try: val = float(node.pressure)
                    except (ValueError, TypeError): pass
                current_pressures[node_name] = val

            current_flowrates: Dict[str, float] = {}
            for link_name in wn.link_name_list:
                link = wn.get_link(link_name)
                val = 0.0
                if link and hasattr(link, 'flow') and link.flow is not None:
                    try: val = float(link.flow)
                    except (ValueError, TypeError): pass
                current_flowrates[link_name] = val
            
            yield {
                'time': sim_time_seconds, 'time_str': current_time_str,
                'pressures': current_pressures, 'flowrates': current_flowrates,
                'converged': converged
            }
            if not converged:
                st.warning(f"Simulation non-convergence at {current_time_str}.")
                break
        except Exception as e:
            st.error(f"Error during sim step at {sim_time_seconds}s: {e}")
            break

def get_time_str(sim):
    """Return formatted simulation time string, using simulator's internal methods if available."""
    if hasattr(sim, '_get_time'):
        try:
            return sim._get_time()
        except Exception:
            pass
    if hasattr(sim, 'get_time'):
        try:
            return sim.get_time()
        except Exception:
            pass
    if hasattr(sim, 'get_sim_time'):
        try:
            sec = sim.get_sim_time()
            return str(datetime.timedelta(seconds=sec))
        except Exception:
            pass
    return '00:00:00'

# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="WNTR Real-time Simulation")
st.title("Real-time Water Network Simulation Monitoring")

# Initialize session state
default_session_state = {
    'simulation_running': False, 'pressure_df_dict': {}, 'flowrate_df_dict': {},
    'selected_nodes': [], 'selected_links': [],
    'all_node_names': [], 'all_link_names': [], 'wn': None, 'log_messages': []
}
for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

if st.session_state.wn is None: # Load WN only once
    st.session_state.wn = load_water_network_model(INP_FILE)
    if st.session_state.wn:
        st.session_state.all_node_names = st.session_state.wn.node_name_list
        st.session_state.all_link_names = st.session_state.wn.link_name_list
        st.sidebar.success(f"Network '{INP_FILE}' loaded." if MWNTR_AVAILABLE else "Mock network loaded.")
    elif MWNTR_AVAILABLE : # Failed to load INP, but MWNTR is there
        st.sidebar.error(f"Failed to load network: {INP_FILE}. Check file and path.")


# Sidebar
st.sidebar.header("Controls & Selections")
if st.sidebar.button("Start Simulation", key="start_button", disabled=st.session_state.simulation_running or not st.session_state.wn):
    if not st.session_state.wn:
         st.sidebar.error("Cannot start: Water Network Model not loaded.")
    else:
        st.session_state.simulation_running = True
        st.session_state.log_messages = ["Simulation started..."]
        # Reset data DataFrames for selected items only
        st.session_state.pressure_df_dict = {
            node: pd.DataFrame(columns=['Time (s)', node]).set_index('Time (s)')
            for node in st.session_state.selected_nodes
        }
        st.session_state.flowrate_df_dict = {
            link: pd.DataFrame(columns=['Time (s)', link]).set_index('Time (s)')
            for link in st.session_state.selected_links
        }

if st.session_state.all_node_names:
    st.session_state.selected_nodes = st.sidebar.multiselect(
        "Nodes for Pressure Plot", st.session_state.all_node_names,
        default=st.session_state.selected_nodes or st.session_state.all_node_names[:min(3, len(st.session_state.all_node_names))]
    )
else:
    st.sidebar.caption("No nodes available for selection.")

if st.session_state.all_link_names:
    st.session_state.selected_links = st.sidebar.multiselect(
        "Links for Flowrate Plot", st.session_state.all_link_names,
        default=st.session_state.selected_links or st.session_state.all_link_names[:min(3, len(st.session_state.all_link_names))]
    )
else:
    st.sidebar.caption("No links available for selection.")

# Main area layout
status_placeholder = st.empty()
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.subheader("Node Pressures (m)")
    pressure_chart_placeholder = st.empty()
    latest_pressures_placeholder = st.empty() # New placeholder for latest pressures
with chart_col2:
    st.subheader("Link Flowrates (CMS)") # Assuming CMS, adjust if different
    flowrate_chart_placeholder = st.empty()
    latest_flowrates_placeholder = st.empty() # New placeholder for latest flowrates

# Simulation Loop
if st.session_state.simulation_running:
    current_wn = st.session_state.wn
    # Instantiate simulator (actual or mock)
    sim_instance = MWNTRInteractiveSimulator(current_wn) if MWNTR_AVAILABLE else _MockMWNTRInteractiveSimulator(current_wn)

    # Ensure DataFrame dicts exist for newly selected items
    for node in st.session_state.selected_nodes:
        if node not in st.session_state.pressure_df_dict:
            st.session_state.pressure_df_dict[node] = pd.DataFrame(columns=['Time (s)', node]).set_index('Time (s)')
    for link in st.session_state.selected_links:
        if link not in st.session_state.flowrate_df_dict:
            st.session_state.flowrate_df_dict[link] = pd.DataFrame(columns=['Time (s)', link]).set_index('Time (s)')
            
    simulation_completed_normally = False
    last_time_str = "00:00:00"

    for data_step in run_simulation_step_by_step(sim_instance, current_wn):
        sim_time_s = data_step['time']
        last_time_str = data_step['time_str']
        status_placeholder.text(f"Running Simulation... Time: {last_time_str} ({sim_time_s}s)")

        # Update Pressure Data & Chart
        if st.session_state.selected_nodes:
            temp_pressure_plot_df = pd.DataFrame()
            for node_name in st.session_state.selected_nodes:
                if node_name in data_step['pressures']:
                    pressure_val = data_step['pressures'][node_name]
                    # Append new data to the specific node's DataFrame
                    new_row = pd.DataFrame({node_name: [pressure_val]}, index=[sim_time_s])
                    st.session_state.pressure_df_dict[node_name] = pd.concat([st.session_state.pressure_df_dict[node_name], new_row])
                # For plotting, combine selected series
                if node_name in st.session_state.pressure_df_dict and not st.session_state.pressure_df_dict[node_name].empty:
                    temp_pressure_plot_df = pd.concat([temp_pressure_plot_df, st.session_state.pressure_df_dict[node_name]], axis=1)
            if not temp_pressure_plot_df.empty:
                pressure_chart_placeholder.line_chart(temp_pressure_plot_df)
            
            # Display latest pressure values
            if st.session_state.selected_nodes:
                pressure_md_list = ["**Latest Pressures:**"]
                for node_name in st.session_state.selected_nodes:
                    if node_name in data_step['pressures']:
                        pressure_val = data_step['pressures'][node_name]
                        pressure_md_list.append(f"- {node_name}: {pressure_val:.2f} m")
                if len(pressure_md_list) > 1:
                    latest_pressures_placeholder.markdown("\\n".join(pressure_md_list))
                else:
                    latest_pressures_placeholder.empty()

        # Update Flowrate Data & Chart
        if st.session_state.selected_links:
            temp_flowrate_plot_df = pd.DataFrame()
            for link_name in st.session_state.selected_links:
                if link_name in data_step['flowrates']:
                    flowrate_val = data_step['flowrates'][link_name]
                    new_row = pd.DataFrame({link_name: [flowrate_val]}, index=[sim_time_s])
                    st.session_state.flowrate_df_dict[link_name] = pd.concat([st.session_state.flowrate_df_dict[link_name], new_row])
                if link_name in st.session_state.flowrate_df_dict and not st.session_state.flowrate_df_dict[link_name].empty:
                    temp_flowrate_plot_df = pd.concat([temp_flowrate_plot_df, st.session_state.flowrate_df_dict[link_name]], axis=1)
            if not temp_flowrate_plot_df.empty:
                flowrate_chart_placeholder.line_chart(temp_flowrate_plot_df)

            # Display latest flowrate values
            if st.session_state.selected_links:
                flowrate_md_list = ["**Latest Flowrates:**"]
                for link_name in st.session_state.selected_links:
                    if link_name in data_step['flowrates']:
                        flowrate_val = data_step['flowrates'][link_name]
                        flowrate_md_list.append(f"- {link_name}: {flowrate_val:.3f} CMS")
                if len(flowrate_md_list) > 1:
                    latest_flowrates_placeholder.markdown("\\n".join(flowrate_md_list))
                else:
                    latest_flowrates_placeholder.empty()

        if not data_step['converged'] and (MWNTR_AVAILABLE and st.session_state.wn is not None):
            st.session_state.log_messages.append(f"HALTED: Non-convergence at {last_time_str}.")
            simulation_completed_normally = False
            break 
        simulation_completed_normally = True
    
    st.session_state.simulation_running = False
    if simulation_completed_normally:
        status_placeholder.success(f"Simulation finished at {last_time_str}.")
        st.session_state.log_messages.append(f"Simulation finished at {last_time_str}.")
    else:
        status_placeholder.warning(f"Simulation stopped or failed at {last_time_str}.")
        # Log message already added for non-convergence

# Log display area in sidebar
st.sidebar.markdown("--- Logs ---")
log_display_area = st.sidebar.empty()
if st.session_state.log_messages:
    log_display_area.text_area("Messages", "\n".join(st.session_state.log_messages), height=150, key="log_text_area")
else:
    log_display_area.caption("No simulation messages.")

st.markdown("---")
if not MWNTR_AVAILABLE:
    st.warning("MWNTR library is not installed. The app is running with MOCK data and MOCK simulation. Install mwntr for actual simulations.")
st.caption(f"INP File: '{INP_FILE}' (Ensure it's accessible). Sim Duration: {SIMULATION_DURATION_SECONDS // 3600}hr. Report Step: {REPORT_TIMESTEP_SECONDS}s.") 