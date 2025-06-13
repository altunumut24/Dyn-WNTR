import random
import sys
import time
import mwntr # type: ignore
from typing import List, Dict, Tuple, cast

from mwntr.sim.interactive_network_simulator import MWNTRInteractiveSimulator # type: ignore
from mwntr.network import WaterNetworkModel # type: ignore

# These functions are from the original main.py, kept for structural similarity
# but not directly used by the main() function if NET_4.inp is loaded.
def create_water_network_model() -> WaterNetworkModel:
    # 1. Create a new water network model
    wn: WaterNetworkModel = mwntr.network.WaterNetworkModel()

    wn.options.hydraulic.demand_model = 'PDD'

    pattern_house1: List[float] = [1.0]*8 + [5.0]*8 + [1.0]*8  
    pattern_house2: List[float] = [1.0]*8 + [1.0]*8 + [5.0]*8  
    pattern_house3: List[float] = [5.0]*8 + [1.0]*8 + [1.0]*8  
    slow_pattern_house: List[float] = [1.0]*8 + [2.0]*12 + [1.0]*4    
    pump_speed_pattern: List[float] = [1.0]*24
    
    wn.add_pattern('slow_pattern_house', slow_pattern_house)
    wn.add_pattern('house1_pattern', pattern_house1)
    wn.add_pattern('house2_pattern', pattern_house2)
    wn.add_pattern('house3_pattern', pattern_house3)
    wn.add_pattern('pump_speed_pattern', pump_speed_pattern)

    wn.add_curve('Pump1_Curve', 'HEAD' , [(0, 50), (5, 45), (10, 40)])
    wn.add_curve('Pump2_Curve', 'HEAD' , [(0, 90), (5, 85), (10, 80)])

    wn.add_tank('R1', elevation=50, init_level=100, max_level=5000, min_level=0.0, coordinates=(-50, 50))

    wn.add_junction('J0', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(0, 100))
    wn.add_junction('J1', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(50, 100))
    wn.add_junction('J2', base_demand=0.0, elevation=100.0, demand_pattern=None, coordinates=(100, 100))
    wn.add_junction('J3', base_demand=0.0, elevation=100.0, demand_pattern=None, coordinates=(100, 50))
    wn.add_junction('J4', base_demand=0.0, elevation=100.0, demand_pattern=None, coordinates=(100, 0))
    wn.add_junction('J5', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(50, 0))
    wn.add_junction('J6', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(0, 0))
    wn.add_junction('J7', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(0, 50))
    wn.add_junction('J8', base_demand=0.0, elevation=10.0, demand_pattern=None, coordinates=(50, 50))

    wn.add_pipe('P_R1_J7', 'R1', 'J7', length=50, diameter=0.6, roughness=100, minor_loss=0)
    
    wn.add_pipe('PR0', 'J0', 'J1', length=50, diameter=0.6, roughness=100 , minor_loss=0)
    wn.add_pump('P1',  'J1', 'J2', initial_status='OPEN', pump_type='POWER', speed=1.0) # Changed speed to float
    wn.add_pipe('PR2', 'J2', 'J3', length=50, diameter=0.6, roughness=100, minor_loss=0)
    wn.add_pipe('PR3', 'J3', 'J4', length=50, diameter=0.6, roughness=100, minor_loss=0)
    wn.add_pipe('PR4', 'J5', 'J4', length=50, diameter=0.6, roughness=100, minor_loss=0)
    wn.add_pipe('PR5', 'J6', 'J5', length=50, diameter=0.6, roughness=100, minor_loss=0)
    wn.add_pipe('PR6', 'J7', 'J6', length=50, diameter=0.6, roughness=100, minor_loss=0)
    wn.add_pipe('PR7', 'J7', 'J0', length=50, diameter=0.6, roughness=100, minor_loss=0)
    wn.add_pipe('PR8', 'J7', 'J8', length=50, diameter=0.6, roughness=100, minor_loss=0)
    wn.add_pipe('PR9', 'J8', 'J3', length=50, diameter=0.6, roughness=100, minor_loss=0)

    wn.add_junction('H1', base_demand=0.5, elevation=10.0, demand_pattern=None, coordinates=(120, 100))
    wn.add_junction('H2', base_demand=0.5, elevation=10.0, demand_pattern=None, coordinates=(120, 50))
    wn.add_junction('H3', base_demand=0.5, elevation=10.0, demand_pattern=None, coordinates=(120, 0))

    wn.add_pipe('PH1', 'J2', 'H1', length=20, diameter=0.6, roughness=100, minor_loss=0)
    wn.add_pipe('PH3', 'J4', 'H3', length=20, diameter=0.6, roughness=100, minor_loss=0)
    wn.add_pipe('PH2', 'J3', 'H2', length=20, diameter=0.6, roughness=100, minor_loss=0)

    return wn

def create_new_water_model() -> WaterNetworkModel:
    wn: WaterNetworkModel = mwntr.network.WaterNetworkModel()
    
    pattern_house1: List[float] = [1.0] + [2.0] + [3.0] + [4.0] + [5.0] + [6.0] + [1.0] + [2.0] + [3.0] + [4.0] + [5.0] + [6.0] + [1.0] + [2.0] + [3.0] + [4.0] + [5.0] + [6.0] + [1.0] + [2.0] + [3.0] + [4.0] + [5.0] + [6.0]  
    pattern_house2: List[float] = [0.5]*7 + [1.5]*4 + [0.5]*6 + [1.5]*4 + [0.5]*3  
    pattern_house3: List[float] = [2.5]*8 + [0.0]*12 + [2.5]*4    
    slow_pattern_house: List[float] = [1.0]*8 + [2.0]*12 + [1.0]*4    
    pump_speed_pattern: List[float] = [1.0]*24

    wn.add_pattern('slow_pattern_house', slow_pattern_house)
    wn.add_pattern('house1_pattern', pattern_house1)
    wn.add_pattern('house2_pattern', pattern_house2)
    wn.add_pattern('house3_pattern', pattern_house3)
    wn.add_pattern('pump_speed_pattern', pump_speed_pattern)

    wn.options.hydraulic.demand_model = 'PDD'
    wn.add_reservoir('R1', base_head=100, coordinates=(0, 50))
    wn.add_tank('T1', elevation=30, init_level=5, min_level=2, max_level=10, diameter=10, coordinates=(100, 50))

    junction_data: Dict[str, Dict[str, float | Tuple[float, float]]] = {
        'J1': {'demand': 0.05, 'elevation': 10, 'coordinates': (50, 40)},
        'J2': {'demand': 0.04, 'elevation': 12, 'coordinates': (75, 30)},
        'J3': {'demand': 0.03, 'elevation': 15, 'coordinates': (100, 20)},
        'J4': {'demand': 0.06, 'elevation': 8, 'coordinates': (120, 40)},
    }
    for j, data in junction_data.items():
        wn.add_junction(j, base_demand=cast(float, data['demand']), elevation=cast(float, data['elevation']), coordinates=cast(Tuple[float,float], data['coordinates']))

    wn.add_pipe('P1', 'R1', 'J1', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('P3', 'J2', 'J3', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('P5', 'J4', 'T1', length=50, diameter=0.3, roughness=100)
    wn.add_pipe('P6', 'J2', 'T1', length=70, diameter=0.3, roughness=100)

    wn.add_curve('Pump1_Curve', 'HEAD' , [(0, 50), (5, 45), (10, 40)])
    wn.add_curve('Pump2_Curve', 'HEAD' , [(0, 90), (5, 85), (10, 80)])
    wn.add_pump('Pump1', 'J1', 'J2', pump_type='HEAD', pump_parameter='Pump1_Curve', speed=1.0)
    wn.add_valve('V1', 'J3', 'J4', valve_type='PRV', diameter=0.2, initial_setting=30.0) # Added .0 to initial_setting
    return wn

def main() -> None:
    one_day_in_seconds: int = 86400
    global_timestep: int = 5 # seconds

    # Load network model from .inp file
    wn: WaterNetworkModel = mwntr.network.WaterNetworkModel('NET_4.inp')
    
    # Expand patterns to match simulation duration and timestep
    wn.add_pattern('house1_pattern', MWNTRInteractiveSimulator.expand_pattern_to_simulation_duration([1,5,1], global_timestep, simulation_duration=one_day_in_seconds))
    wn.add_pattern('ptn_1', MWNTRInteractiveSimulator.expand_pattern_to_simulation_duration([1,3,5,3,1], global_timestep, simulation_duration=one_day_in_seconds))
        
    sim: MWNTRInteractiveSimulator = MWNTRInteractiveSimulator(wn)
    sim.init_simulation(duration=one_day_in_seconds, global_timestep=global_timestep)
    
    start_time_exec: float = time.time()

    # Variables for random events
    has_active_leak: List[str] = []
    has_active_demand: List[str] = []
    closed_pipe: List[str] = []

    all_nodes_to_monitor: List[str] = cast(List[str], wn.node_name_list)
    all_links_to_monitor: List[str] = cast(List[str], wn.link_name_list)
    junction_list_for_events: List[str] = cast(List[str], wn.junction_name_list)


    # Initialization for monitoring statistical changes
    previous_pressures: Dict[str, float] = {}
    previous_flowrates: Dict[str, float] = {}
    first_step_processed: bool = False
    
    # Simulation run counter
    sim_run_count: int = 0 # Renamed from i to be more descriptive
    
    # Get node and link indices for quick lookup
    node_name_to_index: Dict[str, int] = {name: i for i, name in enumerate(all_nodes_to_monitor)}
    link_name_to_index: Dict[str, int] = {name: i for i, name in enumerate(all_links_to_monitor)}


    try:
        while not sim.is_terminated():
            current_time_seconds: float = sim.get_sim_time()
            h: int = int(current_time_seconds // 3600)
            m: int = int((current_time_seconds % 3600) // 60)
            s: int = int(current_time_seconds % 60)
            current_time_str: str = f"{h:02d}:{m:02d}:{s:02d}"

            # Randomly introduce events (leaks, demand changes, pipe closures)
            r: float = random.random()
            if r < 0.05: # Probability of any event happening
                r2: float = random.random()
                if r2 < 0.3: # Leak event
                    if (len(has_active_leak) == 0 or random.random() < 0.5) and junction_list_for_events:    
                        node_name: str = random.choice(junction_list_for_events)
                        sim.start_leak(node_name, 0.1) # Example leak area
                        has_active_leak.append(node_name)
                        print(f"INFO [{current_time_str}]: Leak started on {node_name}")
                    elif has_active_leak:
                        node_name = random.choice(has_active_leak)
                        sim.stop_leak(node_name)
                        has_active_leak.remove(node_name)
                        print(f"INFO [{current_time_str}]: Leak stopped on {node_name}")
                elif r2 < 0.6: # Demand change event
                    if (len(has_active_demand) == 0 or random.random() < 0.5) and junction_list_for_events:    
                        node_name = random.choice(junction_list_for_events)
                        sim.change_demand(node_name, 1.0, name='ptn_1') # Example new base demand & pattern. Made 1.0
                        has_active_demand.append(node_name)
                        print(f"INFO [{current_time_str}]: Demand added on {node_name} using ptn_1")
                    elif has_active_demand:
                        node_name = random.choice(has_active_demand)
                        sim.change_demand(node_name) # Reverts to original base_demand and pattern
                        has_active_demand.remove(node_name)
                        print(f"INFO [{current_time_str}]: Demand removed on {node_name}")
                else: # Pipe closure/opening event
                    if (len(closed_pipe) == 0 or random.random() < 0.5) and all_links_to_monitor:    
                        link_name: str = random.choice(all_links_to_monitor) 
                        link_obj = wn.get_link(link_name) # Get link object to check its type
                        if hasattr(link_obj, 'link_type') and link_obj.link_type == 'Pipe':
                             sim.close_pipe(link_name)
                             closed_pipe.append(link_name)
                             print(f"INFO [{current_time_str}]: Pipe closed {link_name}")
                        # else:
                        #     print(f"DEBUG [{current_time_str}]: Skipped closing {link_name}, not a Pipe (type: {getattr(link_obj, 'link_type', 'N/A')}).")
                    elif closed_pipe:
                        link_name = random.choice(closed_pipe)
                        sim.open_pipe(link_name) # Assumes only pipes were added to closed_pipe list
                        closed_pipe.remove(link_name)
                        print(f"INFO [{current_time_str}]: Pipe opened {link_name}")

            sim.step_sim() # Advance simulation by one hydraulic timestep

            # ---- Monitoring logic ----
            # We will now attempt to get pressures and flowrates directly from the wn object,
            # assuming the simulator updates it after each step.

            print(f"\\n--- Monitoring at Sim Time: {current_time_str} ({current_time_seconds:.2f}s) ---")

            # Monitor Pressures
            current_pressures: Dict[str, float] = {}
            total_pressure_change_abs: float = 0.0
            num_pressure_values_changed: int = 0
            node_data_successfully_fetched: bool = False

            for node_name in all_nodes_to_monitor:
                try:
                    node = wn.get_node(node_name)
                    if node is not None and hasattr(node, 'pressure'):
                        pressure_val = node.pressure # Assuming this returns a float or can be cast
                        if pressure_val is not None:
                            pressure: float = float(pressure_val)
                            current_pressures[node_name] = pressure
                            node_data_successfully_fetched = True # Mark that we got at least one value

                            if first_step_processed and node_name in previous_pressures:
                                change: float = pressure - previous_pressures[node_name]
                                if abs(change) > 1e-5: # Threshold for significant change
                                    print(f"  Node '{node_name}': Pressure changed by {change:+.4f} (now: {pressure:.4f}, prev: {previous_pressures[node_name]:.4f})")
                                    total_pressure_change_abs += abs(change)
                                    num_pressure_values_changed +=1
                            # elif not first_step_processed:
                            # print(f"  Node '{node_name}': Initial Pressure = {pressure:.4f}")
                        else:
                            if not first_step_processed: # Only print warnings repeatedly if still in initial phase
                                print(f"  Node '{node_name}': pressure attribute is None.")
                    else:
                        if not first_step_processed:
                             print(f"  Node '{node_name}': Node object is None or has no 'pressure' attribute.")
                except AttributeError:
                    if not first_step_processed:
                        print(f"  Node '{node_name}': AttributeError accessing pressure (possibly not available for this node type or at this sim stage).")
                except Exception as e:
                    print(f"  Node '{node_name}': Error processing pressure: {e}")
            
            if not node_data_successfully_fetched and not first_step_processed:
                print("  Could not fetch any node pressure data in this step via wn.get_node().pressure.")
            
            previous_pressures = current_pressures.copy()

            # Monitor Flowrates
            current_flowrates: Dict[str, float] = {}
            total_flowrate_change_abs: float = 0.0
            num_flowrate_values_changed: int = 0
            link_data_successfully_fetched_this_step: bool = False # Reset each step

            # Persistent debug for the first link after initial data processing
            if first_step_processed and all_links_to_monitor: # Check if list is not empty
                debug_link_name = all_links_to_monitor[0]
                print(f"  FLOWRATE_DEBUG (Link: '{debug_link_name}', Time: {current_time_str}):")
                try:
                    link_obj_for_debug = wn.get_link(debug_link_name)
                    if link_obj_for_debug is not None:
                        print(f"    '{debug_link_name}' object: {type(link_obj_for_debug)}, Link Type: {getattr(link_obj_for_debug, 'link_type', 'N/A')}")
                        if hasattr(link_obj_for_debug, 'flow'):
                            flow_val_debug = link_obj_for_debug.flow
                            print(f"    '{debug_link_name}'.flow exists. Value: {flow_val_debug}, Type: {type(flow_val_debug)}")
                            if flow_val_debug is not None:
                                try:
                                    float_flow_debug = float(flow_val_debug)
                                    print(f"    '{debug_link_name}' .flow converted to float: {float_flow_debug}")
                                except ValueError as ve_debug:
                                    print(f"    '{debug_link_name}' .flow ('{flow_val_debug}') ValueError on float(): {ve_debug}")
                                except TypeError as te_debug:
                                    print(f"    '{debug_link_name}' .flow ('{flow_val_debug}') TypeError on float(): {te_debug}")
                            else:
                                print(f"    '{debug_link_name}' .flow value is None.")
                        else:
                            print(f"    '{debug_link_name}' has NO .flow attribute.")
                    else:
                        print(f"    wn.get_link('{debug_link_name}') returned None for FLOWRATE_DEBUG.")
                except Exception as e_debug_link:
                    print(f"    Error during FLOWRATE_DEBUG for link '{debug_link_name}': {e_debug_link}")
            
            # Process all links for statistics
            for link_name in all_links_to_monitor:
                try:
                    link = wn.get_link(link_name)
                    if link is not None and hasattr(link, 'flow'):
                        flowrate_val = link.flow
                        if flowrate_val is not None:
                            try:
                                flowrate: float = float(flowrate_val)
                                current_flowrates[link_name] = flowrate
                                link_data_successfully_fetched_this_step = True # Set if ANY link provides valid data

                                if first_step_processed and link_name in previous_flowrates:
                                    change: float = flowrate - previous_flowrates[link_name]
                                    if abs(change) > 1e-5: 
                                        print(f"  Link '{link_name}': Flowrate changed by {change:+.4f} (now: {flowrate:.4f}, prev: {previous_flowrates[link_name]:.4f})")
                                        total_flowrate_change_abs += abs(change)
                                        num_flowrate_values_changed +=1
                            except (ValueError, TypeError):
                                # This link's flowrate_val is not a valid float, do not process it for changes. 
                                # Only log this extensively if it's the debugged link and still during first_step_processed checks, 
                                # or if no data has ever been successfully processed.
                                if not first_step_processed or (first_step_processed and link_name == (all_links_to_monitor[0] if all_links_to_monitor else None) ):
                                     print(f"  Link '{link_name}': flow value '{flowrate_val}' (type: {type(flowrate_val)}) could not be converted to float for stats.")
                        # else: flowrate_val is None. Logging for this handled by FLOWRATE_DEBUG for the first link.
                    # else: link is None or has no flowrate attribute. Logging for this handled by FLOWRATE_DEBUG for the first link.
                except Exception as e:
                    print(f"  Link '{link_name}': Error processing flow for stats: {e}")
            
            previous_flowrates = current_flowrates.copy()

            if first_step_processed:
                if num_pressure_values_changed > 0:
                    avg_p_change: float = total_pressure_change_abs / num_pressure_values_changed
                    print(f"  Avg pressure change this step (for {num_pressure_values_changed} nodes with >1e-5 change): {avg_p_change:.4f}")
                elif node_data_successfully_fetched : 
                    print("  No significant pressure changes this step.")
                
                if num_flowrate_values_changed > 0:
                    avg_f_change: float = total_flowrate_change_abs / num_flowrate_values_changed
                    print(f"  Avg flowrate change this step (for {num_flowrate_values_changed} links with >1e-5 change): {avg_f_change:.4f}")
                elif link_data_successfully_fetched_this_step: 
                    print("  No significant flowrate changes this step (or all fetched values were identical to previous).")
                else: # No link data successfully fetched this step at all
                    print("  Could not fetch any valid link flowrate data this step to calculate averages/changes.")
            
            if (node_data_successfully_fetched or link_data_successfully_fetched_this_step) and not first_step_processed:
                print("\\n--- Initial values recorded (or attempted). Monitoring changes from next step. ---\\n") # Added newline for clarity
                first_step_processed = True
            elif not first_step_processed:
                 print("--- Waiting for initial data from wn.get_node/link properties to become available/populated. ---")
            # ---- Monitoring logic ends ----

        # End of simulation loop
        end_time_exec: float = time.time()
                
        if sim.get_sim_time() >= one_day_in_seconds - global_timestep: # Check if simulation completed fully
            sim.dump_results_to_csv() # Save results if completed
            sim_run_count += 1
            print(f"\\nSimulation {sim_run_count} completed successfully in {end_time_exec - start_time_exec:.2f} seconds.")
            results_path_str = str(sim.results_path) # Ensure it's a string
            print(f"Results saved to CSV files in the '{results_path_str}' directory.")

        else:
            print(f"\\nSimulation terminated at {current_time_str} ({sim.get_sim_time():.2f}s) before reaching the full duration of {one_day_in_seconds // 3600} hours.")
            print(f"Total execution time: {end_time_exec - start_time_exec:.2f} seconds.")

    except KeyboardInterrupt:
        print("\\nSimulation interrupted by user.")
        sys.exit()
    except Exception as e:
        print(f"\\nAn error occurred during the simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # sim.close() # if such a method exists in MWNTRInteractiveSimulator, it's good practice.
        print("Simulation ended.")


if __name__ == "__main__":
    main() 