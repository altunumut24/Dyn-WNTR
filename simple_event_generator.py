"""
Simple Event Generator for Water Network Simulation

This script runs the main.py simulation logic to generate realistic events
and saves them to JSON format for batch simulation.

Based on the existing main.py approach with random event generation
during simulation runtime.
"""

import random
import json
import time
import sys
from datetime import datetime
from modules.simulation import load_network_model
from mwntr.sim.interactive_network_simulator import MWNTRInteractiveSimulator


def generate_events_from_simulation(inp_file='NET_4.inp', duration_hours=8, timestep_minutes=1):
    """
    Generate events by running simulation logic similar to main.py
    
    Args:
        inp_file: Path to the .inp network file
        duration_hours: Simulation duration in hours  
        timestep_minutes: Simulation timestep in minutes
        
    Returns:
        dict: Events data with metadata ready for JSON export
    """
    
    print(f"🎲 Generating events for {inp_file}...")
    print(f"Duration: {duration_hours} hours, Timestep: {timestep_minutes} minutes")
    
    # Load network
    wn = load_network_model(inp_file)
    duration_seconds = duration_hours * 3600
    global_timestep = timestep_minutes * 60
    
    # Initialize simulator
    sim = MWNTRInteractiveSimulator(wn)
    sim.init_simulation(duration=duration_seconds, global_timestep=global_timestep)
    
    # Track active events (same as main.py)
    has_active_leak = []
    has_active_demand = []
    closed_pipe = []
    
    # Get network elements
    node_list = list(wn.junction_name_list)
    link_list = list(wn.link_name_list)
    
    events = []
    current_time = 0
    
    print("Running simulation and generating events...")
    
    try:
        while not sim.is_terminated():
            current_time = sim.get_sim_time()
            
            # Random event generation (similar to main.py logic)
            r = random.random()
            if r < 0.05:  # 5% chance per step
                r2 = random.random()
                
                if r2 < 0.3:  # 30% leak events
                    if len(has_active_leak) == 0 or random.random() < 0.5:    
                        # Start leak
                        node = random.choice(node_list)
                        leak_area = random.uniform(0.005, 0.05)
                        leak_coeff = random.uniform(0.6, 0.85)
                        
                        sim.start_leak(node, leak_area, leak_coeff)
                        has_active_leak.append(node)
                        
                        event = {
                            "time": int(current_time),
                            "element_name": node,
                            "element_type": "Junction",
                            "element_category": "node",
                            "event_type": "start_leak",
                            "parameters": {
                                "leak_area": round(leak_area, 6),
                                "leak_discharge_coefficient": round(leak_coeff, 3)
                            },
                            "description": f"Leak started on {node} (area: {leak_area:.4f}m², coeff: {leak_coeff:.2f})"
                        }
                        events.append(event)
                        print(f"T+{current_time//60:04.0f}min: Leak started on {node}")
                        
                    else:
                        # Stop leak
                        node = random.choice(has_active_leak)
                        sim.stop_leak(node)
                        has_active_leak.remove(node)
                        
                        event = {
                            "time": int(current_time),
                            "element_name": node,
                            "element_type": "Junction", 
                            "element_category": "node",
                            "event_type": "stop_leak",
                            "parameters": {},
                            "description": f"Leak stopped on {node}"
                        }
                        events.append(event)
                        print(f"T+{current_time//60:04.0f}min: Leak stopped on {node}")
                        
                elif r2 < 0.6:  # 30% demand events
                    if len(has_active_demand) == 0 or random.random() < 0.5:    
                        # Add demand
                        node = random.choice(node_list)
                        demand_value = random.uniform(0.01, 0.3)
                        
                        sim.add_demand(node, demand_value, category='auto_generated')
                        has_active_demand.append(node)
                        
                        event = {
                            "time": int(current_time),
                            "element_name": node,
                            "element_type": "Junction",
                            "element_category": "node", 
                            "event_type": "add_demand",
                            "parameters": {
                                "base_demand": round(demand_value, 4),
                                "pattern_name": None,
                                "category": "auto_generated"
                            },
                            "description": f"Additional demand added to {node} ({demand_value:.3f} m³/s)"
                        }
                        events.append(event)
                        print(f"T+{current_time//60:04.0f}min: Demand added to {node}")
                        
                    else:
                        # Remove demand
                        node = random.choice(has_active_demand)
                        sim.remove_demand(node, 'auto_generated')
                        has_active_demand.remove(node)
                        
                        event = {
                            "time": int(current_time),
                            "element_name": node,
                            "element_type": "Junction",
                            "element_category": "node",
                            "event_type": "remove_demand",
                            "parameters": {
                                "name": "auto_generated"
                            },
                            "description": f"Additional demand removed from {node}"
                        }
                        events.append(event)
                        print(f"T+{current_time//60:04.0f}min: Demand removed from {node}")
                        
                else:  # 40% pipe events
                    if len(closed_pipe) == 0 or random.random() < 0.5:    
                        # Close pipe
                        link = random.choice(link_list)
                        if link not in closed_pipe:
                            sim.close_pipe(link)
                            closed_pipe.append(link)
                            
                            event = {
                                "time": int(current_time),
                                "element_name": link,
                                "element_type": "Pipe",
                                "element_category": "link",
                                "event_type": "close_pipe", 
                                "parameters": {},
                                "description": f"Pipe {link} closed"
                            }
                            events.append(event)
                            print(f"T+{current_time//60:04.0f}min: Pipe closed {link}")
                            
                    else:
                        # Open pipe
                        link = random.choice(closed_pipe)
                        sim.open_pipe(link)
                        closed_pipe.remove(link)
                        
                        event = {
                            "time": int(current_time),
                            "element_name": link,
                            "element_type": "Pipe",
                            "element_category": "link",
                            "event_type": "open_pipe",
                            "parameters": {},
                            "description": f"Pipe {link} opened"
                        }
                        events.append(event)
                        print(f"T+{current_time//60:04.0f}min: Pipe opened {link}")

            # Step simulation
            sim.step_sim()
            
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    except Exception as e:
        print(f"Simulation ended: {e}")
    
    # Create events data structure
    events_data = {
        "metadata": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "network_file": inp_file,
            "simulation_method": "main.py_logic",
            "total_events": len(events),
            "duration_seconds": duration_seconds,
            "duration_hours": duration_hours,
            "timestep_seconds": global_timestep,
            "event_types": list(set(e["event_type"] for e in events)),
            "elements_affected": list(set(e["element_name"] for e in events)),
            "network_info": {
                "nodes": len(wn.node_name_list),
                "links": len(wn.link_name_list),
                "junctions": len(wn.junction_name_list),
                "pipes": len([l for l in wn.link_name_list if wn.get_link(l).link_type == 'Pipe'])
            },
            "description": f"Events generated using main.py simulation logic for {inp_file}"
        },
        "events": events
    }
    
    print(f"✅ Generated {len(events)} events")
    return events_data


def save_generated_events(inp_file='NET_4.inp', duration_hours=8, timestep_minutes=1, output_file=None):
    """
    Generate and save events to JSON file
    
    Args:
        inp_file: Network file to use
        duration_hours: Simulation duration
        timestep_minutes: Simulation timestep
        output_file: Output JSON file (auto-generated if None)
    """
    
    # Generate events
    events_data = generate_events_from_simulation(inp_file, duration_hours, timestep_minutes)
    
    # Create output filename if not provided
    if output_file is None:
        inp_name = inp_file.replace('.inp', '').replace('/', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"generated_events_{inp_name}_{timestamp}.json"
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(events_data, f, indent=2)
    
    print(f"\n💾 Events saved to: {output_file}")
    print(f"📊 Summary:")
    print(f"   • Network: {inp_file}")
    print(f"   • Events: {events_data['metadata']['total_events']}")
    print(f"   • Duration: {duration_hours} hours")
    print(f"   • Event types: {', '.join(events_data['metadata']['event_types'])}")
    print(f"   • Elements affected: {len(events_data['metadata']['elements_affected'])}")
    
    return output_file, events_data


def main():
    """Main function to generate events"""
    
    print("🎲 Simple Water Network Event Generator")
    print("=" * 50)
    
    # Configuration (same as main.py style)
    INP_FILE = 'NET_4.inp'
    DURATION_HOURS = 360000
    TIMESTEP_MINUTES = 60
    
    try:
        output_file, events_data = save_generated_events(
            inp_file=INP_FILE,
            duration_hours=DURATION_HOURS, 
            timestep_minutes=TIMESTEP_MINUTES
        )
        
        print(f"\n🎉 Success! Events ready for batch simulation.")
        print(f"📁 File: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 