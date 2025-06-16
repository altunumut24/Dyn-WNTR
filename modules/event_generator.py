import random
import json
import time
from typing import List, Dict, Any
import mwntr
from mwntr.sim.interactive_network_simulator import MWNTRInteractiveSimulator

def generate_random_events(wn: mwntr.network.WaterNetworkModel, duration_seconds: int, 
                          timestep_seconds: int, event_probability: float = 0.05) -> List[Dict[str, Any]]:
    """Generate random events for simulation and return them as a list."""
    
    events: List[Dict[str, Any]] = []
    
    # Initialize simulator to get network info
    sim = MWNTRInteractiveSimulator(wn)
    sim.init_simulation(duration=duration_seconds, global_timestep=timestep_seconds)
    
    # Track active states
    has_active_leak: List[str] = []
    has_active_demand: List[str] = []
    closed_pipes: List[str] = []
    closed_pumps: List[str] = []
    closed_valves: List[str] = []
    
    # Get available nodes and links
    node_list = wn.junction_name_list + wn.tank_name_list + wn.reservoir_name_list
    junction_list = wn.junction_name_list
    tank_list = wn.tank_name_list if hasattr(wn, 'tank_name_list') else []
    pipe_list = [name for name, link in wn.links() if link.link_type == 'Pipe']
    pump_list = [name for name, link in wn.links() if link.link_type == 'Pump']
    valve_list = [name for name, link in wn.links() if link.link_type == 'Valve']
    
    current_time = 0
    step_count = 0
    
    print(f"Generating events for {duration_seconds/3600:.1f} hours simulation...")
    print(f"Available elements: {len(junction_list)} junctions, {len(tank_list)} tanks, {len(pipe_list)} pipes, {len(pump_list)} pumps, {len(valve_list)} valves")
    
    while current_time < duration_seconds:
        # Random event generation
        if random.random() < event_probability:
            event_type_rand = random.random()
            
            if event_type_rand < 0.3:  # Leak events (30%)
                if len(has_active_leak) == 0 or random.random() < 0.5:
                    # Start new leak
                    target_nodes = junction_list + tank_list
                    if target_nodes:
                        node = random.choice(target_nodes)
                        leak_area = random.uniform(0.005, 0.05)  # 5mm² to 50mm²
                        leak_coeff = random.uniform(0.6, 0.85)
                        
                        event = {
                            'time': current_time,
                            'element_name': node,
                            'element_type': wn.get_node(node).node_type,
                            'element_category': 'node',
                            'event_type': 'start_leak',
                            'parameters': {
                                'leak_area': leak_area,
                                'leak_discharge_coefficient': leak_coeff
                            },
                            'description': f"Leak started on {node} (area: {leak_area:.4f}m², coeff: {leak_coeff:.2f})"
                        }
                        events.append(event)
                        has_active_leak.append(node)
                        print(f"T+{current_time//60:04.0f}min: {event['description']}")
                else:
                    # Stop existing leak
                    if has_active_leak:
                        node = random.choice(has_active_leak)
                        event = {
                            'time': current_time,
                            'element_name': node,
                            'element_type': wn.get_node(node).node_type,
                            'element_category': 'node',
                            'event_type': 'stop_leak',
                            'parameters': {},
                            'description': f"Leak stopped on {node}"
                        }
                        events.append(event)
                        has_active_leak.remove(node)
                        print(f"T+{current_time//60:04.0f}min: {event['description']}")
            
            elif event_type_rand < 0.5:  # Demand events (20%)
                if len(has_active_demand) == 0 or random.random() < 0.6:
                    # Add demand
                    if junction_list:
                        node = random.choice(junction_list)
                        demand_value = random.uniform(0.01, 0.5)  # 0.01 to 0.5 m³/s
                        
                        event = {
                            'time': current_time,
                            'element_name': node,
                            'element_type': 'Junction',
                            'element_category': 'node',
                            'event_type': 'add_demand',
                            'parameters': {
                                'base_demand': demand_value,
                                'pattern_name': None,
                                'category': 'event_generated'
                            },
                            'description': f"Additional demand added to {node} ({demand_value:.3f} m³/s)"
                        }
                        events.append(event)
                        has_active_demand.append(node)
                        print(f"T+{current_time//60:04.0f}min: {event['description']}")
                else:
                    # Remove demand
                    if has_active_demand:
                        node = random.choice(has_active_demand)
                        event = {
                            'time': current_time,
                            'element_name': node,
                            'element_type': 'Junction',
                            'element_category': 'node',
                            'event_type': 'remove_demand',
                            'parameters': {
                                'name': 'event_generated'
                            },
                            'description': f"Additional demand removed from {node}"
                        }
                        events.append(event)
                        has_active_demand.remove(node)
                        print(f"T+{current_time//60:04.0f}min: {event['description']}")
            
            elif event_type_rand < 0.7:  # Pipe events (20%)
                if len(closed_pipes) == 0 or random.random() < 0.5:
                    # Close pipe
                    if pipe_list:
                        pipe = random.choice([p for p in pipe_list if p not in closed_pipes])
                        if pipe:
                            event = {
                                'time': current_time,
                                'element_name': pipe,
                                'element_type': 'Pipe',
                                'element_category': 'link',
                                'event_type': 'close_pipe',
                                'parameters': {},
                                'description': f"Pipe {pipe} closed"
                            }
                            events.append(event)
                            closed_pipes.append(pipe)
                            print(f"T+{current_time//60:04.0f}min: {event['description']}")
                else:
                    # Open pipe
                    if closed_pipes:
                        pipe = random.choice(closed_pipes)
                        event = {
                            'time': current_time,
                            'element_name': pipe,
                            'element_type': 'Pipe',
                            'element_category': 'link',
                            'event_type': 'open_pipe',
                            'parameters': {},
                            'description': f"Pipe {pipe} opened"
                        }
                        events.append(event)
                        closed_pipes.remove(pipe)
                        print(f"T+{current_time//60:04.0f}min: {event['description']}")
            
            elif event_type_rand < 0.85:  # Pump events (15%)
                if pump_list:
                    if len(closed_pumps) == 0 or random.random() < 0.4:
                        # Close pump
                        pump = random.choice([p for p in pump_list if p not in closed_pumps])
                        if pump:
                            event = {
                                'time': current_time,
                                'element_name': pump,
                                'element_type': 'Pump',
                                'element_category': 'link',
                                'event_type': 'close_pump',
                                'parameters': {},
                                'description': f"Pump {pump} stopped"
                            }
                            events.append(event)
                            closed_pumps.append(pump)
                            print(f"T+{current_time//60:04.0f}min: {event['description']}")
                    else:
                        if closed_pumps:
                            # Open pump
                            pump = random.choice(closed_pumps)
                            event = {
                                'time': current_time,
                                'element_name': pump,
                                'element_type': 'Pump',
                                'element_category': 'link',
                                'event_type': 'open_pump',
                                'parameters': {},
                                'description': f"Pump {pump} restarted"
                            }
                            events.append(event)
                            closed_pumps.remove(pump)
                            print(f"T+{current_time//60:04.0f}min: {event['description']}")
                        else:
                            # Change pump speed
                            pump = random.choice(pump_list)
                            new_speed = random.uniform(0.5, 1.5)
                            event = {
                                'time': current_time,
                                'element_name': pump,
                                'element_type': 'Pump',
                                'element_category': 'link',
                                'event_type': 'set_pump_speed',
                                'parameters': {
                                    'speed': new_speed
                                },
                                'description': f"Pump {pump} speed changed to {new_speed:.2f}"
                            }
                            events.append(event)
                            print(f"T+{current_time//60:04.0f}min: {event['description']}")
            
            else:  # Tank/Valve events (15%)
                if tank_list and random.random() < 0.6:
                    # Tank head change
                    tank = random.choice(tank_list)
                    new_head = random.uniform(30, 80)
                    event = {
                        'time': current_time,
                        'element_name': tank,
                        'element_type': 'Tank',
                        'element_category': 'node',
                        'event_type': 'set_tank_head',
                        'parameters': {
                            'head': new_head
                        },
                        'description': f"Tank {tank} head set to {new_head:.1f}m"
                    }
                    events.append(event)
                    print(f"T+{current_time//60:04.0f}min: {event['description']}")
                
                elif valve_list:
                    # Valve operation
                    if len(closed_valves) == 0 or random.random() < 0.5:
                        # Close valve
                        valve = random.choice([v for v in valve_list if v not in closed_valves])
                        if valve:
                            event = {
                                'time': current_time,
                                'element_name': valve,
                                'element_type': 'Valve',
                                'element_category': 'link',
                                'event_type': 'close_valve',
                                'parameters': {},
                                'description': f"Valve {valve} closed"
                            }
                            events.append(event)
                            closed_valves.append(valve)
                            print(f"T+{current_time//60:04.0f}min: {event['description']}")
                    else:
                        # Open valve
                        if closed_valves:
                            valve = random.choice(closed_valves)
                            event = {
                                'time': current_time,
                                'element_name': valve,
                                'element_type': 'Valve',
                                'element_category': 'link',
                                'event_type': 'open_valve',
                                'parameters': {},
                                'description': f"Valve {valve} opened"
                            }
                            events.append(event)
                            closed_valves.remove(valve)
                            print(f"T+{current_time//60:04.0f}min: {event['description']}")
        
        # Advance time
        current_time += timestep_seconds
        step_count += 1
        
        # Progress indicator every 10% of simulation
        if step_count % (duration_seconds // (timestep_seconds * 10)) == 0:
            progress = (current_time / duration_seconds) * 100
            print(f"Progress: {progress:.0f}% - {len(events)} events generated so far")
    
    print(f"\nEvent generation completed! Total events: {len(events)}")
    return events

def save_events_to_json(events: List[Dict[str, Any]], filename: str):
    """Save events to JSON file with metadata."""
    
    event_data = {
        'metadata': {
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_events': len(events),
            'duration_seconds': max([e['time'] for e in events]) if events else 0,
            'event_types': list(set([e['event_type'] for e in events])),
            'elements_affected': list(set([e['element_name'] for e in events])),
            'description': 'Randomly generated water network simulation events'
        },
        'events': events
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(event_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvents saved to: {filename}")
    print(f"Metadata: {event_data['metadata']['total_events']} events over {event_data['metadata']['duration_seconds']/3600:.1f} hours")
    print(f"Event types: {', '.join(event_data['metadata']['event_types'])}")
    print(f"Elements affected: {len(event_data['metadata']['elements_affected'])} unique elements")

def load_network_model(inp_file: str = 'NET_4.inp') -> mwntr.network.WaterNetworkModel:
    """Load water network model from INP file."""
    wn = mwntr.network.WaterNetworkModel(inp_file)
    wn.options.hydraulic.demand_model = 'PDD'
    return wn

def main():
    """Main function to generate and save events."""
    
    # Configuration
    INP_FILE = 'NET_4.inp'
    SIMULATION_DURATION = 10 * 3600  # 2 hours
    TIMESTEP = 60  # 1 minute
    EVENT_PROBABILITY = 0.08  # 8% chance per step (higher than original for more events)
    OUTPUT_FILE = 'generated_events.json'
    
    print("🔧 Water Network Event Generator")
    print("=" * 50)
    print(f"Network file: {INP_FILE}")
    print(f"Simulation duration: {SIMULATION_DURATION/3600:.1f} hours")
    print(f"Time step: {TIMESTEP} seconds")
    print(f"Event probability: {EVENT_PROBABILITY*100:.1f}% per step")
    print(f"Output file: {OUTPUT_FILE}")
    print("=" * 50)
    
    try:
        # Load network
        print("\n📋 Loading network model...")
        wn = load_network_model(INP_FILE)
        print(f"✅ Network loaded: {len(wn.node_name_list)} nodes, {len(wn.link_name_list)} links")
        
        # Generate events
        print("\n🎲 Generating random events...")
        events = generate_random_events(
            wn, 
            SIMULATION_DURATION, 
            TIMESTEP, 
            EVENT_PROBABILITY
        )
        
        # Save to file
        print(f"\n💾 Saving events to {OUTPUT_FILE}...")
        save_events_to_json(events, OUTPUT_FILE)
        
        print("\n✅ Event generation completed successfully!")
        
        # Summary statistics
        print("\n📊 Event Summary:")
        event_types = {}
        for event in events:
            event_type = event['event_type']
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        for event_type, count in sorted(event_types.items()):
            print(f"  - {event_type}: {count} events")
            
    except FileNotFoundError:
        print(f"❌ Error: Network file '{INP_FILE}' not found!")
        print("Please ensure the INP file is in the current directory.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 