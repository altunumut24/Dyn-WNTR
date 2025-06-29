#!/usr/bin/env python3
"""
Test case for Batch Simulator functionality
Tests the batch simulation with generated events and debug output
"""

import json
import os
import sys
import traceback
from pathlib import Path

# Add the modules directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

def test_batch_simulator_with_events():
    """Test batch simulator with generated events JSON file."""
    print("🧪 Testing Batch Simulator with Generated Events")
    print("=" * 60)
    
    try:
        # Step 1: Check for generated events file
        print("\n📁 Step 1: Looking for generated events file...")
        event_files = [f for f in os.listdir('.') if f.startswith('generated_events_NET_4_')]
        
        if not event_files:
            print("❌ No generated events file found!")
            print("   Available files:", [f for f in os.listdir('.') if f.endswith('.json')])
            
            # Try to use sample file instead
            if os.path.exists('sample_batch_events.json'):
                print("📄 Using sample_batch_events.json instead...")
                event_file = 'sample_batch_events.json'
            else:
                print("❌ No sample events file either!")
                return False
        else:
            event_file = event_files[0]
            print(f"✅ Found events file: {event_file}")
        
        # Step 2: Load events from JSON
        print(f"\n📋 Step 2: Loading events from {event_file}...")
        try:
            with open(event_file, 'r') as f:
                event_data = json.load(f)
            
            if isinstance(event_data, dict) and 'events' in event_data:
                events = event_data['events']
                metadata = event_data.get('metadata', {})
            elif isinstance(event_data, list):
                events = event_data
                metadata = {}
            else:
                events = [event_data] if isinstance(event_data, dict) else []
                metadata = {}
            
            print(f"✅ Loaded {len(events)} events")
            print(f"   Metadata: {metadata}")
            
            # Debug: Show first few events
            for i, event in enumerate(events[:3]):
                print(f"   Event {i+1}: {event.get('event_type', 'unknown')} at {event.get('time', 0)}s on {event.get('element_name', 'unknown')}")
                
        except Exception as e:
            print(f"❌ Error loading events: {e}")
            return False
        
        # Step 3: Load WNTR network
        print("\n🌊 Step 3: Loading WNTR network...")
        try:
            import mwntr
            from mwntr.sim.interactive_network_simulator import MWNTRInteractiveSimulator
            
            # Load network
            network_file = 'NET_4.inp'
            print(f"   Loading network from {network_file}...")
            wn = mwntr.network.WaterNetworkModel(network_file)
            print(f"✅ Network loaded: {len(wn.nodes)} nodes, {len(wn.links)} links")
            
        except Exception as e:
            print(f"❌ Error loading network: {e}")
            traceback.print_exc()
            return False
        
        # Step 4: Initialize simulator
        print("\n⚙️ Step 4: Initializing batch simulator...")
        try:
            sim = MWNTRInteractiveSimulator(wn)
            
            # Initialize simulation with 1 hour timestep
            timestep_seconds = 3600  # 1 hour
            duration_seconds = 28800  # 8 hours
            
            print(f"   Timestep: {timestep_seconds}s ({timestep_seconds/3600}h)")
            print(f"   Duration: {duration_seconds}s ({duration_seconds/3600}h)")
            
            sim.init_simulation(
                global_timestep=timestep_seconds,
                duration=duration_seconds
            )
            
            print("✅ Simulator initialized successfully")
            print(f"   Initialized: {sim.initialized_simulation}")
            print(f"   Terminated: {sim.is_terminated()}")
            
        except Exception as e:
            print(f"❌ Error initializing simulator: {e}")
            traceback.print_exc()
            return False
        
        # Step 5: Test event application
        print("\n🎯 Step 5: Testing event application...")
        try:
            from modules.batch_simulator_functions import apply_event_to_batch_simulator
            
            applied_count = 0
            for i, event in enumerate(events[:5]):  # Test first 5 events
                print(f"\n   Testing event {i+1}: {event.get('event_type')} on {event.get('element_name')}")
                
                try:
                    success, message = apply_event_to_batch_simulator(sim, wn, event)
                    if success:
                        applied_count += 1
                        print(f"   ✅ Event applied successfully: {message}")
                    else:
                        print(f"   ⚠️ Event failed: {message}")
                        
                except Exception as e:
                    print(f"   ❌ Event error: {e}")
            
            print(f"\n✅ Applied {applied_count}/{min(5, len(events))} test events")
            
        except Exception as e:
            print(f"❌ Error testing events: {e}")
            traceback.print_exc()
            return False
        
        # Step 6: Run simulation steps
        print("\n▶️ Step 6: Running simulation steps...")
        try:
            step_count = 0
            max_steps = 5  # Test 5 steps
            
            while not sim.is_terminated() and step_count < max_steps:
                current_time = sim.get_sim_time()
                print(f"\n   Step {step_count + 1}: Time = {current_time}s ({current_time/3600:.1f}h)")
                
                # Apply any events that should happen at current time
                current_events = [e for e in events if e['time'] <= current_time and not e.get('applied', False)]
                if current_events:
                    print(f"   📅 {len(current_events)} events due at this time")
                    
                    for event in current_events:
                        try:
                            success, message = apply_event_to_batch_simulator(sim, wn, event)
                            if success:
                                event['applied'] = True
                                print(f"      ✅ Applied: {event['event_type']} on {event['element_name']}")
                            else:
                                print(f"      ⚠️ Failed: {message}")
                        except Exception as e:
                            print(f"      ❌ Error: {e}")
                
                # Run simulation step
                try:
                    sim.step_sim()
                    new_time = sim.get_sim_time()
                    print(f"   ✅ Step completed: {current_time}s → {new_time}s")
                    step_count += 1
                    
                except Exception as e:
                    print(f"   ❌ Step error: {e}")
                    break
            
            print(f"\n✅ Completed {step_count} simulation steps")
            
        except Exception as e:
            print(f"❌ Error running simulation: {e}")
            traceback.print_exc()
            return False
        
        # Step 7: Check results
        print("\n📊 Step 7: Checking simulation results...")
        try:
            if hasattr(sim, 'results') and sim.results:
                results = sim.results
                print("✅ Simulation results available:")
                
                # Check node results
                if hasattr(results, 'node') and not results.node.empty:
                    print(f"   📍 Node data: {results.node.shape[0]} timesteps, {results.node.shape[1]} nodes")
                    print(f"   📍 Sample nodes: {list(results.node.columns)[:5]}")
                else:
                    print("   ⚠️ No node data available")
                
                # Check link results
                if hasattr(results, 'link') and not results.link.empty:
                    print(f"   🔗 Link data: {results.link.shape[0]} timesteps, {results.link.shape[1]} links")
                    print(f"   🔗 Sample links: {list(results.link.columns)[:5]}")
                else:
                    print("   ⚠️ No link data available")
                    
            else:
                print("⚠️ No simulation results available")
                
        except Exception as e:
            print(f"❌ Error checking results: {e}")
            traceback.print_exc()
        
        print("\n🎉 Batch simulator test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Batch simulator test failed: {e}")
        traceback.print_exc()
        return False

def test_dash_batch_simulator():
    """Test the Dash batch simulator functionality."""
    print("\n🌐 Testing Dash Batch Simulator Integration")
    print("=" * 60)
    
    try:
        # Import dash app modules
        from dash_network_app import global_state, load_network_model, initialize_simulation_data
        from modules.batch_simulator_functions import apply_event_to_batch_simulator
        from mwntr.sim.interactive_network_simulator import MWNTRInteractiveSimulator
        
        print("✅ Dash app modules imported successfully")
        
        # Test network loading
        print("\n🌊 Testing network loading...")
        wn = load_network_model('NET_4.inp')
        print(f"✅ Network loaded: {len(wn.nodes)} nodes, {len(wn.links)} links")
        
        # Test simulator creation
        print("\n⚙️ Testing simulator creation...")
        sim = MWNTRInteractiveSimulator(wn)
        sim.init_simulation(global_timestep=3600, duration=28800)
        print("✅ Simulator initialized")
        
        # Test global state
        print("\n🗃️ Testing global state...")
        global_state['batch_wn'] = wn
        global_state['batch_sim'] = sim
        global_state['batch_simulation_data'] = initialize_simulation_data()
        print("✅ Global state configured")
        
        # Test data collection
        print("\n📊 Testing data collection...")
        from dash_network_app import collect_simulation_data
        
        sim_data = initialize_simulation_data()
        monitored_nodes = list(wn.node_name_list)[:5]
        monitored_links = list(wn.link_name_list)[:5]
        
        collect_simulation_data(wn, monitored_nodes, monitored_links, sim_data)
        print(f"✅ Data collected for {len(monitored_nodes)} nodes, {len(monitored_links)} links")
        
        print("\n🎉 Dash integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Dash integration test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Running Batch Simulator Test Suite")
    print("=" * 60)
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Basic batch simulator functionality
    if test_batch_simulator_with_events():
        success_count += 1
    
    # Test 2: Dash integration
    if test_dash_batch_simulator():
        success_count += 1
    
    # Final results
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Batch simulator is working correctly.")
        sys.exit(0)
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
        sys.exit(1) 