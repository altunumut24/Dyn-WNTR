#!/usr/bin/env python3
"""
Test script for the Dash Water Network Simulator.

This script runs the Dash application and performs basic functionality tests.
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_import():
    """Test that all required modules can be imported."""
    try:
        print("Testing imports...")
        
        # Test Dash imports
        import dash
        from dash import dcc, html, Input, Output, State
        import dash_bootstrap_components as dbc
        print("✅ Dash components imported successfully")
        
        # Test our modules
        from modules.config import INP_FILE
        from modules.simulation import load_network_model, initialize_simulation_data
        from modules.visualization import create_network_plot
        from modules.dash_ui_components import create_simulation_status_display
        print("✅ Custom modules imported successfully")
        
        # Test WNTR imports
        from mwntr.sim.interactive_network_simulator import MWNTRInteractiveSimulator
        print("✅ WNTR modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_network_loading():
    """Test that network files can be loaded."""
    try:
        print("\nTesting network loading...")
        
        from modules.simulation import load_network_model
        
        # Test loading example network
        wn = load_network_model("NET_2.inp")
        if wn:
            print(f"✅ Network loaded: {len(wn.node_name_list)} nodes, {len(wn.link_name_list)} links")
            return True
        else:
            print("❌ Failed to load network")
            return False
            
    except Exception as e:
        print(f"❌ Network loading error: {e}")
        return False

def test_dash_app():
    """Test that the Dash app can be created."""
    try:
        print("\nTesting Dash app creation...")
        
        # Import the Dash app
        from dash_network_app import app
        
        if app:
            print("✅ Dash app created successfully")
            print(f"✅ App title: {app.title}")
            return True
        else:
            print("❌ Failed to create Dash app")
            return False
            
    except Exception as e:
        print(f"❌ Dash app creation error: {e}")
        return False

def run_tests():
    """Run all tests."""
    print("🧪 Running Dash Water Network Simulator Tests\n")
    
    tests = [
        ("Import Test", test_import),
        ("Network Loading Test", test_network_loading),
        ("Dash App Test", test_dash_app)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"🔍 {test_name}")
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}\n")
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Dash application is ready to run.")
        print("\nTo start the application, run:")
        print("python dash_network_app.py")
        print("\nThen open your browser to: http://localhost:8050")
        print("\nNote: Using app.run() method (newer Dash versions)")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 