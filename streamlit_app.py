import streamlit as st
import sys
import os
import time
import random
import traceback

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the simulation components
try:
    import mwntr.network
    from mwntr.sim.interactive_network_simulator import MWNTRInteractiveSimulator
    st.success("✅ Successfully imported MWNTR modules")
except ImportError as e:
    st.error(f"❌ Failed to import MWNTR modules: {e}")
    st.stop()

st.title("🌊 Dyn-WNTR Water Network Simulator")
st.markdown("### Cross-Platform Water Network Simulation")

# Display platform information
import platform
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Platform", platform.system())
with col2:
    st.metric("Architecture", platform.machine())
with col3:
    st.metric("Python", f"{sys.version_info.major}.{sys.version_info.minor}")

# Check extension status
st.markdown("### 🔧 Extension Status")

# Check evaluator extension
try:
    from mwntr.sim.aml import evaluator
    if hasattr(evaluator, '_use_fallback') and evaluator._use_fallback:
        st.warning("⚠️ Using fallback implementation for _evaluator (expected on non-Linux platforms)")
    else:
        st.success("✅ Using compiled _evaluator extension")
except Exception as e:
    st.error(f"❌ Error checking _evaluator: {e}")

# Check network isolation extension
try:
    from mwntr.sim.network_isolation import network_isolation
    st.success("✅ Network isolation module loaded")
except Exception as e:
    st.error(f"❌ Error checking network_isolation: {e}")

# Simulation controls
st.markdown("### 🎮 Simulation Controls")

if st.button("🚀 Run Quick Simulation Test"):
    with st.spinner("Running simulation..."):
        try:
            # Create a simple water network
            wn = mwntr.network.WaterNetworkModel()
            
            # Add basic components
            wn.add_reservoir('R1', base_head=100, coordinates=(0, 0))
            wn.add_junction('J1', base_demand=0.1, elevation=10, coordinates=(100, 0))
            wn.add_pipe('P1', 'R1', 'J1', length=100, diameter=0.3, roughness=100)
            
            # Initialize simulator
            sim = MWNTRInteractiveSimulator(wn)
            sim.init_simulation(duration=3600, global_timestep=300)  # 1 hour, 5-minute steps
            
            # Run a few simulation steps
            steps_completed = 0
            max_steps = 5
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while not sim.is_terminated() and steps_completed < max_steps:
                status_text.text(f"Step {steps_completed + 1}/{max_steps} - Time: {sim.get_sim_time()}s")
                
                # Simulate some random events
                if random.random() < 0.3:
                    sim.start_leak('J1', 0.05)
                    st.info(f"🔧 Leak started at step {steps_completed + 1}")
                
                sim.step_sim()
                steps_completed += 1
                progress_bar.progress(steps_completed / max_steps)
                time.sleep(0.5)  # Small delay for visual effect
            
            st.success(f"✅ Simulation completed successfully! Ran {steps_completed} steps.")
            
            # Display some results
            st.markdown("### 📊 Simulation Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Steps Completed", steps_completed)
            with col2:
                st.metric("Final Time", f"{sim.get_sim_time()}s")
                
        except Exception as e:
            st.error(f"❌ Simulation failed: {e}")
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())

# Information section
st.markdown("### ℹ️ About This Test")
st.info("""
This Streamlit app tests the cross-platform compatibility of the Dyn-WNTR water network simulator.

**Key Features Tested:**
- ✅ Module imports and initialization
- ✅ Extension fallback mechanisms  
- ✅ Basic simulation functionality
- ✅ Cross-platform compatibility

**Expected Behavior:**
- **Linux**: Uses compiled extensions for optimal performance
- **Other platforms**: Uses Python fallback implementations
- **All platforms**: Core simulation functionality works
""")

# Dependencies section
st.markdown("### 📦 Dependencies")
with st.expander("View Required Packages"):
    st.code("""
# Core dependencies
numpy
scipy
sympy
matplotlib
networkx
pandas

# Optional (for enhanced features)
plotly
geopandas
""")

# Footer
st.markdown("---")
st.markdown("🔬 **Dyn-WNTR** - Dynamic Water Network Tool for Resilience") 