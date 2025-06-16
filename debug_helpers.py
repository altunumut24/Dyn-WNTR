"""
Debug helpers for Streamlit app - Simple 3-step debugging approach
Only contains the functions actually used in the main app.
"""

import streamlit as st
from typing import Any

def debug_session_state():
    """Display all session state variables in an expandable section."""
    with st.expander("🐛 Debug: Session State", expanded=False):
        st.write("**All Session State Variables:**")
        for key, value in st.session_state.items():
            st.write(f"**{key}:** {type(value).__name__}")
            if isinstance(value, (str, int, float, bool)):
                st.write(f"  Value: `{value}`")
            elif isinstance(value, (list, dict)):
                st.write(f"  Length: {len(value)}")
                if len(value) < 10:  # Show small collections
                    st.json(value)
            else:
                st.write(f"  Object: {str(value)[:100]}...")

def debug_network_state(wn, sim):
    """Debug network and simulator state."""
    with st.expander("🌊 Debug: Network State", expanded=False):
        if wn is None:
            st.error("Network (wn) is None")
            return
        
        st.write("**Network Info:**")
        st.write(f"  Nodes: {len(wn.node_name_list)}")
        st.write(f"  Links: {len(wn.link_name_list)}")
        
        if sim is None:
            st.error("Simulator (sim) is None")
            return
            
        st.write("**Simulator Info:**")
        st.write(f"  Initialized: {getattr(sim, 'initialized_simulation', False)}")
        st.write(f"  Current Time: {getattr(sim, 'current_time', 'Unknown')}")

def debug_event_processing(events: list, current_time: int):
    """Debug event processing logic."""
    with st.expander("⚡ Debug: Event Processing", expanded=False):
        st.write(f"**Current Simulation Time:** {current_time} seconds")
        st.write(f"**Total Events:** {len(events)}")
        
        # Events due now
        due_events = [e for e in events if e.get('scheduled_time', e.get('time', 0)) <= current_time]
        st.write(f"**Events Due Now:** {len(due_events)}")
        
        if due_events:
            for event in due_events:
                st.write(f"  - {event.get('event_type', 'Unknown')} on {event.get('element_name', 'Unknown')}")
        
        # Future events
        future_events = [e for e in events if e.get('scheduled_time', e.get('time', 0)) > current_time]
        st.write(f"**Future Events:** {len(future_events)}") 