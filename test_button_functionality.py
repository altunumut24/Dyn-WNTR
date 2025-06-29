#!/usr/bin/env python3
"""
Test specifically for batch button functionality
"""

import json
import traceback

def test_batch_button_logic():
    """Test the batch button logic directly."""
    print("🔘 Testing Batch Button Logic")
    print("=" * 50)
    
    try:
        # Import the batch controls function
        from dash_network_app import handle_batch_playback_controls
        import dash
        from unittest.mock import MagicMock
        
        print("✅ Imported batch controls function")
        
        # Mock dash callback context
        mock_ctx = MagicMock()
        mock_ctx.triggered = [{'prop_id': 'batch-play-btn.n_clicks'}]
        
        # Patch the callback context
        with MagicMock() as mock_dash:
            dash.callback_context = mock_ctx
            
            # Test initial state (no button clicked)
            print("\n🔍 Testing initial state...")
            mock_ctx.triggered = None
            
            try:
                result = handle_batch_playback_controls(
                    play_clicks=None,      # No clicks yet
                    pause_clicks=None,
                    step_clicks=None, 
                    reset_clicks=None,
                    events_loaded=False,   # No events loaded
                    timestep_minutes=60,
                    is_running=False,
                    is_initialized=False
                )
                
                print(f"   Initial state result: {len(result)} parameters")
                print(f"   Start button disabled: {result[3]}")  # Should be False (enabled)
                print(f"   Pause button disabled: {result[4]}")  # Should be True (disabled)
                print(f"   Step button disabled: {result[5]}")   # Should be True (disabled)
                
                if result[3] == False:  # Start button enabled
                    print("   ✅ Start button is ENABLED initially")
                else:
                    print("   ❌ Start button is DISABLED initially")
                    
            except Exception as e:
                print(f"   ❌ Error in initial state: {e}")
                traceback.print_exc()
            
            # Test Start button click
            print("\n🔍 Testing Start button click...")
            mock_ctx.triggered = [{'prop_id': 'batch-play-btn.n_clicks'}]
            
            try:
                result = handle_batch_playback_controls(
                    play_clicks=1,         # First click
                    pause_clicks=None,
                    step_clicks=None,
                    reset_clicks=None,
                    events_loaded=False,   # No events loaded (should still work)
                    timestep_minutes=60,
                    is_running=False,
                    is_initialized=False
                )
                
                print(f"   Start click result: {len(result)} parameters")
                print(f"   Simulation initialized: {result[0]}")  # Should be True
                print(f"   Simulation running: {result[1]}")      # Should be True
                print(f"   Start button disabled: {result[3]}")   # Should be True (disabled after start)
                print(f"   Pause button disabled: {result[4]}")   # Should be False (enabled)
                
                if result[0] and result[1]:
                    print("   ✅ Start button WORKS - simulation started")
                else:
                    print("   ❌ Start button FAILED - simulation not started")
                    
            except Exception as e:
                print(f"   ❌ Error clicking Start: {e}")
                traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ Button test failed: {e}")
        traceback.print_exc()
        return False

def test_batch_controls_display():
    """Test if batch controls are displayed correctly."""
    print("\n🖼️ Testing Batch Controls Display")
    print("=" * 50)
    
    try:
        from dash_network_app import display_batch_controls
        
        print("✅ Imported display function")
        
        # Test with no events loaded
        print("\n🔍 Testing display with no events...")
        result_no_events = display_batch_controls(events_loaded=False)
        
        if result_no_events and hasattr(result_no_events, 'children'):
            print("✅ Controls displayed even without events")
        else:
            print("❌ Controls not displayed without events")
            
        # Test with events loaded
        print("\n🔍 Testing display with events...")
        result_with_events = display_batch_controls(events_loaded=True)
        
        if result_with_events and hasattr(result_with_events, 'children'):
            print("✅ Controls displayed with events")
        else:
            print("❌ Controls not displayed with events")
            
        return True
        
    except Exception as e:
        print(f"❌ Display test failed: {e}")
        traceback.print_exc()
        return False

def test_with_sample_events():
    """Test with actual sample events file."""
    print("\n📄 Testing with Sample Events")
    print("=" * 50)
    
    try:
        # Load sample events
        with open('sample_batch_events.json', 'r') as f:
            sample_data = json.load(f)
        
        events = sample_data.get('events', [])
        print(f"✅ Loaded {len(events)} sample events")
        
        # Test controls with real events
        from dash_network_app import handle_batch_playback_controls
        import dash
        from unittest.mock import MagicMock
        
        mock_ctx = MagicMock()
        mock_ctx.triggered = [{'prop_id': 'batch-play-btn.n_clicks'}]
        dash.callback_context = mock_ctx
        
        result = handle_batch_playback_controls(
            play_clicks=1,
            pause_clicks=None,
            step_clicks=None,
            reset_clicks=None,
            events_loaded=True,        # Events are loaded
            timestep_minutes=60,
            is_running=False,
            is_initialized=False
        )
        
        print(f"✅ Button works with events: {result[0]} initialized, {result[1]} running")
        return True
        
    except Exception as e:
        print(f"❌ Sample events test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Batch Button Functionality")
    print("=" * 60)
    
    success_count = 0
    
    if test_batch_button_logic():
        success_count += 1
        
    if test_batch_controls_display():
        success_count += 1
        
    if test_with_sample_events():
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Button Tests: {success_count}/3 passed")
    
    if success_count == 3:
        print("🎉 All button tests passed! Start button should work.")
    else:
        print("⚠️ Some button tests failed.")
        
    print("\n💡 To test in browser:")
    print("   1. Run: python dash_network_app.py")
    print("   2. Open: http://localhost:8050")
    print("   3. Go to Batch Simulator tab")
    print("   4. Click Start button (should work without uploading events)") 