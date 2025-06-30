# 🌊 Interactive Water Network Event Simulator

A powerful web-based tool for simulating water network events and analyzing their impacts on hydraulic performance. Built with Streamlit and WNTR (Water Network Tool for Resilience), this simulator provides both interactive and batch simulation capabilities for water distribution system analysis.

![Water Network Simulator](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![WNTR](https://img.shields.io/badge/WNTR-1.2+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🚀 Features

### 🎮 Interactive Mode
- **Real-time Network Visualization**: Interactive maps with clickable nodes and pipes
- **Event Configuration**: Point-and-click event creation with parameter customization
- **Live Simulation**: Step-by-step simulation with immediate visual feedback
- **Dynamic Color Coding**: Pressure and flow visualization with professional color schemes
- **Monitoring Charts**: Real-time time-series plots for selected elements
- **Event Timeline**: Visual timeline of scheduled and applied events
- **Network File Selection**: Choose the default NET_4.inp network or upload a custom EPANET INP file for simulation

### 📋 Batch Simulation Mode
- **JSON Event Loading**: Upload predefined event scenarios from JSON files
- **Automated Execution**: Hands-off simulation with automatic event application
- **Event Timeline Visualization**: Interactive timeline showing event progression
- **Batch Processing**: Run multiple scenarios for comparison and analysis
- **Export Capabilities**: Save results and event sequences for documentation

### 🔧 Event Types Supported
- **Node Events**: Leaks, demand changes, fire fighting scenarios, tank operations
- **Link Events**: Pipe closures, pump operations, valve controls, diameter changes
- **Advanced Events**: Custom parameters for realistic failure scenarios

## 📋 Requirements

- Python 3.8 or higher
- WNTR (Water Network Tool for Resilience)
- Streamlit
- Plotly
- Pandas
- NumPy

## 🛠️ Installation

### Option 1: Quick Start (Recommended)
```bash
# Clone the repository
git clone https://github.com/your-username/interactive-water-network-simulator.git
cd interactive-water-network-simulator

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run interactive_network_app.py
```

### Option 2: Virtual Environment
```bash
# Clone and navigate
git clone https://github.com/your-username/interactive-water-network-simulator.git
cd interactive-water-network-simulator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run interactive_network_app.py
```

### Option 3: Development Setup
```bash
# Clone repository
git clone https://github.com/your-username/interactive-water-network-simulator.git
cd interactive-water-network-simulator

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Run with debug mode
streamlit run interactive_network_app.py
```

## 🎯 Quick Start Guide

### 1. Launch the Application
```bash
streamlit run interactive_network_app.py
```
The application will open in your default web browser at `http://localhost:8501`.

### 2. Interactive Mode
1. **Select Network File**: Use the Network Configuration panel to choose the default NET_4.inp or upload a custom EPANET INP file.
2. **Explore the Network**: Click on nodes and pipes to see their properties
3. **Schedule Events**: Select an element and configure an event (leak, closure, etc.)
4. **Initialize Simulation**: Set duration and time step parameters
5. **Run Simulation**: Use step-by-step execution to see event impacts
6. **Monitor Results**: View real-time charts and network visualization

### 3. Batch Mode
1. **Upload Events**: Use a JSON file with predefined event sequences
2. **Review Timeline**: See when events are scheduled to occur
3. **Run Batch**: Automatic execution with progress tracking
4. **Analyze Results**: Compare different scenarios and outcomes

## 📁 Project Structure

```
interactive-water-network-simulator/
├── interactive_network_app.py      # Main Streamlit application
├── modules/                        # Core functionality modules
│   ├── __init__.py                # Module initialization
│   ├── config.py                  # Configuration and constants
│   ├── simulation.py              # WNTR simulation logic
│   ├── visualization.py           # Network plotting and charts
│   ├── ui_components.py           # Streamlit UI components
│   ├── event_generator.py         # Automatic event generation
│   └── batch_simulator_functions.py # Batch simulation capabilities
├── debug_helpers.py               # Development debugging tools
├── requirements.txt               # Python dependencies
├── NET_4.inp                     # Sample EPANET network file
└── README.md                     # This file
```

## 🔧 Configuration

### Network Files
- Place your EPANET `.inp` files in the project root directory
- Update `INP_FILE` in `modules/config.py` to use your network
- Supported formats: EPANET 2.0+ INP files

### Simulation Settings
Edit `modules/config.py` to customize:
- Default simulation duration and time steps
- Event definitions and parameters
- Color schemes and visualization settings
- UI layout and component defaults

### Event Configuration
Events are defined in `modules/config.py`:
```python
NODE_EVENTS = {
    'Junction': {
        'start_leak': {
            'params': ['leak_area', 'leak_discharge_coefficient'],
            'defaults': [0.01, 0.75]
        }
    }
}
```

## 📊 Usage Examples

### Creating Events Interactively
1. Click on a network element (node or pipe)
2. Select event type from the dropdown
3. Configure parameters (area, flow rate, etc.)
4. Set timing (when to apply the event)
5. Click "Schedule Event"

### Batch Event Format (JSON)
```json
{
    "events": [
        {
            "time": 3600,
            "element_name": "PIPE1",
            "element_type": "Pipe",
            "element_category": "link",
            "event_type": "close_pipe",
            "parameters": {}
        },
        {
            "time": 7200,
            "element_name": "JUNCTION5",
            "element_type": "Junction",
            "element_category": "node",
            "event_type": "start_leak",
            "parameters": {
                "leak_area": 0.02,
                "leak_discharge_coefficient": 0.8
            }
        }
    ],
    "metadata": {
        "description": "Pipe failure and subsequent leak scenario",
        "author": "Water Engineer",
        "scenario_type": "emergency_response"
    }
}
```

## 🎨 Key Components

### Core Modules

#### `simulation.py`
- WNTR integration and network loading
- Event application and execution
- Hydraulic simulation management
- Results collection and organization

#### `visualization.py`
- Interactive network maps with Plotly
- Pressure and flow color coding
- Time-series monitoring charts
- Professional visualization themes

#### `ui_components.py`
- Reusable Streamlit interface elements
- Form handling and user input validation
- Status displays and progress indicators
- Event configuration interfaces

#### `config.py`
- Central configuration management
- Event type definitions
- Color schemes and styling
- Default parameter settings

## 🐛 Debugging and Development

### Debug Mode
Enable debug mode in the sidebar to see:
- Session state variables
- Network model status
- Event processing details
- Performance metrics

### Development Tips
- Use `debug_helpers.py` for troubleshooting
- Monitor browser console for JavaScript errors
- Check Streamlit terminal output for Python errors
- Use browser developer tools for layout debugging

## 🤝 Contributing

1. **Fork the Repository**
   ```bash
   git fork https://github.com/your-username/interactive-water-network-simulator.git
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Add features or fix bugs
   - Update documentation
   - Add tests if applicable

4. **Commit Changes**
   ```bash
   git commit -m "feat: Add your feature description"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## 🔧 Troubleshooting

### Common Issues

**Application won't start:**
- Check Python version (3.8+ required)
- Verify all dependencies are installed
- Ensure INP file exists and is valid

**Network visualization problems:**
- Verify INP file has coordinate data
- Check browser console for JavaScript errors
- Try refreshing the page

**Simulation errors:**
- Validate INP file with EPANET
- Check event parameters are reasonable
- Review debug output for specific errors

**Performance issues:**
- Use smaller networks for testing
- Reduce monitoring elements count
- Increase simulation time steps

### Getting Help
- Check the [Issues](https://github.com/your-username/interactive-water-network-simulator/issues) page
- Review debug output for specific error messages
- Enable debug mode for detailed troubleshooting information

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **WNTR Team**: For the excellent Water Network Tool for Resilience
- **Streamlit**: For the amazing web app framework
- **Plotly**: For interactive visualization capabilities
- **EPANET**: For the foundational hydraulic simulation engine

## 📚 Additional Resources

- [WNTR Documentation](https://wntr.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [EPANET Manual](https://www.epa.gov/water-research/epanet)
- [Water Distribution System Analysis](https://www.springer.com/gp/book/9783030306755)

---

**Built with ❤️ for water engineers and researchers worldwide**

For questions, suggestions, or collaborations, please open an issue or contact the development team.

## 🚀 Deploying to Railway

1. Push the repository to GitHub.
2. Create a new Railway project and link it to the repo.
3. Railway detects the `Procfile` and Python environment automatically.
4. Set the service to be a `web` service (defaults from Procfile).
5. No extra environment variables are required—Railway injects `$PORT` which Dash/Gunicorn binds to.
6. Deploy and open the generated URL to access the app.
