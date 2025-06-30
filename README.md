# 🌊 Dynamic Water Network Simulation Tool

This project is a web-based application for simulating and analyzing water distribution networks. It provides an interactive interface for running simulations, visualizing results, and evaluating the network's performance under various conditions.

## 🚀 Features

-   **Interactive Network Visualization**: Explore the water network topology with an interactive map that displays nodes, pipes, and other components.
-   **Dynamic Simulation**: Run hydraulic simulations and observe the network's behavior over time.
-   **Scenario Management**: Create and manage different simulation scenarios by defining events such as pipe breaks, demand changes, and pump failures.
-   **Results Analysis**: Visualize simulation results, including pressure, flow, and water quality, through dynamic charts and color-coded network maps.
-   **Batch Processing**: Automate the simulation of multiple scenarios using predefined event files in JSON format.
-   **Customizable**: Easily extend the simulator's functionality by modifying the core modules and integrating custom analysis tools.

## 🛠️ Installation

To run the application locally, you'll need to have Python 3.8 or higher installed.

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♀️ Running the Application

Once you've installed the dependencies, you can start the application by running the following command:

```bash
python dash_network_app.py
```

This will launch a local development server, and you can access the application by opening your web browser and navigating to `http://127.0.0.1:8050`.

## 📁 Project Structure

```
├── Dockerfile                  # Configuration for building a Docker image
├── NET_2.inp                   # EPANET input file for network model 2
├── NET_3.inp                   # EPANET input file for network model 3
├── NET_4.inp                   # EPANET input file for network model 4
├── Procfile                    # Heroku deployment configuration
├── README.md                   # This file
├── assets/                     # Static assets (CSS, JS)
├── dash_network_app.py         # Main Dash application file
├── fly.toml                    # Fly.io deployment configuration
├── generated_events1.json      # Predefined events for batch simulation
├── generated_events2.json      # Predefined events for batch simulation
├── main.py                     # Main script (if any)
├── modules/                    # Core application modules
│   ├── __init__.py
│   ├── batch_simulator_functions.py
│   ├── config.py
│   ├── dash_ui_components.py
│   ├── simulation.py
│   └── visualization.py
├── mwntr/                      # Modified WNTR library
├── requirements.txt            # Python dependencies
├── results/                    # Directory for storing simulation results
├── simple_event_generator.py   # Script for generating simple events
├── static/                     # Static files for the web interface
└── venv/                       # Virtual environment directory
```

## 📦 Dependencies

The main dependencies of this project are:

-   **Dash**: A Python framework for building analytical web applications.
-   **WNTR**: The Water Network Tool for Resilience, used for simulating water distribution networks.
-   **Plotly**: A graphing library for creating interactive charts and visualizations.
-   **Pandas**: A data analysis library for handling and manipulating data.

For a complete list of dependencies, please refer to the `requirements.txt` file.

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project, please feel free to fork the repository, make your changes, and submit a pull request.
