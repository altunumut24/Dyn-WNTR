"""
Simple Monitoring System - Fresh Start
======================================

This is a clean, minimal implementation of interactive monitoring
for the water network simulator. No complex real-time systems,
just simple dropdowns that work reliably.

Key features:
- Simple dropdown for selecting nodes/links 
- Immediate chart updates when selections change
- Clear remove functionality 
- No complex button systems or callbacks
"""

import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional

def create_simple_monitoring_section(available_nodes: List[str], available_links: List[str]) -> html.Div:
    """Create a simple monitoring interface with just dropdowns"""
    
    return html.Div([
        # Simple title
        html.H4("📊 Simple Monitoring", className="text-primary mb-3"),
        
        dbc.Row([
            # Pressure monitoring
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🔵 Pressure Monitoring"),
                    dbc.CardBody([
                        html.Label("Select Nodes to Monitor:", className="fw-bold"),
                        dcc.Dropdown(
                            id="simple-pressure-nodes",
                            options=[{'label': f"🔵 {node}", 'value': node} for node in available_nodes],
                            multi=True,
                            placeholder="Select nodes...",
                            value=[]
                        ),
                        html.Div(id="pressure-status", className="mt-2")
                    ])
                ])
            ], width=6),
            
            # Flow monitoring  
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🔗 Flow Monitoring"),
                    dbc.CardBody([
                        html.Label("Select Links to Monitor:", className="fw-bold"),
                        dcc.Dropdown(
                            id="simple-flow-links",
                            options=[{'label': f"🔗 {link}", 'value': link} for link in available_links],
                            multi=True,
                            placeholder="Select links...",
                            value=[]
                        ),
                        html.Div(id="flow-status", className="mt-2")
                    ])
                ])
            ], width=6)
        ], className="mb-4"),
        
        # Charts
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📈 Pressure Chart"),
                    dbc.CardBody([
                        dcc.Graph(id="simple-pressure-chart", style={"height": "400px"})
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📈 Flow Chart"),  
                    dbc.CardBody([
                        dcc.Graph(id="simple-flow-chart", style={"height": "400px"})
                    ])
                ])
            ], width=6)
        ])
    ])

def create_simple_chart(data: Dict, selected_elements: List[str], chart_type: str) -> go.Figure:
    """Create a simple chart with selected elements"""
    
    fig = go.Figure()
    
    if not selected_elements or not data:
        # Empty chart with message
        fig.add_annotation(
            text=f"Select {chart_type.lower()} above to see data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title=f"{chart_type} Monitoring",
            xaxis_title="Time (hours)",
            yaxis_title="Pressure (m)" if chart_type == "Pressure" else "Flow (m³/s)",
            template="plotly_white"
        )
        return fig
    
    # Colors for lines
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Get time data
    time_data = data.get('time', [])
    
    # Add traces for selected elements
    for i, element in enumerate(selected_elements):
        color = colors[i % len(colors)]
        
        if chart_type == "Pressure" and element in data.get('pressure', {}):
            y_data = data['pressure'][element]
            fig.add_trace(go.Scatter(
                x=time_data,
                y=y_data,
                mode='lines+markers',
                name=f"🔵 {element}",
                line=dict(color=color, width=2),
                marker=dict(size=4, color=color)
            ))
        elif chart_type == "Flow" and element in data.get('flow', {}):
            y_data = data['flow'][element]
            fig.add_trace(go.Scatter(
                x=time_data,
                y=y_data,
                mode='lines+markers', 
                name=f"🔗 {element}",
                line=dict(color=color, width=2),
                marker=dict(size=4, color=color)
            ))
    
    fig.update_layout(
        title=f"{chart_type} Monitoring ({len(selected_elements)} elements)",
        xaxis_title="Time (hours)",
        yaxis_title="Pressure (m)" if chart_type == "Pressure" else "Flow (m³/s)",
        template="plotly_white",
        hovermode='x unified'
    )
    
    return fig

def register_simple_callbacks(app: dash.Dash):
    """Register simple callbacks for the monitoring system"""
    
    @app.callback(
        [Output('simple-pressure-chart', 'figure'),
         Output('pressure-status', 'children')],
        [Input('simple-pressure-nodes', 'value'),
         Input('simulation-data', 'data')],
        prevent_initial_call=True
    )
    def update_pressure_chart(selected_nodes, sim_data):
        """Update pressure chart when nodes are selected"""
        
        # Create chart
        fig = create_simple_chart(sim_data, selected_nodes or [], "Pressure")
        
        # Create status message
        if selected_nodes:
            status = dbc.Alert(
                f"✅ Monitoring {len(selected_nodes)} node(s): {', '.join(selected_nodes)}",
                color="success"
            )
        else:
            status = dbc.Alert("ℹ️ No nodes selected", color="info")
        
        return fig, status
    
    @app.callback(
        [Output('simple-flow-chart', 'figure'),
         Output('flow-status', 'children')],
        [Input('simple-flow-links', 'value'),
         Input('simulation-data', 'data')],
        prevent_initial_call=True
    )
    def update_flow_chart(selected_links, sim_data):
        """Update flow chart when links are selected"""
        
        # Create chart
        fig = create_simple_chart(sim_data, selected_links or [], "Flow")
        
        # Create status message
        if selected_links:
            status = dbc.Alert(
                f"✅ Monitoring {len(selected_links)} link(s): {', '.join(selected_links)}",
                color="success"
            )
        else:
            status = dbc.Alert("ℹ️ No links selected", color="info")
        
        return fig, status

# Usage instructions for integration:
"""
To integrate this simple monitoring system:

1. Import this module in your main app
2. Replace the complex monitoring section with create_simple_monitoring_section()
3. Call register_simple_callbacks(app) to register the callbacks
4. Remove all the complex real-time monitoring code

Example:
    from simple_monitoring import create_simple_monitoring_section, register_simple_callbacks
    
    # In your layout:
    monitoring_section = create_simple_monitoring_section(available_nodes, available_links)
    
    # After app creation:
    register_simple_callbacks(app)
""" 