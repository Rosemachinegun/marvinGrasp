import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main():
    # Load the extracted euler angles
    npz_path = "results/pose_euler_analysis.npz"
    if not os.path.exists(npz_path):
        print(f"Error: Could not find {npz_path}. Please run data_analysis_node.py first.")
        return

    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path)
    euler_deg = data["euler_angles_deg"]
    
    # Subsample for the 3D scatter to keep the browser responsive
    sample_size = 100000
    if euler_deg.shape[0] > sample_size:
        print(f"Subsampling {sample_size} points for the 3D scatter plot (from {euler_deg.shape[0]} total).")
        idx = np.random.choice(euler_deg.shape[0], sample_size, replace=False)
        euler_deg_sample = euler_deg[idx]
    else:
        euler_deg_sample = euler_deg

    print("Generating interactive Plotly figure...")
    # Create subplots: 3 histograms for X, Y, Z and 1 3D scatter plot
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "histogram"}, {"type": "histogram"}],
               [{"type": "histogram"}, {"type": "scatter3d"}]],
        subplot_titles=("X (Roll) Distribution", "Y (Pitch) Distribution", 
                        "Z (Yaw) Distribution", f"3D Scatter (Sampled {len(euler_deg_sample)})")
    )

    # Add 1D Histograms
    fig.add_trace(go.Histogram(x=euler_deg[:, 0], name='Roll (X)', marker_color='#1f77b4', nbinsx=90), row=1, col=1)
    fig.add_trace(go.Histogram(x=euler_deg[:, 1], name='Pitch (Y)', marker_color='#ff7f0e', nbinsx=90), row=1, col=2)
    fig.add_trace(go.Histogram(x=euler_deg[:, 2], name='Yaw (Z)', marker_color='#2ca02c', nbinsx=90), row=2, col=1)

    # Add 3D Scatter
    fig.add_trace(go.Scatter3d(
        x=euler_deg_sample[:, 0], 
        y=euler_deg_sample[:, 1], 
        z=euler_deg_sample[:, 2],
        mode='markers',
        marker=dict(
            size=2, 
            color=euler_deg_sample[:, 2], # Color by Z-axis
            colorscale='Viridis', 
            opacity=0.6
        ),
        name='3D Poses'
    ), row=2, col=2)

    # Clean up layout
    fig.update_layout(
        height=900, 
        width=1200, 
        title_text="Interactive Euler Angle Data Distribution (Degrees)",
        showlegend=False
    )
    
    # Configure axes labels
    fig.update_xaxes(title_text="Degrees", row=1, col=1)
    fig.update_xaxes(title_text="Degrees", row=1, col=2)
    fig.update_xaxes(title_text="Degrees", row=2, col=1)
    
    # Configure 3D axes labels
    fig.update_scenes(
        xaxis_title='Roll (X)',
        yaxis_title='Pitch (Y)',
        zaxis_title='Yaw (Z)',
        row=2, col=2
    )

    html_file = "results/euler_interactive_plot.html"
    fig.write_html(html_file)
    print(f"Interactive plot successfully saved to {html_file}")
    
    # Try to open the generated HTML automatically in the user's default browser
    import webbrowser
    browser_opened = webbrowser.open(f"file://{os.path.abspath(html_file)}")
    if browser_opened:
        print("Opened the plot in your default web browser.")
    else:
        print("Could not open browser automatically. You can manually open the HTML file.")

if __name__ == "__main__":
    main()
