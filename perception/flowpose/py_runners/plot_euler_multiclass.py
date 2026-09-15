import os
import glob
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main():
    # Directory containing per-class analysis result
    data_dir = "results/pose_by_class"
    if not os.path.exists(data_dir):
        print(f"Error: Could not find {data_dir}. Please run data_analysis_node_by_cls.py first.")
        return

    npz_files = sorted(glob.glob(os.path.join(data_dir, "pose_euler_analysis_*.npz")))
    if not npz_files:
        print(f"No npz files found in {data_dir}.")
        return

    # Extract class names
    classes = []
    data_dict = {}
    for f in npz_files:
        cls_name = os.path.basename(f).replace("pose_euler_analysis_", "").replace(".npz", "")
        classes.append(cls_name)
        data_dict[cls_name] = np.load(f)["euler_angles_deg"]

    print(f"Found {len(classes)} classes: {classes}")

    # Create subplots: each class gets 1 row, 4 columns (X, Y, Z, 3D Scatter)
    num_classes = len(classes)
    
    # Specs for subplots
    specs = []
    subplot_titles = []
    
    for cls in classes:
        specs.append([
            {"type": "histogram"}, 
            {"type": "histogram"}, 
            {"type": "histogram"}, 
            {"type": "scatter3d"}
        ])
        subplot_titles.extend([
            f"{cls} - Roll (X)", 
            f"{cls} - Pitch (Y)", 
            f"{cls} - Yaw (Z)", 
            f"{cls} - 3D Poses"
        ])

    print("Generating interactive Plotly figure...")
    fig = make_subplots(
        rows=num_classes, cols=4,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.03
    )

    sample_size = 5000  # reduce sample size for multiclass to avoid browser crash

    # Add components for each class
    for i, cls in enumerate(classes):
        row = i + 1
        euler_deg = data_dict[cls]
        
        # Subsample for the 3D scatter to keep the browser responsive
        if euler_deg.shape[0] > sample_size:
            idx = np.random.choice(euler_deg.shape[0], sample_size, replace=False)
            euler_deg_sample = euler_deg[idx]
        else:
            euler_deg_sample = euler_deg

        # Add 1D Histograms
        fig.add_trace(go.Histogram(x=euler_deg[:, 0], name=f'{cls} Roll', marker_color='#1f77b4', nbinsx=90), row=row, col=1)
        fig.add_trace(go.Histogram(x=euler_deg[:, 1], name=f'{cls} Pitch', marker_color='#ff7f0e', nbinsx=90), row=row, col=2)
        fig.add_trace(go.Histogram(x=euler_deg[:, 2], name=f'{cls} Yaw', marker_color='#2ca02c', nbinsx=90), row=row, col=3)

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
            name=f'{cls} 3D'
        ), row=row, col=4)
        
        # Configure 3D axes labels per row
        fig.update_scenes(
            xaxis_title='Roll (X)',
            yaxis_title='Pitch (Y)',
            zaxis_title='Yaw (Z)',
            row=row, col=4
        )

    # Clean up layout
    total_height = max(800, 350 * num_classes)
    fig.update_layout(
        height=total_height, 
        width=1600, 
        title_text="Multi-Class interactive Euler Angle Data Distribution",
        showlegend=False,
    )

    html_file = "results/euler_multiclass_interactive_plot.html"
    fig.write_html(html_file)
    print(f"Interactive summary plot successfully saved to {html_file}")
    
    # Try to open the generated HTML automatically in the user's default browser
    import webbrowser
    browser_opened = webbrowser.open(f"file://{os.path.abspath(html_file)}")
    if browser_opened:
        print("Opened the plot in your default web browser.")
    else:
        print("Could not open browser automatically. You can manually open the HTML file.")

if __name__ == "__main__":
    main()
