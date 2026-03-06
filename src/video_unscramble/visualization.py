import base64
import glob
import os

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.manifold import TSNE

from .core import compute_histogram_features

def load_images_from_folder(folder):
    """
    Load all JPEG image paths from a folder.

    Args:
        folder: Path to the directory containing `.jpg` images.

    Returns:
        Sorted list of image file paths.
    """
    image_paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    return image_paths

def generate_plotly_visualization(frames, labels, output_html, image_paths=None):
    """
    Generate and save an interactive 2D t-SNE plot of frame clusters.

    Args:
        frames: List of frames.
        labels: Cluster labels for each frame (0 = outlier, 1 = inlier).
        output_html: Path to save the HTML visualization.
        image_paths: Optional list of file paths to images for click previews.

    Returns:
        None. Saves an HTML file with interactive plot.
    """
    cluster_labels = {"Cluster 0": "Outliers",
                     "Cluster 1": "Inliers"}
    cluster_colors = {"Cluster 0": "#FF4B6D",
                     "Cluster 1": "#00D4AA"}
    
    feats = compute_histogram_features(frames, 
                                       bins=64,
                                       resize=None)
    pts2d = TSNE(n_components=2,
                random_state=42,
                perplexity=100,
                learning_rate=200).fit_transform(feats)
    if image_paths is None:
        image_paths = [f"frame_{i:05d}.jpg" for i in range(len(frames))]
    rel_paths = [os.path.relpath(p) for p in image_paths]
    
    def make_preview_tag(img, w=256):
        _, buf = cv2.imencode('.jpg', img)
        b64 = base64.b64encode(buf).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"
    
    previews = [make_preview_tag(f) for f in frames]
    df = pd.DataFrame({
        "x": pts2d[:,0],
        "y": pts2d[:,1],
        "cluster": labels,
        "preview": previews,
        "path": rel_paths
    })
    fig = go.Figure()
    for c in sorted(df.cluster.unique()):
        d = df[df.cluster == c]
        cluster_key = f"Cluster {c}"
        fig.add_trace(go.Scatter(
            x=d.x, y=d.y,
            mode="markers",
            name=cluster_labels[cluster_key],
            marker=dict(
                size=12,
                opacity=0.85,
                color=cluster_colors[cluster_key],
                line=dict(width=2, color='white'),
                symbol='circle'
            ),
            customdata=np.stack([d.preview, d.path], axis=1),
            hovertemplate=""
        ))
    
    x_range = [pts2d[:,0].min() - 2, pts2d[:,0].max() + 2]
    y_range = [pts2d[:,1].min() - 2, pts2d[:,1].max() + 2]
    
    fig.update_layout(
        title=dict(
            text="Interactive t-SNE video tampered frames clustering",
            font=dict(size=24, color='#2E3440', family="Arial Black"),
            x=0.5
        ),
        xaxis=dict(visible=False, range=x_range),
        yaxis=dict(visible=False, range=y_range),
        plot_bgcolor='#F8FAFC',
        paper_bgcolor='#FFFFFF',
        hoverlabel=dict(align="left"),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=14, color='#2E3440'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#E5E7EB',
            borderwidth=1
        ),
        font=dict(family="Arial", color='#374151'),
        margin=dict(l=80, r=150, t=100, b=80),
        width=1000,
        height=700
    )
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    js = """
    <script>
    // wait for the plot to be fully rendered
    document.querySelectorAll('.plotly-graph-div').forEach(function(gd){
        var layout = gd.layout;
        var xrange = layout.xaxis.range;
        var yrange = layout.yaxis.range;
        var xspan = xrange[1] - xrange[0];
        var yspan = yrange[1] - yrange[0];
        var imgSize = Math.min(xspan, yspan) * 0.15;
        
        // on hover: show preview
        gd.on('plotly_hover', function(e){
            var p = e.points[0].customdata[0];
            var x = e.points[0].x, y = e.points[0].y;
            
            // Adjust position to keep image within plot bounds
            var adjustedX = x;
            var adjustedY = y;
            
            if (x + imgSize/2 > xrange[1]) adjustedX = xrange[1] - imgSize/2;
            if (x - imgSize/2 < xrange[0]) adjustedX = xrange[0] + imgSize/2;
            if (y + imgSize/2 > yrange[1]) adjustedY = yrange[1] - imgSize/2;
            if (y - imgSize/2 < yrange[0]) adjustedY = yrange[0] + imgSize/2;
            
            var imgObj = {
                source: p,
                xref: 'x', yref: 'y',
                x: adjustedX, y: adjustedY,
                sizex: imgSize, sizey: imgSize,
                xanchor: 'center', yanchor: 'middle'
            };
            Plotly.relayout(gd, {'images':[imgObj]});
        });
        // on unhover: remove preview
        gd.on('plotly_unhover', function(e){
            Plotly.relayout(gd, {'images':[]});
        });
        // on click: open full image
        gd.on('plotly_click', function(e){
            var url = e.points[0].customdata[1];
            window.open(url, '_blank');
        });
    });
    </script>
    """
    html = html.replace("</body>", js + "</body>")
    with open(output_html, "w") as f:
        f.write(html)
    print(f"Interactive t-SNE visualization saved to: {output_html}")
