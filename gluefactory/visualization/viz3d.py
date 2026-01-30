"""
3D visualization based on plotly.
Works for a small number of points and cameras, might be slow otherwise.

1) Initialize a figure with `init_figure`
2) Add 3D points, camera frustums, or both as a pycolmap.Reconstruction

Written by Paul-Edouard Sarlin and Philipp Lindenberger.
"""

from typing import Optional, Sequence

import numpy as np
import plotly.graph_objects as go
import pycolmap
import torch


def init_figure(height: int = 800, projection: str = "orthographic") -> go.Figure:
    """Initialize a 3D figure."""
    fig = go.Figure()
    axes = dict(
        visible=False,
        showbackground=False,
        showgrid=False,
        showline=False,
        showticklabels=True,
        autorange=True,
    )
    fig.update_layout(
        template="plotly_dark",
        height=height,
        scene_camera=dict(
            projection=dict(
                type=projection,
            )
        ),
        scene=dict(
            xaxis=axes,
            yaxis=axes,
            zaxis=axes,
            aspectmode="data",
            dragmode="orbit",
        ),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="left", x=0.1),
    )
    return fig


def to_homogeneous(points):
    pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
    return np.concatenate([points, pad], axis=-1)


def plot_camera(
    fig: go.Figure,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    color: str = "rgb(0, 0, 255)",
    name: Optional[str] = None,
    legendgroup: Optional[str] = None,
    fill: bool = False,
    size: float = 1.0,
    text: Optional[str] = None,
):
    """Plot a camera frustum from pose and intrinsic matrix."""
    W, H = K[0, 2] * 2, K[1, 2] * 2
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H], [0, 0]])
    if size is not None:
        image_extent = max(size * W / 1024.0, size * H / 1024.0)
        world_extent = max(W, H) / (K[0, 0] + K[1, 1]) / 0.5
        scale = 0.5 * image_extent / world_extent
    else:
        scale = 1.0
    corners = to_homogeneous(corners) @ np.linalg.inv(K).T
    corners = (corners / 2 * scale) @ R.T + t
    legendgroup = legendgroup if legendgroup is not None else name

    x, y, z = np.concatenate(([t], corners)).T
    i = [0, 0, 0, 0]
    j = [1, 2, 3, 4]
    k = [2, 3, 4, 1]

    if fill:
        pyramid = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            color=color,
            i=i,
            j=j,
            k=k,
            legendgroup=legendgroup,
            name=name,
            showlegend=False,
            hovertemplate=text.replace("\n", "<br>") if text else None,
        )
        fig.add_trace(pyramid)

    triangles = np.vstack((i, j, k)).T
    vertices = np.concatenate(([t], corners))
    tri_points = np.array([vertices[i] for i in triangles.reshape(-1)])
    x, y, z = tri_points.T

    pyramid = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        legendgroup=legendgroup,
        name=name,
        line=dict(color=color, width=1),
        showlegend=False,
        hovertemplate=text.replace("\n", "<br>") if text else None,
    )
    fig.add_trace(pyramid)


def plot_cameras(
    fig: go.Figure,
    w_t_c: np.ndarray,
    camera: pycolmap.Camera,
    scale: float = 3.0,
    color: str = "rgb(0, 0, 255)",
    name: Optional[str] = None,
    legendgroup: Optional[str] = None,
    fill: bool = False,
):
    """Plot camera frustums from a pycolmap.Reconstruction."""
    for i, (w_t_ci, cami) in enumerate(zip(w_t_c, camera)):
        R = w_t_ci.R.cpu().numpy()
        t = w_t_ci.t.cpu().numpy()
        K = cami.K.cpu().numpy()
        name_i = name
        if isinstance(name, Sequence):
            name_i = name[i]
        plot_camera(
            fig,
            R,
            t,
            K,
            color=color,
            name=f"{name_i} {i}" if name_i else str(i),
            legendgroup=legendgroup,
            fill=fill,
            size=scale,
            text=f"Camera {i}\n" + (name_i if name_i else ""),
        )


def plot_points(
    fig: go.Figure,
    pts: np.ndarray,
    color: str = "rgba(255, 0, 0, 1)",
    ps: int = 2,
    colorscale: Optional[str] = None,
    name: Optional[str] = None,
    edge_filter: float | None = None,
    cmin: float | None = None,
    cmax: float | None = None,
    **kwargs,
):
    """Plot a set of 3D points."""

    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().numpy()
    if isinstance(color, torch.Tensor):
        color = color.detach().cpu().numpy()
    x, y, z = pts.T
    colorbar = None if colorscale is None else dict(thickness=20, orientation="h")
    tr = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        name=name,
        legendgroup=name,
        marker=dict(
            size=ps,
            color=color,
            line_width=0.0,
            colorscale=colorscale,
            colorbar=colorbar,
            cmin=cmin,
            cmax=cmax,
        ),
        **kwargs,
    )
    fig.add_trace(tr)
