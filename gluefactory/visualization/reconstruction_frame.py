"""Plotly-based 3D reconstruction visualizer for the inspect flow."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pycolmap

from flexglue.eval.viztools import serve_plotly_figure

from ..geometry.reconstruction import Reconstruction
from . import viz3d

# Colors for different benchmarks
COLORS = [
    "rgb(31, 119, 180)",
    "rgb(255, 127, 14)",
    "rgb(44, 160, 44)",
    "rgb(214, 39, 40)",
]


def plot_cameras_batched(fig, rec, color="rgb(0, 0, 255)", name=None, scale=3.0):
    """Plot all camera frustums as a single Scatter3d trace (fast)."""
    all_x, all_y, all_z = [], [], []
    for i in range(rec.w_t_c.shape[0]):
        R = rec.w_t_c[i].R.cpu().numpy()
        t = rec.w_t_c[i].t.cpu().numpy()
        K = rec.get_camera(i).K.cpu().numpy()
        W, H = K[0, 2] * 2, K[1, 2] * 2
        corners = np.array([[0, 0], [W, 0], [W, H], [0, H], [0, 0]])
        image_extent = max(scale * W / 1024.0, scale * H / 1024.0)
        world_extent = max(W, H) / (K[0, 0] + K[1, 1]) / 0.5
        s = 0.5 * image_extent / world_extent
        corners_h = np.concatenate([corners, np.ones((corners.shape[0], 1))], axis=1)
        corners_3d = (corners_h @ np.linalg.inv(K).T / 2 * s) @ R.T + t
        # Draw lines: center->c0, c0->c1, c1->c2, c2->c3, c3->c0,
        #             center->c1, center->c2, center->c3
        center = t
        for a, b in [
            (center, corners_3d[0]),
            (corners_3d[0], corners_3d[1]),
            (corners_3d[1], corners_3d[2]),
            (corners_3d[2], corners_3d[3]),
            (corners_3d[3], corners_3d[0]),
            (center, corners_3d[1]),
            (center, corners_3d[2]),
            (center, corners_3d[3]),
        ]:
            all_x.extend([a[0], b[0], None])
            all_y.extend([a[1], b[1], None])
            all_z.extend([a[2], b[2], None])

    fig.add_trace(
        go.Scatter3d(
            x=all_x,
            y=all_y,
            z=all_z,
            mode="lines",
            name=name,
            legendgroup=name,
            line=dict(color=color, width=1),
        )
    )


def plot_colmap_points(fig, colmap_rec, name=None, ps=1, max_points=5_000):
    """Extract and plot 3D points from a pycolmap.Reconstruction."""
    points3D = colmap_rec.points3D
    if len(points3D) == 0:
        return
    pts = np.array([p.xyz for p in points3D.values()])
    colors = np.array([p.color for p in points3D.values()], dtype=np.uint8)

    # Subsample if too many points
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts, colors = pts[idx], colors[idx]

    # Pack RGB into single int for plotly: avoids per-point string overhead
    color_ints = (
        colors[:, 0].astype(np.uint32) * 65536
        + colors[:, 1].astype(np.uint32) * 256
        + colors[:, 2].astype(np.uint32)
    )
    # Convert to #RRGGBB hex strings
    color_hex = [f"#{c:06x}" for c in color_ints]

    x, y, z = pts.T
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            name=name,
            legendgroup=name,
            marker=dict(
                size=ps,
                color=color_hex,
                line_width=0.0,
            ),
        )
    )


class ReconstructionFrame:
    """Plotly 3D visualization of a COLMAP reconstruction."""

    default_conf = {
        "default": "reconstruction",
        "max_points": 500_000,
    }

    def __init__(self, conf, data, preds, title=None, event=1, summaries=None):
        self.conf = conf
        self.data = data
        self.preds = preds
        self.names = list(preds.keys())
        self.summaries = summaries
        self.fig = None
        self._build(title)

    def _build(self, title=None):
        fig = viz3d.init_figure(height=800)

        rec_data = self.data.get("reconstruction")
        gt_colmap_rec = None
        # Plot GT cameras if reference_sfm is available
        if rec_data is not None:
            ref_sfm = rec_data[0] if isinstance(rec_data, list) else rec_data
            if hasattr(ref_sfm, "reference_sfm") and ref_sfm.reference_sfm is not None:
                gt_colmap_rec = ref_sfm.reference_sfm
                if isinstance(gt_colmap_rec, Path):
                    gt_colmap_rec = pycolmap.Reconstruction(gt_colmap_rec)
                gt_rec = Reconstruction.from_colmap(gt_colmap_rec)
                plot_cameras_batched(
                    fig,
                    gt_rec,
                    color="rgb(255, 0, 0)",
                    name="GT cameras",
                    scale=3.0,
                )

        # Plot estimated reconstruction for each benchmark
        for i, (name, pred) in enumerate(self.preds.items()):
            output_dir = pred.get("output_dir")
            if output_dir is None:
                continue
            output_dir = output_dir[0] if isinstance(output_dir, list) else output_dir

            colmap_rec = pycolmap.Reconstruction(str(output_dir))

            # Coarsely align estimated poses to GT via Sim3d
            if gt_colmap_rec is not None:
                sim3 = pycolmap.align_reconstructions_via_proj_centers(
                    colmap_rec, gt_colmap_rec, max_proj_center_error=1.0
                )
                if sim3 is not None:
                    colmap_rec.transform(sim3)

            color = COLORS[i % len(COLORS)]

            rec = Reconstruction.from_colmap(colmap_rec)
            plot_cameras_batched(fig, rec, color=color, name=name, scale=3.0)
            plot_colmap_points(
                fig,
                colmap_rec,
                name=f"{name} points",
                ps=1,
                max_points=self.conf.get("max_points", 5_000),
            )

        if title:
            fig.update_layout(title=title)

        self.fig = fig
        serve_plotly_figure(fig)

    def show(self):
        if self.fig is not None:
            serve_plotly_figure(self.fig)

    def close(self):
        pass
