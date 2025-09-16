import inspect
import sys
import warnings

import matplotlib.pyplot as plt
import torch
from matplotlib.backend_tools import ToolToggleBase
from matplotlib.widgets import RadioButtons, Slider

from ..geometry import depth, epipolar, homography
from . import viz2d

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    plt.rcParams["toolbar"] = "toolmanager"


_COMMON = {
    "DRAW_LINE_MODE": "auto",  # "on", "off", "auto"
    "MAX_NUM_LINES": 4096,  # maximum number of lines to draw in auto mode
    "DRAW_LINE_WIDTH": 0.5,  # default line width
    "DRAW_LINE_ALPHA": 0.5,  # default line alpha
}


def auto_linewidth(num_lines: int) -> float:
    """Get a line width based on the number of lines and the current mode."""
    if _COMMON["DRAW_LINE_MODE"] == "on" or (
        _COMMON["DRAW_LINE_MODE"] == "auto" and num_lines < _COMMON["MAX_NUM_LINES"]
    ):
        return _COMMON["DRAW_LINE_WIDTH"]
    else:
        return 0.0


class RadioHideTool(ToolToggleBase):
    """Radio button tool."""

    default_keymap = "R"
    description = " "
    default_toggled = False
    radio_group = "default"

    def __init__(
        self, *args, options=[], active=None, callback_fn=None, keymap="R", **kwargs
    ):
        if "description" in kwargs:
            self.description = kwargs.pop("description")
        super().__init__(*args, **kwargs)
        self.f = 1.0
        self.options = options
        self.callback_fn = callback_fn
        self.active = self.options.index(active) if active else 0
        self.default_keymap = keymap

        self.enabled = self.default_toggled

    def build_radios(self):
        w = 0.3
        self.radios_ax = self.figure.add_axes([1.0 - w, 0.2, w, 0.7], zorder=1)
        # self.radios_ax = self.figure.add_axes([0.5-w/2, 1.0-0.2, w, 0.2], zorder=1)
        self.radios = RadioButtons(self.radios_ax, self.options, active=self.active)
        for r in self.radios.labels:
            r.set_fontsize(8)
        self.radios.on_clicked(self.on_radio_clicked)

    def enable(self, *args):
        size = self.figure.get_size_inches()
        size[0] *= self.f
        self.build_radios()
        self.figure.canvas.draw_idle()
        self.enabled = True

    def disable(self, *args):
        size = self.figure.get_size_inches()
        size[0] /= self.f
        self.radios_ax.remove()
        self.radios = None
        self.figure.canvas.draw_idle()
        self.enabled = False

    def on_radio_clicked(self, value):
        self.active = self.options.index(value)
        enabled = self.enabled
        if enabled:
            self.disable()
        if self.callback_fn is not None:
            self.callback_fn(value)
        if enabled:
            self.enable()


class ToggleTool(ToolToggleBase):
    """Toggle a callback function."""

    default_keymap = "t"
    description = " "

    def __init__(self, *args, callback_fn=None, keymap="t", **kwargs):
        if "description" in kwargs:
            self.description = kwargs.pop("description")
        super().__init__(*args, **kwargs)
        self.f = 1.0
        self.callback_fn = callback_fn
        self.default_keymap = keymap
        self.enabled = self.default_toggled

    def enable(self, *args):
        self.callback_fn(True)

    def disable(self, *args):
        self.callback_fn(False)


def add_whitespace_left(fig, factor):
    w, h = fig.get_size_inches()
    left = fig.subplotpars.left
    fig.set_size_inches([w * (1 + factor), h])
    fig.subplots_adjust(left=(factor + left) / (1 + factor))


def add_whitespace_bottom(fig, factor):
    w, h = fig.get_size_inches()
    b = fig.subplotpars.bottom
    fig.set_size_inches([w, h * (1 + factor)])
    fig.subplots_adjust(bottom=(factor + b) / (1 + factor))
    fig.canvas.draw_idle()


class KeypointPlot:
    plot_name = "keypoints"
    required_keys = ["keypoints0", "keypoints1"]

    def __init__(self, fig, axes, data, preds):
        for i, name in enumerate(preds):
            pred = preds[name]
            viz2d.plot_keypoints(
                [pred["keypoints0"][0], pred["keypoints1"][0]], axes=axes[i]
            )


class LinePlot:
    plot_name = "lines"
    required_keys = ["lines0", "lines1"]

    def __init__(self, fig, axes, data, preds):
        for i, name in enumerate(preds):
            pred = preds[name]
            viz2d.plot_lines([pred["lines0"][0], pred["lines1"][0]])


class KeypointRankingPlot:
    plot_name = "keypoint_ranking"
    required_keys = ["keypoints0", "keypoints1", "keypoint_scores0", "keypoint_scores1"]

    def __init__(self, fig, axes, data, preds):
        for i, name in enumerate(preds):
            pred = preds[name]
            kp0, kp1 = pred["keypoints0"][0], pred["keypoints1"][0]
            sc0, sc1 = pred["keypoint_scores0"][0], pred["keypoint_scores1"][0]

            viz2d.plot_keypoints(
                [kp0, kp1],
                axes=axes[i],
                colors=[viz2d.cm_ranking(sc0), viz2d.cm_ranking(sc1)],
            )


class KeypointScoresPlot:
    plot_name = "keypoint_scores"
    required_keys = ["keypoints0", "keypoints1", "keypoint_scores0", "keypoint_scores1"]

    def __init__(self, fig, axes, data, preds):
        for i, name in enumerate(preds):
            pred = preds[name]
            kp0, kp1 = pred["keypoints0"][0], pred["keypoints1"][0]
            sc0, sc1 = pred["keypoint_scores0"][0], pred["keypoint_scores1"][0]
            viz2d.plot_keypoints(
                [kp0, kp1],
                axes=axes[i],
                colors=[viz2d.cm_RdGn(sc0), viz2d.cm_RdGn(sc1)],
            )


class HeatmapPlot:
    plot_name = "heatmaps"
    required_keys = ["heatmap0", "heatmap1"]

    def __init__(self, fig, axes, data, preds):
        self.artists = []
        for i, name in enumerate(preds):
            pred = preds[name]
            heatmaps = [pred["heatmap0"][0, 0], pred["heatmap1"][0, 0]]
            heatmaps = [torch.sigmoid(h) if h.min() < 0.0 else h for h in heatmaps]
            self.artists += viz2d.plot_heatmaps(heatmaps, axes=axes[i], cmap="rainbow")

    def clear(self):
        for x in self.artists:
            x.remove()


class ImagePlot:
    plot_name = "images"
    required_keys = []

    def __init__(self, fig, axes, data, preds):
        pass


class MatchesPlot:
    plot_name = "matches"
    required_keys = ["keypoints0", "keypoints1", "matches0"]

    def __init__(self, fig, axes, _, preds):
        self.fig = fig
        self.sbpars = {
            k: v
            for k, v in vars(fig.subplotpars).items()
            if k in ["left", "right", "top", "bottom"]
        }

        for i, name in enumerate(preds):
            pred = preds[name]
            kp0, kp1 = pred["keypoints0"][0], pred["keypoints1"][0]
            m0 = pred["matches0"][0]
            valid = m0 > -1
            kpm0 = kp0[valid]
            kpm1 = kp1[m0[valid]]
            mscores = pred["matching_scores0"][0][valid]
            viz2d.plot_matches(
                kpm0,
                kpm1,
                axes=axes[i],
                labels=mscores,
                lw=auto_linewidth(kpm0.shape[0]),
                a=_COMMON["DRAW_LINE_ALPHA"],
            )


class MatchScoresPlot:
    plot_name = "matching_scores"
    required_keys = ["keypoints0", "keypoints1", "matches0", "matching_scores0"]

    def __init__(self, fig, axes, _, preds):
        self.fig = fig
        self.sbpars = {
            k: v
            for k, v in vars(fig.subplotpars).items()
            if k in ["left", "right", "top", "bottom"]
        }

        for i, name in enumerate(preds):
            pred = preds[name]
            viz2d.plot_keypoints(
                [pred["keypoints0"][0], pred["keypoints1"][0]],
                axes=axes[i],
                colors="blue",
            )
            kp0, kp1 = pred["keypoints0"][0], pred["keypoints1"][0]
            m0 = pred["matches0"][0]
            valid = m0 > -1
            kpm0 = kp0[valid]
            kpm1 = kp1[m0[valid]]
            mscores = pred["matching_scores0"][0][valid]
            viz2d.plot_matches(
                kpm0,
                kpm1,
                color=viz2d.cm_RdGn(mscores).tolist(),
                axes=axes[i],
                labels=mscores,
                lw=auto_linewidth(kpm0.shape[0]),
                a=_COMMON["DRAW_LINE_ALPHA"],
            )


class LineMatchesPlot:
    plot_name = "line_matches"
    required_keys = ["lines0", "lines1", "line_matches0"]

    def __init__(self, fig, axes, data, preds):
        self.fig = fig
        self.sbpars = {
            k: v
            for k, v in vars(fig.subplotpars).items()
            if k in ["left", "right", "top", "bottom"]
        }

        for i, name in enumerate(preds):
            pred = preds[name]
            lines0, lines1 = pred["lines0"][0], pred["lines1"][0]
            m0 = pred["line_matches0"][0]
            valid = m0 > -1
            m_lines0 = lines0[valid]
            m_lines1 = lines1[m0[valid]]
            viz2d.plot_color_line_matches([m_lines0, m_lines1])


class GtMatchesPlot:
    plot_name = "gt_matches"
    required_keys = ["keypoints0", "keypoints1", "matches0", "gt_matches0"]

    def __init__(self, fig, axes, data, preds):
        self.fig = fig
        self.sbpars = {
            k: v
            for k, v in vars(fig.subplotpars).items()
            if k in ["left", "right", "top", "bottom"]
        }

        for i, name in enumerate(preds):
            pred = preds[name]
            viz2d.plot_keypoints(
                [pred["keypoints0"][0], pred["keypoints1"][0]],
                axes=axes[i],
                colors="blue",
            )
            kp0, kp1 = pred["keypoints0"][0], pred["keypoints1"][0]
            m0 = pred["matches0"][0]
            gtm0 = pred["gt_matches0"][0]
            valid = (m0 > -1) & (gtm0 >= -1)
            kpm0 = kp0[valid]
            kpm1 = kp1[m0[valid]]
            correct = gtm0[valid] == m0[valid]
            viz2d.plot_matches(
                kpm0,
                kpm1,
                color=viz2d.cm_RdGn(correct).tolist(),
                axes=axes[i],
                labels=correct,
                lw=auto_linewidth(kpm0.shape[0]),
                a=_COMMON["DRAW_LINE_ALPHA"],
            )


class GtLineMatchesPlot:
    plot_name = "gt_line_matches"
    required_keys = ["lines0", "lines1", "line_matches0", "line_gt_matches0"]

    def __init__(self, fig, axes, data, preds):
        self.fig = fig
        self.sbpars = {
            k: v
            for k, v in vars(fig.subplotpars).items()
            if k in ["left", "right", "top", "bottom"]
        }

        for i, name in enumerate(preds):
            pred = preds[name]
            lines0, lines1 = pred["lines0"][0], pred["lines1"][0]
            m0 = pred["line_matches0"][0]
            gtm0 = pred["gt_line_matches0"][0]
            valid = (m0 > -1) & (gtm0 >= -1)
            m_lines0 = lines0[valid]
            m_lines1 = lines1[m0[valid]]
            viz2d.plot_color_line_matches([m_lines0, m_lines1])


class HomographyMatchesPlot:
    plot_name = "homography"
    required_keys = ["keypoints0", "keypoints1", "matches0", "H_0to1"]

    def error_fn(self, kpm0, kpm1, data):
        """Calculate the reprojection error."""
        H_0to1 = data["H_0to1"][0]
        return homography.sym_homography_error(kpm0, kpm1, H_0to1), torch.ones(
            kpm0.shape[0], dtype=torch.bool, device=kpm0.device
        )

    def __init__(self, fig, axes, data, preds):
        self.fig = fig
        self.sbpars = {
            k: v
            for k, v in vars(fig.subplotpars).items()
            if k in ["left", "right", "top", "bottom"]
        }

        add_whitespace_bottom(fig, 0.1)

        self.range_ax = fig.add_axes([0.3, 0.02, 0.4, 0.06])
        self.range = Slider(
            self.range_ax,
            label="Error [px]",
            valmin=0,
            valmax=5,
            valinit=3.0,
            valstep=1.0,
        )
        self.range.on_changed(self.color_matches)
        self.axes = axes
        self.errors = []

        for i, name in enumerate(preds):
            pred = preds[name]
            viz2d.plot_keypoints(
                [pred["keypoints0"][0], pred["keypoints1"][0]],
                axes=axes[i],
                colors="blue",
            )
            kp0, kp1 = pred["keypoints0"][0], pred["keypoints1"][0]
            m0 = pred["matches0"][0]
            valid = m0 > -1
            kpm0 = kp0[valid]
            kpm1 = kp1[m0[valid]]
            errors, valid_m = self.error_fn(kpm0, kpm1, data)
            errors, valid_m = errors.cpu().numpy(), valid_m.cpu().numpy()
            viz2d.plot_matches(
                kpm0[valid_m],
                kpm1[valid_m],
                color=viz2d.cm_RdGn(errors[valid_m] < self.range.val).tolist(),
                axes=axes[i],
                labels=errors[valid_m],
                lw=auto_linewidth(kpm0[valid_m].shape[0]),
                a=_COMMON["DRAW_LINE_ALPHA"],
            )
            self.errors.append(errors[valid_m])

    def clear(self):
        w, h = self.fig.get_size_inches()
        self.fig.set_size_inches(w, h / 1.1)
        self.fig.subplots_adjust(**self.sbpars)
        self.range_ax.remove()

    def color_matches(self, threshold):
        # Update line colors.
        for line in self.fig.artists:
            label = line.get_label()
            line.set_color(viz2d.cm_RdGn([float(label) < threshold])[0])
        # Update match colors.
        for errors, axes in zip(self.errors, self.axes):
            for ax in axes:
                for coll in ax.collections:
                    arr = coll.get_facecolors()
                    if arr is not None and arr.shape[0] == errors.shape[0]:
                        coll.set_facecolors(viz2d.cm_RdGn(errors < threshold).tolist())


class ReprojectionMatchesPlot(HomographyMatchesPlot):
    plot_name = "depth_matches"
    required_keys = [
        "keypoints0",
        "keypoints1",
        "matches0",
        "view0.depth",
        "view1.depth",
        "T_0to1",
    ]

    def error_fn(self, kpm0, kpm1, data):
        """Calculate the reprojection error."""
        reproj_error, valid = depth.symmetric_reprojection_error(
            kpm0[None],
            kpm1[None],
            data["view0"]["camera"],
            data["view1"]["camera"],
            data["T_0to1"],
            data["view0"]["depth"],
            data["view1"]["depth"],
        )

        reproj_error, valid = reproj_error[0], valid[0]
        return reproj_error, valid


class EpipolarMatchesPlot:
    plot_name = "epipolar_matches"
    required_keys = [
        "keypoints0",
        "keypoints1",
        "matches0",
        "T_0to1",
        "view0.camera",
        "view1.camera",
    ]

    def __init__(self, fig, axes, data, preds):
        self.fig = fig
        self.axes = axes
        self.sbpars = {
            k: v
            for k, v in vars(fig.subplotpars).items()
            if k in ["left", "right", "top", "bottom"]
        }

        add_whitespace_bottom(fig, 0.1)

        self.range_ax = fig.add_axes([0.3, 0.02, 0.4, 0.06])
        self.range = Slider(
            self.range_ax,
            label="Epipolar Error [px]",
            valmin=0,
            valmax=5,
            valinit=3.0,
            valstep=1.0,
        )
        self.range.on_changed(self.color_matches)

        camera0 = data["view0"]["camera"][0]
        camera1 = data["view1"]["camera"][0]
        T_0to1 = data["T_0to1"][0]

        self.errors = []
        for i, name in enumerate(preds):
            pred = preds[name]
            viz2d.plot_keypoints(
                [pred["keypoints0"][0], pred["keypoints1"][0]],
                axes=axes[i],
                colors="blue",
            )
            kp0, kp1 = pred["keypoints0"][0], pred["keypoints1"][0]
            m0 = pred["matches0"][0]
            valid = m0 > -1
            kpm0 = kp0[valid]
            kpm1 = kp1[m0[valid]]

            errors = epipolar.generalized_epi_dist(
                kpm0,
                kpm1,
                camera0,
                camera1,
                T_0to1,
                all=False,
                essential=False,
            )
            viz2d.plot_matches(
                kpm0,
                kpm1,
                color=viz2d.cm_RdGn(errors < self.range.val).tolist(),
                axes=axes[i],
                labels=errors.numpy(),
                lw=auto_linewidth(kpm0.shape[0]),
                a=_COMMON["DRAW_LINE_ALPHA"],
            )

            self.errors.append(errors.numpy())

        self.F = epipolar.T_to_F(camera0, camera1, T_0to1)

    def clear(self):
        w, h = self.fig.get_size_inches()
        self.fig.set_size_inches(w, h / 1.1)
        self.fig.subplots_adjust(**self.sbpars)
        self.range_ax.remove()

    def color_matches(self, threshold):
        # Update line colors.
        for art in self.fig.artists:
            label = art.get_label()
            if label is not None:
                art.set_color(viz2d.cm_RdGn([float(label) < threshold])[0])
        # Update match colors.
        for errors, axes in zip(self.errors, self.axes):
            for ax in axes:
                for coll in ax.collections:
                    arr = coll.get_facecolors()
                    if arr is not None and arr.shape[0] == errors.shape[0]:
                        coll.set_facecolors(viz2d.cm_RdGn(errors < threshold).tolist())

    def click_artist(self, event):
        art = event.artist
        if art.get_label() is not None:
            if hasattr(art, "epilines"):
                [
                    x.set_visible(not x.get_visible())
                    for x in art.epilines
                    if x is not None
                ]
            else:
                xy1 = art.xy1
                xy2 = art.xy2
                line0 = viz2d.get_line(self.F.transpose(0, 1), xy2)[:, 0]
                line1 = viz2d.get_line(self.F, xy1)[:, 0]
                art.epilines = [
                    viz2d.draw_epipolar_line(line0, art.axesA),
                    viz2d.draw_epipolar_line(line1, art.axesB),
                ]


__plot_dict__ = {
    obj.plot_name: obj
    for _, obj in inspect.getmembers(sys.modules[__name__], predicate=inspect.isclass)
    if hasattr(obj, "plot_name")
}
