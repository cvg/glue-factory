import inspect
import sys
import warnings

import matplotlib.pyplot as plt
import torch
from matplotlib.backend_tools import ToolToggleBase
from matplotlib.widgets import RadioButtons, Slider

from ..geometry.epipolar import T_to_F, generalized_epi_dist
from ..geometry.homography import sym_homography_error
from ..visualization.viz2d import (
    cm_ranking,
    cm_RdGn,
    draw_epipolar_line,
    get_line,
    plot_color_line_matches,
    plot_heatmaps,
    plot_keypoints,
    plot_lines,
    plot_matches,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    plt.rcParams["toolbar"] = "toolmanager"


class RadioHideTool(ToolToggleBase):
    """Show lines with a given gid."""

    default_keymap = "R"
    description = "Show by gid"
    default_toggled = False
    radio_group = "default"

    def __init__(
        self, *args, options=[], active=None, callback_fn=None, keymap="R", **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.f = 1.0
        self.options = options
        self.callback_fn = callback_fn
        self.active = self.options.index(active) if active else 0
        self.default_keymap = keymap

        self.enabled = self.default_toggled

    def build_radios(self):
        w = 0.2
        self.radios_ax = self.figure.add_axes([1.0 - w, 0.7, w, 0.2], zorder=1)
        # self.radios_ax = self.figure.add_axes([0.5-w/2, 1.0-0.2, w, 0.2], zorder=1)
        self.radios = RadioButtons(self.radios_ax, self.options, active=self.active)
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
    """Show lines with a given gid."""

    default_keymap = "t"
    description = "Show by gid"

    def __init__(self, *args, callback_fn=None, keymap="t", **kwargs):
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
            plot_keypoints([pred["keypoints0"][0], pred["keypoints1"][0]], axes=axes[i])


class LinePlot:
    plot_name = "lines"
    required_keys = ["lines0", "lines1"]

    def __init__(self, fig, axes, data, preds):
        for i, name in enumerate(preds):
            pred = preds[name]
            plot_lines([pred["lines0"][0], pred["lines1"][0]])


class KeypointRankingPlot:
    plot_name = "keypoint_ranking"
    required_keys = ["keypoints0", "keypoints1", "keypoint_scores0", "keypoint_scores1"]

    def __init__(self, fig, axes, data, preds):
        for i, name in enumerate(preds):
            pred = preds[name]
            kp0, kp1 = pred["keypoints0"][0], pred["keypoints1"][0]
            sc0, sc1 = pred["keypoint_scores0"][0], pred["keypoint_scores1"][0]

            plot_keypoints(
                [kp0, kp1], axes=axes[i], colors=[cm_ranking(sc0), cm_ranking(sc1)]
            )


class KeypointScoresPlot:
    plot_name = "keypoint_scores"
    required_keys = ["keypoints0", "keypoints1", "keypoint_scores0", "keypoint_scores1"]

    def __init__(self, fig, axes, data, preds):
        for i, name in enumerate(preds):
            pred = preds[name]
            kp0, kp1 = pred["keypoints0"][0], pred["keypoints1"][0]
            sc0, sc1 = pred["keypoint_scores0"][0], pred["keypoint_scores1"][0]
            plot_keypoints(
                [kp0, kp1], axes=axes[i], colors=[cm_RdGn(sc0), cm_RdGn(sc1)]
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
            self.artists += plot_heatmaps(heatmaps, axes=axes[i], cmap="rainbow")

    def clear(self):
        for x in self.artists:
            x.remove()


class ImagePlot:
    plot_name = "images"
    required_keys = ["view0", "view1"]

    def __init__(self, fig, axes, data, preds):
        pass


class MatchesPlot:
    plot_name = "matches"
    required_keys = ["keypoints0", "keypoints1", "matches0", "matching_scores0"]

    def __init__(self, fig, axes, data, preds):
        self.fig = fig
        self.sbpars = {
            k: v
            for k, v in vars(fig.subplotpars).items()
            if k in ["left", "right", "top", "bottom"]
        }

        for i, name in enumerate(preds):
            pred = preds[name]
            plot_keypoints(
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
            plot_matches(
                kpm0,
                kpm1,
                color=cm_RdGn(mscores).tolist(),
                axes=axes[i],
                labels=mscores,
                lw=0.5,
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
            plot_color_line_matches([m_lines0, m_lines1])


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
            plot_keypoints(
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
            plot_matches(
                kpm0,
                kpm1,
                color=cm_RdGn(correct).tolist(),
                axes=axes[i],
                labels=correct,
                lw=0.5,
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
            plot_color_line_matches([m_lines0, m_lines1])


class HomographyMatchesPlot:
    plot_name = "homography"
    required_keys = ["keypoints0", "keypoints1", "matches0", "H_0to1"]

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
            label="Homography Error",
            valmin=0,
            valmax=5,
            valinit=3.0,
            valstep=1.0,
        )
        self.range.on_changed(self.color_matches)

        for i, name in enumerate(preds):
            pred = preds[name]
            plot_keypoints(
                [pred["keypoints0"][0], pred["keypoints1"][0]],
                axes=axes[i],
                colors="blue",
            )
            kp0, kp1 = pred["keypoints0"][0], pred["keypoints1"][0]
            m0 = pred["matches0"][0]
            valid = m0 > -1
            kpm0 = kp0[valid]
            kpm1 = kp1[m0[valid]]
            errors = sym_homography_error(kpm0, kpm1, data["H_0to1"][0])
            plot_matches(
                kpm0,
                kpm1,
                color=cm_RdGn(errors < self.range.val).tolist(),
                axes=axes[i],
                labels=errors.numpy(),
                lw=0.5,
            )

    def clear(self):
        w, h = self.fig.get_size_inches()
        self.fig.set_size_inches(w, h / 1.1)
        self.fig.subplots_adjust(**self.sbpars)
        self.range_ax.remove()

    def color_matches(self, args):
        for line in self.fig.artists:
            label = line.get_label()
            line.set_color(cm_RdGn([float(label) < args])[0])


class EpipolarMatchesPlot:
    plot_name = "epipolar_matches"
    required_keys = ["keypoints0", "keypoints1", "matches0", "T_0to1", "view0", "view1"]

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

        for i, name in enumerate(preds):
            pred = preds[name]
            plot_keypoints(
                [pred["keypoints0"][0], pred["keypoints1"][0]],
                axes=axes[i],
                colors="blue",
            )
            kp0, kp1 = pred["keypoints0"][0], pred["keypoints1"][0]
            m0 = pred["matches0"][0]
            valid = m0 > -1
            kpm0 = kp0[valid]
            kpm1 = kp1[m0[valid]]

            errors = generalized_epi_dist(
                kpm0,
                kpm1,
                camera0,
                camera1,
                T_0to1,
                all=False,
                essential=False,
            )
            plot_matches(
                kpm0,
                kpm1,
                color=cm_RdGn(errors < self.range.val).tolist(),
                axes=axes[i],
                labels=errors.numpy(),
                lw=0.5,
            )

        self.F = T_to_F(camera0, camera1, T_0to1)

    def clear(self):
        w, h = self.fig.get_size_inches()
        self.fig.set_size_inches(w, h / 1.1)
        self.fig.subplots_adjust(**self.sbpars)
        self.range_ax.remove()

    def color_matches(self, args):
        for art in self.fig.artists:
            label = art.get_label()
            if label is not None:
                art.set_color(cm_RdGn([float(label) < args])[0])

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
                line0 = get_line(self.F.transpose(0, 1), xy2)[:, 0]
                line1 = get_line(self.F, xy1)[:, 0]
                art.epilines = [
                    draw_epipolar_line(line0, art.axesA),
                    draw_epipolar_line(line1, art.axesB),
                ]


__plot_dict__ = {
    obj.plot_name: obj
    for _, obj in inspect.getmembers(sys.modules[__name__], predicate=inspect.isclass)
    if hasattr(obj, "plot_name")
}
