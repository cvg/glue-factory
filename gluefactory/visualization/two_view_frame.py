import pprint

import numpy as np

from . import viz2d
from .tools import RadioHideTool, ToggleTool, __plot_dict__


class FormatPrinter(pprint.PrettyPrinter):
    def __init__(self, formats):
        super(FormatPrinter, self).__init__()
        self.formats = formats

    def format(self, obj, ctx, maxlvl, lvl):
        if type(obj) in self.formats:
            return self.formats[type(obj)] % obj, 1, 0
        return pprint.PrettyPrinter.format(self, obj, ctx, maxlvl, lvl)


class TwoViewFrame:
    default_conf = {
        "default": "matches",
        "summary_visible": False,
    }

    plot_dict = __plot_dict__

    childs = []

    event_to_image = [None, "color", "depth", "color+depth"]

    def __init__(self, conf, data, preds, title=None, event=1, summaries=None):
        self.conf = conf
        self.data = data
        self.preds = preds
        self.names = list(preds.keys())
        self.plot = self.event_to_image[event]
        self.summaries = summaries
        self.fig, self.axes, self.summary_arts = self.init_frame()
        if title is not None:
            self.fig.canvas.manager.set_window_title(title)

        keys = None
        for _, pred in preds.items():
            if keys is None:
                keys = set(pred.keys())
            else:
                keys = keys.intersection(pred.keys())
        keys = keys.union(data.keys())

        self.options = [
            k for k, v in self.plot_dict.items() if set(v.required_keys).issubset(keys)
        ]
        self.handle = None
        self.radios = self.fig.canvas.manager.toolmanager.add_tool(
            "switch plot",
            RadioHideTool,
            options=self.options,
            callback_fn=self.draw,
            active=conf.default,
            keymap="R",
        )

        self.toggle_summary = self.fig.canvas.manager.toolmanager.add_tool(
            "toggle summary",
            ToggleTool,
            toggled=self.conf.summary_visible,
            callback_fn=self.set_summary_visible,
            keymap="t",
        )

        if self.fig.canvas.manager.toolbar is not None:
            self.fig.canvas.manager.toolbar.add_tool("switch plot", "navigation")
        self.draw(conf.default)

    def init_frame(self):
        """initialize frame"""
        view0, view1 = self.data["view0"], self.data["view1"]
        if self.plot == "color" or self.plot == "color+depth":
            imgs = [
                view0["image"][0].permute(1, 2, 0),
                view1["image"][0].permute(1, 2, 0),
            ]
        elif self.plot == "depth":
            imgs = [view0["depth"][0], view1["depth"][0]]
        else:
            raise ValueError(self.plot)
        imgs = [imgs for _ in self.names]  # repeat for each model

        fig, axes = viz2d.plot_image_grid(imgs, return_fig=True, titles=None, figs=5)
        [viz2d.add_text(0, n, axes=axes[i]) for i, n in enumerate(self.names)]

        if (
            self.plot == "color+depth"
            and "depth" in view0.keys()
            and view0["depth"] is not None
        ):
            hmaps = [[view0["depth"][0], view1["depth"][0]] for _ in self.names]
            [
                viz2d.plot_heatmaps(hmaps[i], axes=axes[i], cmap="Spectral")
                for i, _ in enumerate(hmaps)
            ]

        fig.canvas.mpl_connect("pick_event", self.click_artist)
        if self.summaries is not None:
            formatter = FormatPrinter({np.float32: "%.4f", np.float64: "%.4f"})
            toggle_artists = [
                viz2d.add_text(
                    0,
                    formatter.pformat(self.summaries[n]),
                    axes=axes[i],
                    pos=(0.01, 0.01),
                    va="bottom",
                    backgroundcolor=(0, 0, 0, 0.5),
                    visible=self.conf.summary_visible,
                )
                for i, n in enumerate(self.names)
            ]
        else:
            toggle_artists = []
        return fig, axes, toggle_artists

    def draw(self, value):
        """redraw content in frame"""
        self.clear()
        self.conf.default = value
        self.handle = self.plot_dict[value](self.fig, self.axes, self.data, self.preds)
        return self.handle

    def clear(self):
        if self.handle is not None:
            try:
                self.handle.clear()
            except AttributeError:
                pass
        self.handle = None
        for row in self.axes:
            for ax in row:
                [li.remove() for li in ax.lines]
                [c.remove() for c in ax.collections]
        self.fig.artists.clear()
        self.fig.canvas.draw_idle()
        self.handle = None

    def click_artist(self, event):
        art = event.artist
        select = art.get_arrowstyle().arrow == "-"
        art.set_arrowstyle("<|-|>" if select else "-")
        if select:
            art.set_zorder(1)
        if hasattr(self.handle, "click_artist"):
            self.handle.click_artist(event)
        self.fig.canvas.draw_idle()

    def set_summary_visible(self, visible):
        self.conf.summary_visible = visible
        [s.set_visible(visible) for s in self.summary_arts]
        self.fig.canvas.draw_idle()
