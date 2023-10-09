import functools
import traceback
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from omegaconf import OmegaConf

from ..datasets.base_dataset import collate

# from ..eval.export_predictions import load_predictions
from ..models.cache_loader import CacheLoader
from .tools import RadioHideTool


class GlobalFrame:
    default_conf = {
        "x": "???",
        "y": "???",
        "diff": False,
        "child": {},
        "remove_outliers": False,
    }

    child_frame = None  # MatchFrame

    childs = []

    lines = []

    scatters = {}

    def __init__(
        self, conf, results, loader, predictions, title=None, child_frame=None
    ):
        self.child_frame = child_frame
        if self.child_frame is not None:
            # We do NOT merge inside the child frame to keep settings across figs
            self.default_conf["child"] = self.child_frame.default_conf

        self.conf = OmegaConf.merge(self.default_conf, conf)
        self.results = results
        self.loader = loader
        self.predictions = predictions
        self.metrics = set()
        for k, v in results.items():
            self.metrics.update(v.keys())
        self.metrics = sorted(list(self.metrics))

        self.conf.x = conf["x"] if conf["x"] else self.metrics[0]
        self.conf.y = conf["y"] if conf["y"] else self.metrics[1]

        assert self.conf.x in self.metrics
        assert self.conf.y in self.metrics

        self.names = list(results)
        self.fig, self.axes = self.init_frame()
        if title is not None:
            self.fig.canvas.manager.set_window_title(title)

        self.xradios = self.fig.canvas.manager.toolmanager.add_tool(
            "x",
            RadioHideTool,
            options=self.metrics,
            callback_fn=self.update_x,
            active=self.conf.x,
            keymap="x",
        )

        self.yradios = self.fig.canvas.manager.toolmanager.add_tool(
            "y",
            RadioHideTool,
            options=self.metrics,
            callback_fn=self.update_y,
            active=self.conf.y,
            keymap="y",
        )
        if self.fig.canvas.manager.toolbar is not None:
            self.fig.canvas.manager.toolbar.add_tool("x", "navigation")
            self.fig.canvas.manager.toolbar.add_tool("y", "navigation")

    def init_frame(self):
        """initialize frame"""
        fig, ax = plt.subplots()
        ax.set_title("click on points")
        diffb_ax = fig.add_axes([0.01, 0.02, 0.12, 0.06])
        self.diffb = Button(diffb_ax, label="diff_only")
        self.diffb.on_clicked(self.diff_clicked)
        fig.canvas.mpl_connect("pick_event", self.on_scatter_pick)
        fig.canvas.mpl_connect("motion_notify_event", self.hover)
        return fig, ax

    def draw(self):
        """redraw content in frame"""
        self.scatters = {}
        self.axes.clear()
        self.axes.set_xlabel(self.conf.x)
        self.axes.set_ylabel(self.conf.y)

        refx = 0.0
        refy = 0.0
        x_cat = isinstance(self.results[self.names[0]][self.conf.x][0], (bytes, str))
        y_cat = isinstance(self.results[self.names[0]][self.conf.y][0], (bytes, str))

        if self.conf.diff:
            if not x_cat:
                refx = np.array(self.results[self.names[0]][self.conf.x])
            if not y_cat:
                refy = np.array(self.results[self.names[0]][self.conf.y])
        for name in list(self.results.keys()):
            x = np.array(self.results[name][self.conf.x])
            y = np.array(self.results[name][self.conf.y])

            if x_cat and np.char.isdigit(x.astype(str)).all():
                x = x.astype(int)
            if y_cat and np.char.isdigit(y.astype(str)).all():
                y = y.astype(int)

            x = x if x_cat else x - refx
            y = y if y_cat else y - refy

            (s,) = self.axes.plot(
                x, y, "o", markersize=3, label=name, picker=True, pickradius=5
            )
            self.scatters[name] = s

            if x_cat and not y_cat:
                xunique, ind, xinv, xbin = np.unique(
                    x, return_inverse=True, return_counts=True, return_index=True
                )
                ybin = np.bincount(xinv, weights=y)
                sort_ax = np.argsort(ind)
                self.axes.step(
                    xunique[sort_ax],
                    (ybin / xbin)[sort_ax],
                    where="mid",
                    color=s.get_color(),
                )

            if not x_cat:
                xavg = np.nan_to_num(x).mean()
                self.axes.axvline(xavg, c=s.get_color(), zorder=1, alpha=1.0)
                xmed = np.median(x - refx)
                self.axes.axvline(
                    xmed,
                    c=s.get_color(),
                    zorder=0,
                    alpha=0.5,
                    linestyle="dashed",
                    visible=False,
                )

            if not y_cat:
                yavg = np.nan_to_num(y).mean()
                self.axes.axhline(yavg, c=s.get_color(), zorder=1, alpha=0.5)
                ymed = np.median(y - refy)
                self.axes.axhline(
                    ymed,
                    c=s.get_color(),
                    zorder=0,
                    alpha=0.5,
                    linestyle="dashed",
                    visible=False,
                )
            if x_cat and x.dtype == object and xunique.shape[0] > 5:
                self.axes.set_xticklabels(xunique[sort_ax], rotation=90)
        self.axes.legend()

    def on_scatter_pick(self, handle):
        try:
            art = handle.artist
            try:
                event = handle.mouseevent.button.value
            except AttributeError:
                return
            name = art.get_label()
            ind = handle.ind[0]
            # draw lines
            self.spawn_child(name, ind, event=event)
        except Exception:
            traceback.print_exc()
            exit(0)

    def spawn_child(self, model_name, ind, event=None):
        [line.remove() for line in self.lines]
        self.lines = []

        x_source = self.scatters[model_name].get_xdata()[ind]
        y_source = self.scatters[model_name].get_ydata()[ind]
        for oname in self.names:
            xn = self.scatters[oname].get_xdata()[ind]
            yn = self.scatters[oname].get_ydata()[ind]

            (ln,) = self.axes.plot([x_source, xn], [y_source, yn], "r")
            self.lines.append(ln)

        self.fig.canvas.draw_idle()

        if self.child_frame is None:
            return

        data = collate([self.loader.dataset[ind]])

        preds = {}

        for name, pfile in self.predictions.items():
            preds[name] = CacheLoader({"path": str(pfile), "add_data_path": False})(
                data
            )
        summaries_i = {
            name: {k: v[ind] for k, v in res.items() if k != "names"}
            for name, res in self.results.items()
        }
        frame = self.child_frame(
            self.conf.child,
            deepcopy(data),
            preds,
            title=str(data["name"][0]),
            event=event,
            summaries=summaries_i,
        )

        frame.fig.canvas.mpl_connect(
            "key_press_event",
            functools.partial(
                self.on_childframe_key_event, frame=frame, ind=ind, event=event
            ),
        )
        self.childs.append(frame)
        # if plt.rcParams['backend'] == 'webagg':
        #     self.fig.canvas.manager_class.refresh_all()
        self.childs[-1].fig.show()

    def hover(self, event):
        if event.inaxes == self.axes:
            for _, s in self.scatters.items():
                cont, ind = s.contains(event)
                if cont:
                    ind = ind["ind"][0]
                    xdata, ydata = s.get_data()
                    [line.remove() for line in self.lines]
                    self.lines = []

                    for oname in self.names:
                        xn = self.scatters[oname].get_xdata()[ind]
                        yn = self.scatters[oname].get_ydata()[ind]

                        (ln,) = self.axes.plot(
                            [xdata[ind], xn],
                            [ydata[ind], yn],
                            "black",
                            zorder=0,
                            alpha=0.5,
                        )
                        self.lines.append(ln)
                    self.fig.canvas.draw_idle()
                    break

    def diff_clicked(self, args):
        self.conf.diff = not self.conf.diff
        self.draw()
        self.fig.canvas.draw_idle()

    def update_x(self, x):
        self.conf.x = x
        self.draw()

    def update_y(self, y):
        self.conf.y = y
        self.draw()

    def on_childframe_key_event(self, key_event, frame, ind, event):
        if key_event.key == "delete":
            plt.close(frame.fig)
            self.childs.remove(frame)
        elif key_event.key in ["left", "right", "shift+left", "shift+right"]:
            key = key_event.key
            if key.startswith("shift+"):
                key = key.replace("shift+", "")
            else:
                plt.close(frame.fig)
                self.childs.remove(frame)
            new_ind = ind + 1 if key_event.key == "right" else ind - 1
            self.spawn_child(
                self.names[0],
                new_ind % len(self.loader),
                event=event,
            )
