import json

import h5py
import numpy as np
from omegaconf import OmegaConf


def load_eval(dir):
    summaries, results = {}, {}
    with h5py.File(str(dir / "results.h5"), "r") as hfile:
        for k in hfile.keys():
            r = np.array(hfile[k])
            if len(r.shape) < 3:
                results[k] = r
        for k, v in hfile.attrs.items():
            summaries[k] = v
    with open(dir / "summaries.json", "r") as f:
        s = json.load(f)
    summaries = {k: v if v is not None else np.nan for k, v in s.items()}
    return summaries, results


def save_eval(dir, summaries, figures, results):
    with h5py.File(str(dir / "results.h5"), "w") as hfile:
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(arr.dtype, np.number):
                arr = arr.astype("object")
            hfile.create_dataset(k, data=arr)
        # just to be safe, not used in practice
        for k, v in summaries.items():
            hfile.attrs[k] = v
    s = {
        k: float(v) if np.isfinite(v) else None
        for k, v in summaries.items()
        if not isinstance(v, list)
    }
    s = {**s, **{k: v for k, v in summaries.items() if isinstance(v, list)}}
    with open(dir / "summaries.json", "w") as f:
        json.dump(s, f, indent=4)

    for fig_name, fig in figures.items():
        fig.savefig(dir / f"{fig_name}.png")


def exists_eval(dir):
    return (dir / "results.h5").exists() and (dir / "summaries.json").exists()


class EvalPipeline:
    default_conf = {}

    export_keys = []
    optional_export_keys = []

    def __init__(self, conf):
        """Assumes"""
        self.default_conf = OmegaConf.create(self.default_conf)
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self._init(self.conf)

    def _init(self, conf):
        pass

    @classmethod
    def get_dataloader(self, data_conf=None):
        """Returns a data loader with samples for each eval datapoint"""
        raise NotImplementedError

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        """Export a prediction file for each eval datapoint"""
        raise NotImplementedError

    def run_eval(self, loader, pred_file):
        """Run the eval on cached predictions"""
        raise NotImplementedError

    def run(self, experiment_dir, model=None, overwrite=False, overwrite_eval=False):
        """Run export+eval loop"""
        self.save_conf(
            experiment_dir, overwrite=overwrite, overwrite_eval=overwrite_eval
        )
        pred_file = self.get_predictions(
            experiment_dir, model=model, overwrite=overwrite
        )

        f = {}
        if not exists_eval(experiment_dir) or overwrite_eval or overwrite:
            s, f, r = self.run_eval(self.get_dataloader(), pred_file)
            save_eval(experiment_dir, s, f, r)
        s, r = load_eval(experiment_dir)
        return s, f, r

    def save_conf(self, experiment_dir, overwrite=False, overwrite_eval=False):
        # store config
        conf_output_path = experiment_dir / "conf.yaml"
        if conf_output_path.exists():
            saved_conf = OmegaConf.load(conf_output_path)
            if (saved_conf.data != self.conf.data) or (
                saved_conf.model != self.conf.model
            ):
                assert (
                    overwrite
                ), "configs changed, add --overwrite to rerun experiment with new conf"
            if saved_conf.eval != self.conf.eval:
                assert (
                    overwrite or overwrite_eval
                ), "eval configs changed, add --overwrite_eval to rerun evaluation"
        OmegaConf.save(self.conf, experiment_dir / "conf.yaml")
