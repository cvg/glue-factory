"""
Run this to tune the Weakfilter.
Example-Conf at: glue-factory/jpl_scripts/weak_filter_tuning_conf.yaml
Ex execution(assume venv activated and in glue-factory/jpl_scripts):
    python tune_weakfilter.py test_exp_name --conf=weak_filter_tuning_conf.yaml --num_s=3
"""

import argparse
import logging
import random
from pathlib import Path
from pprint import pprint

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import trange

from gluefactory.datasets import get_dataset
from gluefactory.models import get_model
from gluefactory.models.deeplsd_inference import DeepLSD
from gluefactory.settings import DATA_PATH

line_neighborhood = 5  # in px used to nortmalize/ denormalize df

logging.basicConfig(
    level=logging.INFO,  # Set the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s %(levelname)s: %(message)s",  # Set the log message format
    datefmt="%Y-%m-%d %H:%M:%S",  # Set the date/time format
)


default_conf = {
    # Add params for tuner here
    "tuner_conf": {
        # default conf of weak filter autotuner is in class for it
    },
    "dataset": {
        "name": "oxford_paris_mini_1view_jpldd",
        "reshape": 800,  # ex. 800
        "train_batch_size": 1,
        "split": "train",
        "val_batch_size": 1,
        "test_batch_size": 1,
        "multiscale_learning": {
            "do": False,
            "scales_list": [800, 600, 400],
            "scale_selection": "round-robin",  # random or round-robin
        },
        "load_features": {
            "do": True,
            "check_exists": True,
            "point_gt": {
                "data_keys": [
                    "superpoint_heatmap",
                    "gt_keypoints",
                    "gt_keypoints_scores",
                ],
                "use_score_heatmap": True,
            },
            "line_gt": {
                "data_keys": ["deeplsd_distance_field", "deeplsd_angle_field"],
                "enforce_threshold": 5.0,  # Enforce values in distance field to be no greater than this value
            },
        },
    },
    "jpl_model": {
        "name": "joint_point_line_extractor",
        "max_num_keypoints": 800,  # setting for training, for eval: -1
        "timeit": True,  # override timeit: False from BaseModel
        "line_df_decoder_channels": 32,
        "line_af_decoder_channels": 32,
        "line_detection": {
            "do": False,  # in the tuner we only need
        },
        "checkpoint": "/local/home/rkreft/shared_team_folder/outputs/training/oxparis_800_focal/checkpoint_best.tar",
    },
    "deeplsd_model": {
        "detect_lines": True,
        "line_detection_params": {
            "merge": False,
            "filtering": True,
            "grad_thresh": 3,
            "grad_nfa": True,
        },
        "weights": "DeepLSD/weights/deeplsd_md.tar",
    },
}

# Define utility methods


def get_deep_lsd_model(dlsd_conf, device="cuda"):
    deeplsd_conf = OmegaConf.create(dlsd_conf)
    ckpt_path = DATA_PATH / deeplsd_conf.weights
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    deeplsd_net = DeepLSD(deeplsd_conf)
    deeplsd_net.load_state_dict(ckpt["model"])
    deeplsd_net = deeplsd_net.to(device).eval()
    return deeplsd_net


def get_jpl_model(jpl_conf: dict, device: str):
    jpl_model = get_model("joint_point_line_extractor")(jpl_conf)
    jpl_model.to(device)
    jpl_model.eval()
    return jpl_model


def get_line_extractor_model(weak_filter_cfg: dict, samples: int, device: str):
    line_extractor_conf = {
        "max_point_size": 1500,  # max num of keypoints considered
        "min_line_length": 10,
        "max_line_length": None,
        "samples": [samples],
        "distance_map": weak_filter_cfg,
        "mlp_conf": None,
        "nms": False,
        "debug": False,
        "device": device,
    }
    line_extractor = get_model("lines.pold2_extractor")(line_extractor_conf).to(device)
    line_extractor.eval()
    return line_extractor


def hash_list_to_identifier(values_list):
    values_str = str(values_list)
    identifier = hash(values_str)
    return identifier


def count_common_lines(lines1, lines2, tolerance=1e-8):
    """
    Count the number of common lines between two sets of lines.
    Each line is represented by its endpoints.
    Line [A,B] is considered equal to [B,A].

    Args:
        lines1 (torch.Tensor): Tensor of shape (N, 2, 2) containing N lines
        lines2 (torch.Tensor): Tensor of shape (M, 2, 2) containing M lines
        tolerance (float): Numerical tolerance for floating point comparison

    Returns:
        int: Number of lines that appear in both sets
    """
    # Verify input shapes
    assert lines1.shape[1:] == (2, 2) and lines2.shape[1:] == (
        2,
        2,
    ), "Both tensors must have shape (N, 2, 2) and (M, 2, 2)"
    lines1 = lines1.float()
    lines2 = lines2.float()

    # Reshape lines1 for broadcasting
    # Shape: (N, 1, 2, 2)
    lines1_expanded = lines1.unsqueeze(1)

    # Reshape lines2 for broadcasting
    # Shape: (1, M, 2, 2)
    lines2_expanded = lines2.unsqueeze(0)

    # Compare each line from lines1 with each line from lines2
    # We need to check both possible endpoint orderings

    # First check: Compare lines as they are
    # Shape: (N, M, 2, 2)
    diff_direct = lines1_expanded - lines2_expanded

    # Second check: Compare with reversed endpoints
    # Reverse the endpoints of lines2
    lines2_reversed = lines2_expanded.flip(dims=[2])
    diff_reversed = lines1_expanded - lines2_reversed

    # Calculate distances for both cases
    # Shape: (N, M)
    distances_direct = torch.norm(diff_direct, dim=3).sum(dim=2)
    distances_reversed = torch.norm(diff_reversed, dim=3).sum(dim=2)

    # A line is common if either the direct or reversed comparison matches
    matches = torch.min(distances_direct, distances_reversed) <= tolerance

    # Count how many lines from lines1 have at least one match in lines2
    common_lines = torch.any(matches, dim=1).sum().item()

    return common_lines


class WeakFilterAutoTuner:
    """
    Automatically tunes the weakfilter based on the defined Metric.
    It takes the images described by eval_indices and tunes the weak filter according to performance on them.
    It varies parameters according to given ranges and starts with an initial conf.
    It stores the best configs and metrics for seen configurations.
    Thus it can be restored.
    Usage:
        - initialize
        - then run as many steps as you want calling the step() function. THis will descice for a novel combination of
          parameters and evaluate the performance on it.

    """

    weak_filter_default_conf = {
        "eval_indices": [
            100,
            101,
            102,
            104,
            105,
            506,
        ],  # image indices in the dataset that we use to tune the weakfilter
        "use_deeplsd_df": False,  # if true uses distance field from deeplsd. Otherwise from jpl model
        "use_deeplsd_af": False,  # if true uses angle field from deeplsd. Otherwise from jpl model
        "use_deeplsd_keypoints": True,  # if true uses line ep from deeplsd as kp. Otherwise from jpl model
        # now describe initial parameter value for weak-filter, as well as optimization range
        # for each param: paramname: [do-tune:bool, initial-value, min-value, max-value, step_size]
        # min and max values are inclusive
        "parameters": {
            "max_value": [False, 5, 5, 5, 0],  # this value is used to normalize
            "threshold": [
                True,
                0.5,
                0.05,
                0.95,
                0.05,
            ],  # threshold to have binary df=True
            "smooth_threshold": [True, 0.85, 0.05, 0.95, 0.05],
            "avg_filter_size": [False, 13, 13, 13, 0],
            "avg_filter_padding": [False, 6, 6, 6, 0],
            "avg_filter_stride": [False, 1, 1, 1, 0],
            "inlier_ratio": [True, 0.5, 0.05, 0.95, 0.05],
            "max_accepted_mean_value": [True, 0.3, 0.05, 0.95, 0.05],
            # samples is separate and not configured in the weak filter
            "samples": [True, 8, 4, 50, 2],
        },
        "metric": {
            "weight_num_lines": 0.25,
            "weight_common_lines": 15,
            "aggregation": "mean",  # options: 'mean', 'median'
        },
        "save_every_iter": 100,  # number of iterations after which checkpoints are saved periodically
        "debug": False,  # is true, stores images with detected lines for the best configs
        "debug_folder": "debug_images",  # folder based on current directory where debug images are stored
        "random_init": False,
        "restart_on_stagnation": False,
    }

    def __init__(
        self, conf, dataset, jpl_model, dlsd_model, exp_name, device="cpu"
    ) -> None:
        """
        Initializes the Autotuner. The Autotuner can load state if wanted.

        Args:
            dataset (_type_): iterable dataset to get images from
            eval_indices (_type_): indices of images to evaluate metrics on
            jpl_model (_type_, optional): our model. Defaults to None.
            device (str, optional): device to run on. Defaults to 'cpu'.
            exp_name (_type_, optional): experiment_name used to store state
        """
        # TODO: add state loading
        logging.info("Initialize...")
        self.conf = OmegaConf.merge(
            OmegaConf.create(WeakFilterAutoTuner.weak_filter_default_conf), conf
        )  # overall conf containing everything

        self.minimum_accepted_metric_value = (
            -99999
        )  # if metric value is smaller, skip this neighbor
        self.exp_name = exp_name
        self.device = device
        self.jpl_model = jpl_model  # jpl model to serve DF/AF
        self.dlsd_model = (
            dlsd_model  # deeplsd model to generate gt for metric and possibly AF/DF
        )
        self.dataset = dataset  # iterable dataset
        self.eval_indices = list(self.conf["eval_indices"])
        self.line_detector_to_be_tuned = None
        self.do_rand_init = self.conf["random_init"]
        self.do_restart_on_stagnation = self.conf["restart_on_stagnation"]
        # jpl model must exists if one of the elements needed for line detection is not from deeplsd
        self.jpl_model_used = not (
            self.conf["use_deeplsd_df"]
            and self.conf["use_deeplsd_af"]
            and self.conf["use_deeplsd_keypoints"]
        )
        assert self.jpl_model_used and (self.jpl_model is not None)

        # initialize eval images
        logging.info("Prepare data for eval-images...")
        self.eval_img_data = {}
        for idx in self.eval_indices:
            img = self.dataset[idx]
            df, af, kp, dlsd_lines = self.calculate_outputs(img)
            self.eval_img_data[str(idx)] = {
                "df": df.detach(),
                "af": af.detach(),
                "kp": kp.detach(),
                "dlsd_lines": dlsd_lines.detach(),
            }
        # delete models as they are not needed anymore
        del self.jpl_model
        del self.dlsd_model

        # set current config (to initial one)
        self.current_config = (
            self.generate_random_config()
            if self.do_rand_init
            else {k: v[1] for k, v in self.conf["parameters"].items()}
        )
        self.adjustable_params = {k for k, v in self.conf["parameters"].items() if v[0]}
        logging.info(f"Initial Info: {self.current_config}")
        # set current state
        self.num_iter = 0

        # create metrics state (ndarray or pandas df) and set best inices
        # each row gets an id(hash of cfg), metric, config-values() Order as in dict in default_conf
        self.results = np.zeros((0, 11))

        logging.info("Run evaluation for initial config...")
        # run 1st evaluation with initial config (this also sets current line detector)
        self.best_metric_value = self.calculate_metric_for_config(self.current_config)
        logging.info(f"Initial Metric-Value: {self.best_metric_value}")

    def get_top_k_confs_by_metric(self, topk: int = 5):
        """
        Return topk configs with their metrics. This is done after the search.
        """
        COL_IDX = 1  # at colidx 1 we have metric
        topk_indices = np.argpartition(self.results[:, COL_IDX], -topk)[-topk:]
        confs = []
        metrics = []
        for k in topk_indices:
            c = self.results[k, :]
            conf = dict(zip(self.conf["parameters"].keys(), c[2:]))
            confs.append(conf)
            metrics.append(c[1])
        return confs, metrics

    def generate_random_config(self):
        """
        Takes config to look at variable parameters and generates random one.
        Complet random.
        """
        params = self.conf["parameters"]
        conf = {k: v[1] for k, v in params.items()}
        for param in params.keys():
            if not params[param][0]:
                continue
            min = params[param][2]
            max = params[param][3]
            step = params[param][4]
            val = random.choice([x for x in np.arange(min, max + step, step)])
            conf[param] = float(np.round(val, 2))
        conf["samples"] = int(conf["samples"])
        return conf

    def calculate_outputs(self, img: torch.Tensor):
        """
        Depending on config settings, calculates distance field, angle_field and keypoints from dlsd or jpl model.
        In any case returns dlsd_lines.
        """
        img_size = img["image_size"]
        with torch.no_grad():
            img_torch = img["image"].to(device).unsqueeze(0)
            img = (img_torch[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            c_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            gray_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)
            inputs = {
                "image": torch.tensor(gray_img, dtype=torch.float, device=device)[
                    None, None
                ]
                / 255.0
            }
            deeplsd_output = self.dlsd_model(inputs)
            if self.jpl_model_used:
                output_jpl = self.jpl_model({"image": img_torch})
        df = (
            deeplsd_output["df"][0]
            if self.conf["use_deeplsd_df"]
            else output_jpl["line_distancefield"][0]
        )
        af = (
            deeplsd_output["line-level"][0]
            if self.conf["use_deeplsd_af"]
            else output_jpl["line_distancefield"][0]
        )
        deeplsd_lines = torch.tensor(deeplsd_output["lines"][0].astype(int))
        deeplsd_lines[:, :, 0] = torch.clamp(
            deeplsd_lines[:, :, 0], 0, img.shape[1] - 1
        )
        deeplsd_lines[:, :, 1] = torch.clamp(
            deeplsd_lines[:, :, 1], 0, img.shape[0] - 1
        )
        keypoints_deeplsd = torch.cat((deeplsd_lines[:, 0], deeplsd_lines[:, 1]))
        kp = (
            keypoints_deeplsd
            if self.conf["use_deeplsd_keypoints"]
            else output_jpl["keypoints"][0]
        )
        return (
            df.to(self.device),
            af.to(self.device),
            kp.to(self.device),
            deeplsd_lines.to(self.device),
        )

    def calculate_metric_single_image(
        self, line_extractor_out_lines: torch.Tensor, gt_dlsd_lines: torch.Tensor
    ) -> float:
        """
        Based on evaluation results and line gt calculate metric.
        Weighted sum of number of candidate lines + how many of dlsd lines were found.
        Higher Metric is better!

        Args:
            line_extractor_out_lines (torch.Tensor): Shape Nx2x2
            gt_dlsd_lines (torch.Tensor): Shape Mx2x2
        """
        num_lines = line_extractor_out_lines.shape[0]
        num_dlsd_lines = gt_dlsd_lines.shape[0]
        common_lines = count_common_lines(line_extractor_out_lines, gt_dlsd_lines)
        if num_lines == 0:
            return -99999999
        print(f"Num-Lines: {num_lines}")
        print(f"Num-Common-Lines: {common_lines}")
        more_lines_factor = 1 if num_lines >= num_dlsd_lines else 3
        count_lines_part = -(
            self.conf["metric"]["weight_num_lines"]
            * more_lines_factor
            * abs(num_lines - num_dlsd_lines)
        )
        common_lines_part = self.conf["metric"]["weight_common_lines"] * common_lines
        metric = count_lines_part + common_lines_part
        # print(f"Metric: {metric}")
        return metric

    def calculate_overall_metric(self, single_metrics: np.ndarray, mode="mean"):
        """
        Calculates metric for a config for all eval images and combines to one metric.

        For now it just computes mean or median depending on config
        """
        if mode == "mean":
            return np.mean(single_metrics)
        elif mode == "median":
            return np.median(single_metrics)
        else:
            raise NotImplementedError("Use option mean or median...")

    def save_state(self):
        """
        based on current status, saves array of all configs with their metric results to file.
        Methods to get best configs and images for them, are also containe din this script.
        """
        out_path = (
            Path(__file__).parent
            / "autotune_out"
            / self.exp_name
            / f"results_{self.num_iter}.npy"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path), self.results)

    def step(self):
        """
        We are using Combinatorial Optimization Algorithm, so we take the following approach:
            1. start with an initial config
            2. determine all neighbouring configs
            3. For each neighbour calculate metric
            4. choose neighbour with best metric value and set this as next current conf
            5. if no neighbour is better choose random move
        Ideas for future: simmulated annealing
        """
        # do we have to consider looking up whether conf value was already calculated?
        # -> no if we only go to better value neighbours
        neighbours = self.get_possible_neighbour_confs()
        # pprint(neighbours)
        best_neigh_conf = None
        best_neigh_metric = -999999999
        for neigh_conf in neighbours:
            metric = self.calculate_metric_for_config(neigh_conf)
            # store results and update best values
            if metric > best_neigh_metric:
                best_neigh_metric = metric
                best_neigh_conf = neigh_conf
        # Finally update current config to best one (TODO: here we could also do random moves)
        # now we exit on local optimum
        if (
            best_neigh_metric > self.best_metric_value
            and not best_neigh_metric < self.minimum_accepted_metric_value
        ):
            print(f"New Best cfg: {best_neigh_conf}")
            print(f"With Metric: {best_neigh_metric}")
            self.best_metric_value = best_neigh_metric
            self.current_config = best_neigh_conf
        elif self.do_restart_on_stagnation:
            # set current conf to random one
            logging.info("Stagnation... Random restart...")
            self.current_config = (
                self.generate_random_config()
            )  # TODO: add check that was not visited yet
            self.best_metric_value = -999999999
        else:
            self.save_state()
            print("Reached local optimum & dont restart on stagnation")
            print("3 best configs:\n")
            # print 5 best confs with metrics
            confs, metrics = self.get_top_k_confs_by_metric(topk=3)
            for conf, metric in zip(confs, metrics):
                print(f"metric: {metric}\nconf: {conf}\n")
            exit()

    def get_possible_neighbour_confs(self):
        """
        First determines which parameters are to vary.
        Then returns a list of tuples describing changes to current config.
        As neighbours we only consider one change at a time.
        """
        # check if conf already looked at if yes-ignore
        neighbours = []
        curr_conf = dict(self.current_config)
        for param_name in self.adjustable_params:
            curr_val = curr_conf[param_name]
            max_val = self.conf["parameters"][param_name][3]
            min_val = self.conf["parameters"][param_name][2]
            step_size = self.conf["parameters"][param_name][4]
            if curr_val - step_size >= min_val:
                n = (param_name, curr_val - step_size)
                new_conf = self.apply_conf_change_from_neighbour(curr_conf, n)
                hash_ident = hash_list_to_identifier(list(new_conf.values()))
                if not np.isin(self.results[:, 0], hash_ident).any().item():
                    neighbours.append(new_conf)
            if curr_val + step_size <= max_val:
                n = (param_name, curr_val + step_size)
                new_conf = self.apply_conf_change_from_neighbour(curr_conf, n)
                hash_ident = hash_list_to_identifier(list(new_conf.values()))
                if not np.isin(self.results[:, 0], hash_ident).any().item():
                    neighbours.append(new_conf)
        logging.info(f"Num-neighbours: {len(neighbours)}")
        return neighbours

    def apply_conf_change_from_neighbour(self, conf: dict, neighbour: tuple):
        """
        Takes config and applies indicated by tuple representing neighbour
        """
        n_conf = conf.copy()
        n_conf[neighbour[0]] = (
            float(np.round(neighbour[1], 2))
            if neighbour[0] != "samples"
            else int(neighbour[1])
        )
        return n_conf

    def save_to_results(self, metric, config):
        cfg_val_list = list(config.values())
        res_array = [hash_list_to_identifier(cfg_val_list), metric, *cfg_val_list]
        self.results = np.vstack([self.results, np.array(res_array)])

    def calculate_metric_for_config(self, config):
        """
        Evaluates Metric for current config. Stores results. Chooses new Config.
        """
        # set line extractor to current_conf
        del self.line_detector_to_be_tuned
        self.line_detector_to_be_tuned = get_line_extractor_model(
            config, int(config["samples"]), self.device
        )

        results = np.zeros(len(self.eval_indices))
        # run detection for all test images
        for i, idx in enumerate(self.eval_indices):
            data = self.eval_img_data[str(idx)]
            inp = {
                "points": data["kp"].clone(),
                "distance_map": data["df"].clone(),
                "angle_map": data["af"].clone(),
                "descriptors": torch.zeros((data["kp"].shape[0], 128)).cuda(),
            }
            out = self.line_detector_to_be_tuned(inp)
            res = self.calculate_metric_single_image(
                out["lines"], data["dlsd_lines"].clone()
            )
            results[i] = res
            if res < self.minimum_accepted_metric_value:
                # each image must have a minimum metric value
                break
        overall_metric = self.calculate_overall_metric(
            results, mode=self.conf["metric"]["aggregation"]
        )

        self.save_to_results(overall_metric, config)

        # update iter and set new conf
        self.num_iter += 1
        # return so next step decision logic does not need to query results
        return overall_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="-WeakFilter Autotuner-")
    # Add the 'restore' argument (optional string)
    parser.add_argument("exp_name", type=str)
    parser.add_argument("--conf", type=str, help="path to config file")
    # parser.add_argument('--restore', type=str, default=None, help='Path to a model checkpoint to restore')
    # Add the 'num_steps' argument (optional integer with a default value)
    parser.add_argument(
        "--num_steps", type=int, default=100, help="Number of training steps"
    )
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cuda")
    args = parser.parse_args()

    # Parse the command-line arguments
    default_conf = OmegaConf.create(default_conf)
    conf = OmegaConf.merge(default_conf, OmegaConf.load(args.conf))
    dset_conf = conf["dataset"]
    model_conf = conf["jpl_model"]
    dlsd_conf = conf["deeplsd_model"]
    tuner_conf = conf["tuner_conf"]

    exp_name = args.exp_name
    # restore_path = args.restore
    num_steps = args.num_steps

    # figure out device
    if args.device == "cuda":
        assert torch.cuda.is_available()
    elif args.device == "mps":
        assert torch.backends.mps.is_built()

    device = args.device
    print(f"Using Device: {device}")

    # Initialize Dataset
    dataset = get_dataset(dset_conf.name)(dset_conf)
    ds = dataset.get_dataset(dset_conf.split)

    # Initialize JPL and deeplsd Models
    jpl_model = get_jpl_model(model_conf, device=device)
    deeplsd_model = get_deep_lsd_model(dlsd_conf, device=device)

    # initialize Autotuner
    tuner = WeakFilterAutoTuner(
        tuner_conf, ds, jpl_model, deeplsd_model, exp_name, device
    )

    # for now do one step
    for step in trange(num_steps, desc="Running Optimization", unit="step"):
        tuner.step()
        tuner.save_state()
    confs, metrics = tuner.get_top_k_confs_by_metric(topk=3)
    for conf, metric in zip(confs, metrics):
        print(f"metric: {metric}\nconf: {conf}\n")
    logging.info("Done!")
