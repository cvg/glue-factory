import numpy as np
import torch
from kornia.geometry.homography import find_homography_dlt

from .. import robust_estimators
from ..geometry import depth, epipolar, gt_generation, homography, reconstruction
from ..utils import misc, tools, types


def check_keys_recursive(d, pattern):
    if isinstance(pattern, dict):
        {check_keys_recursive(d[k], v) for k, v in pattern.items()}
    else:
        for k in pattern:
            assert k in d.keys()


def get_matches_scores(kpts0, kpts1, matches0, mscores0):
    m0 = matches0 > -1
    m1 = matches0[m0]
    pts0 = kpts0[m0]
    pts1 = kpts1[m1]
    scores = mscores0[m0]
    return pts0, pts1, scores


def eval_per_batch_item(data: dict, pred: dict, eval_f, *args, **kwargs):
    # Batched data
    results = [
        eval_f(data_i, pred_i, *args, **kwargs)
        for data_i, pred_i in zip(misc.index_batch(data), misc.index_batch(pred))
    ]
    # Return a dictionary of lists with the evaluation of each item
    return {k: [r[k] for r in results] for k in results[0].keys()}


def eval_matches_epipolar(data: dict, pred: dict, essential: bool = True) -> dict:
    check_keys_recursive(data, ["view0", "view1", "T_0to1"])
    check_keys_recursive(
        pred, ["keypoints0", "keypoints1", "matches0", "matching_scores0"]
    )

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0, scores0 = pred["matches0"], pred["matching_scores0"]
    pts0, pts1, scores = get_matches_scores(kp0, kp1, m0, scores0)

    results = {}

    # match metrics
    n_epi_err = epipolar.generalized_epi_dist(
        pts0[None],
        pts1[None],
        data["view0"]["camera"],
        data["view1"]["camera"],
        data["T_0to1"],
        False,
        essential=essential,
    )[0]
    if essential:
        results["epi_prec@1e-4"] = (n_epi_err < 1e-4).float().mean().nan_to_num()
        results["epi_prec@5e-4"] = (n_epi_err < 5e-4).float().mean().nan_to_num()
        results["epi_prec@1e-3"] = (n_epi_err < 1e-3).float().mean().nan_to_num()
    else:
        results["epi_prec@1px"] = (n_epi_err < 1).float().mean().nan_to_num()
        results["epi_prec@3px"] = (n_epi_err < 3).float().mean().nan_to_num()
        results["epi_prec@5px"] = (n_epi_err < 5).float().mean().nan_to_num()

    results["num_matches"] = pts0.shape[0]
    results["num_keypoints"] = (kp0.shape[0] + kp1.shape[0]) / 2.0

    return results


def eval_matches_depth(data: dict, pred: dict) -> dict:
    check_keys_recursive(data, ["view0", "view1", "T_0to1"])
    check_keys_recursive(data["view0"], ["depth", "camera"])
    check_keys_recursive(data["view1"], ["depth", "camera"])
    check_keys_recursive(
        pred, ["keypoints0", "keypoints1", "matches0", "matching_scores0"]
    )

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0, scores0 = pred["matches0"], pred["matching_scores0"]
    pts0, pts1, _ = get_matches_scores(kp0, kp1, m0, scores0)

    camera0, camera1 = data["view0"]["camera"], data["view1"]["camera"]
    T_0to1 = data["T_0to1"]

    depth0 = data["view0"]["depth"]
    depth1 = data["view1"]["depth"]

    reproj_error, valid = depth.symmetric_reprojection_error(
        pts0[None],
        pts1[None],
        camera0,
        camera1,
        T_0to1,
        depth0,
        depth1,
    )
    reproj_error, valid = reproj_error[0], valid[0]

    results = {}
    reproj_error = reproj_error[valid].nan_to_num(nan=float("inf"))
    results["reproj_prec@1px"] = (reproj_error < 1).float().mean().nan_to_num().item()
    results["reproj_prec@3px"] = (reproj_error < 3).float().mean().nan_to_num().item()
    results["reproj_prec@5px"] = (reproj_error < 5).float().mean().nan_to_num().item()
    results["covisible"] = valid.float().sum().item()
    results["covisible_percent"] = valid.float().mean().item() * 100.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gt_pred = gt_generation.gt_matches_from_pose_depth(
        kp0[None].to(device),
        kp1[None].to(device),
        misc.batch_to_device(data, device),
        pos_th=3.0,
        neg_th=5.0,
    )

    def recall(m, gt_m):
        mask = (gt_m > -1).float()
        return ((m == gt_m) * mask).sum(1) / (1e-8 + mask.sum(1))

    results["gt_match_recall@3px"] = recall(
        pred["matches0"][None], gt_pred["matches0"].cpu()
    )[0].item()

    def precision(m, gt_m):
        mask = ((m > -1) & (gt_m >= -1)).float()
        return ((m == gt_m) * mask).sum(1) / (1e-8 + mask.sum(1))

    results["gt_match_precision@3px"] = precision(
        pred["matches0"][None], gt_pred["matches0"].cpu()
    )[0].item()
    return results


def eval_matches_homography(data: dict, pred: dict) -> dict:
    check_keys_recursive(data, ["H_0to1"])
    check_keys_recursive(
        pred, ["keypoints0", "keypoints1", "matches0", "matching_scores0"]
    )

    H_gt = data["H_0to1"]
    if H_gt.ndim > 2:
        return eval_per_batch_item(data, pred, eval_matches_homography)

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0, scores0 = pred["matches0"], pred["matching_scores0"]
    pts0, pts1, scores = get_matches_scores(kp0, kp1, m0, scores0)
    err = homography.sym_homography_error(pts0, pts1, H_gt)
    results = {}
    results["prec@1px"] = (err < 1).float().mean().nan_to_num().item()
    results["prec@3px"] = (err < 3).float().mean().nan_to_num().item()
    results["num_matches"] = pts0.shape[0]
    results["num_keypoints"] = (kp0.shape[0] + kp1.shape[0]) / 2.0
    return results


def eval_relative_pose_robust(data, pred, conf):
    check_keys_recursive(data, ["view0", "view1", "T_0to1"])
    check_keys_recursive(
        pred, ["keypoints0", "keypoints1", "matches0", "matching_scores0"]
    )

    T_gt = data["T_0to1"]
    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0, scores0 = pred["matches0"], pred["matching_scores0"]
    pts0, pts1, scores = get_matches_scores(kp0, kp1, m0, scores0)

    results = {}

    estimator = robust_estimators.load_estimator("relative_pose", conf["estimator"])(
        conf
    )
    data_ = {
        "m_kpts0": pts0,
        "m_kpts1": pts1,
        "camera0": data["view0"]["camera"][0],
        "camera1": data["view1"]["camera"][0],
    }
    est = estimator(data_)

    if not est["success"]:
        results["rel_pose_error"] = float("inf")
        results["ransac_inl"] = 0
        results["ransac_inl%"] = 0
    else:
        # R, t, inl = ret
        M: reconstruction.Pose = est["M_0to1"]
        inl = est["inliers"].numpy()
        t_error, r_error = M.angular_drdt(T_gt[0])
        results["rel_pose_error"] = max(r_error, t_error)
        results["ransac_inl"] = np.sum(inl)
        results["ransac_inl%"] = np.mean(inl)

    return results


def eval_homography_robust(data, pred, conf):
    H_gt = data["H_0to1"]
    if H_gt.ndim > 2:
        return eval_per_batch_item(data, pred, eval_relative_pose_robust, conf)

    estimator = robust_estimators.load_estimator("homography", conf["estimator"])(conf)

    data_ = {}
    if "keypoints0" in pred:
        kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
        m0, scores0 = pred["matches0"], pred["matching_scores0"]
        pts0, pts1, _ = get_matches_scores(kp0, kp1, m0, scores0)
        data_["m_kpts0"] = pts0
        data_["m_kpts1"] = pts1
    if "lines0" in pred:
        if "orig_lines0" in pred:
            lines0 = pred["orig_lines0"]
            lines1 = pred["orig_lines1"]
        else:
            lines0 = pred["lines0"]
            lines1 = pred["lines1"]
        m_lines0, m_lines1, _ = get_matches_scores(
            lines0, lines1, pred["line_matches0"], pred["line_matching_scores0"]
        )
        data_["m_lines0"] = m_lines0
        data_["m_lines1"] = m_lines1

    est = estimator(data_)
    if est["success"]:
        M = est["M_0to1"]
        error_r = homography.homography_corner_error(
            M, H_gt, data["view0"]["image_size"]
        ).item()
    else:
        error_r = float("inf")

    results = {}
    results["H_error_ransac"] = error_r
    if "inliers" in est:
        inl = est["inliers"]
        results["ransac_inl"] = inl.float().sum().item()
        results["ransac_inl%"] = inl.float().sum().item() / max(len(inl), 1)

    return results


def eval_homography_dlt(data, pred):
    H_gt = data["H_0to1"]
    H_inf = torch.ones_like(H_gt) * float("inf")

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0, scores0 = pred["matches0"], pred["matching_scores0"]
    pts0, pts1, scores = get_matches_scores(kp0, kp1, m0, scores0)
    scores = scores.to(pts0)
    results = {}
    try:
        if H_gt.ndim == 2:
            pts0, pts1, scores = pts0[None], pts1[None], scores[None]
        h_dlt = find_homography_dlt(pts0, pts1, scores)
        if H_gt.ndim == 2:
            h_dlt = h_dlt[0]
    except AssertionError:
        h_dlt = H_inf

    error_dlt = homography.homography_corner_error(
        h_dlt, H_gt, data["view0"]["image_size"]
    )
    results["H_error_dlt"] = error_dlt.item()
    return results


def eval_poses(pose_results, auc_ths, key, unit="°"):
    pose_aucs = {}
    best_th = -1
    for th, results_i in pose_results.items():
        pose_aucs[th] = tools.AUCMetric(auc_ths, results_i[key]).compute()
    mAAs = {k: np.mean(v) for k, v in pose_aucs.items()}
    best_th = max(mAAs, key=mAAs.get)

    if len(pose_aucs) > -1:
        print("Tested ransac setup with following results:")
        print("AUC", pose_aucs)
        print("mAA", mAAs)
        print("best threshold =", best_th)

    summaries = {}

    for i, ath in enumerate(auc_ths):
        summaries[f"{key}@{ath}{unit}"] = pose_aucs[best_th][i]
    summaries[f"{key}_mAA"] = mAAs[best_th]

    for k, v in pose_results[best_th].items():
        arr = np.array(v)
        if not np.issubdtype(np.array(v).dtype, np.number):
            continue
        summaries[f"m{k}"] = round(np.median(arr), 3)
    return summaries, best_th


def get_tp_fp_pts(pred_matches, gt_matches, pred_scores):
    """
    Computes the True Positives (TP), False positives (FP), the score associated
    to each match and the number of positives for a set of matches.
    """
    assert pred_matches.shape == pred_scores.shape
    ignore_mask = gt_matches != types.IGNORE_FEATURE
    pred_matches, gt_matches, pred_scores = (
        pred_matches[ignore_mask],
        gt_matches[ignore_mask],
        pred_scores[ignore_mask],
    )
    num_pos = np.sum(gt_matches != -1)
    pred_positives = pred_matches != -1
    tp = pred_matches[pred_positives] == gt_matches[pred_positives]
    fp = pred_matches[pred_positives] != gt_matches[pred_positives]
    scores = pred_scores[pred_positives]
    return tp, fp, scores, num_pos


def AP(tp, fp):
    recall = tp
    precision = tp / np.maximum(tp + fp, 1e-9)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])
    return ap


def aggregate_pr_results(results, suffix=""):
    tp_list = np.concatenate(results["tp" + suffix], axis=0)
    fp_list = np.concatenate(results["fp" + suffix], axis=0)
    scores_list = np.concatenate(results["scores" + suffix], axis=0)
    n_gt = max(results["num_pos" + suffix], 1)

    out = {}
    idx = np.argsort(scores_list)[::-1]
    tp_vals = np.cumsum(tp_list[idx]) / n_gt
    fp_vals = np.cumsum(fp_list[idx]) / n_gt
    out["curve_recall" + suffix] = tp_vals
    out["curve_precision" + suffix] = tp_vals / np.maximum(tp_vals + fp_vals, 1e-9)
    out["AP" + suffix] = AP(tp_vals, fp_vals) * 100
    return out
