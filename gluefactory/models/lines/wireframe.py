import torch
from sklearn.cluster import DBSCAN

from .. import get_model
from ..base_model import BaseModel


def sample_descriptors_corner_conv(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = keypoints / (keypoints.new_tensor([w, h]) * s)
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=False
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


def lines_to_wireframe(
    lines, line_scores, all_descs, s, nms_radius, force_num_lines, max_num_lines
):
    """Given a set of lines, their score and dense descriptors,
        merge close-by endpoints and compute a wireframe defined by
        its junctions and connectivity.
    Returns:
        junctions: list of [num_junc, 2] tensors listing all wireframe junctions
        junc_scores: list of [num_junc] tensors with the junction score
        junc_descs: list of [dim, num_junc] tensors with the junction descriptors
        connectivity: list of [num_junc, num_junc] bool arrays with True when 2
        junctions are connected
        new_lines: the new set of [b_size, num_lines, 2, 2] lines
        lines_junc_idx: a [b_size, num_lines, 2] tensor with the indices of the
        junctions of each endpoint
        num_true_junctions: a list of the number of valid junctions for each image
        in the batch, i.e. before filling with random ones
    """
    b_size, _, h, w = all_descs.shape
    device = lines.device
    h, w = h * s, w * s
    endpoints = lines.reshape(b_size, -1, 2)

    (
        junctions,
        junc_scores,
        connectivity,
        new_lines,
        lines_junc_idx,
        num_true_junctions,
    ) = ([], [], [], [], [], [])
    for bs in range(b_size):
        # Cluster the junctions that are close-by
        db = DBSCAN(eps=nms_radius, min_samples=1).fit(endpoints[bs].cpu().numpy())
        clusters = db.labels_
        n_clusters = len(set(clusters))
        num_true_junctions.append(n_clusters)

        # Compute the average junction and score for each cluster
        clusters = torch.tensor(clusters, dtype=torch.long, device=device)
        new_junc = torch.zeros(n_clusters, 2, dtype=torch.float, device=device)
        new_junc.scatter_reduce_(
            0,
            clusters[:, None].repeat(1, 2),
            endpoints[bs],
            reduce="mean",
            include_self=False,
        )
        junctions.append(new_junc)
        new_scores = torch.zeros(n_clusters, dtype=torch.float, device=device)
        new_scores.scatter_reduce_(
            0,
            clusters,
            torch.repeat_interleave(line_scores[bs], 2),
            reduce="mean",
            include_self=False,
        )
        junc_scores.append(new_scores)

        # Compute the new lines
        new_lines.append(junctions[-1][clusters].reshape(-1, 2, 2))
        lines_junc_idx.append(clusters.reshape(-1, 2))

        if force_num_lines:
            # Add random junctions (with no connectivity)
            missing = max_num_lines * 2 - len(junctions[-1])
            junctions[-1] = torch.cat(
                [
                    junctions[-1],
                    torch.rand(missing, 2).to(lines)
                    * lines.new_tensor([[w - 1, h - 1]]),
                ],
                dim=0,
            )
            junc_scores[-1] = torch.cat(
                [junc_scores[-1], torch.zeros(missing).to(lines)], dim=0
            )

            junc_connect = torch.eye(max_num_lines * 2, dtype=torch.bool, device=device)
            pairs = clusters.reshape(-1, 2)  # these pairs are connected by a line
            junc_connect[pairs[:, 0], pairs[:, 1]] = True
            junc_connect[pairs[:, 1], pairs[:, 0]] = True
            connectivity.append(junc_connect)
        else:
            # Compute the junction connectivity
            junc_connect = torch.eye(n_clusters, dtype=torch.bool, device=device)
            pairs = clusters.reshape(-1, 2)  # these pairs are connected by a line
            junc_connect[pairs[:, 0], pairs[:, 1]] = True
            junc_connect[pairs[:, 1], pairs[:, 0]] = True
            connectivity.append(junc_connect)

    junctions = torch.stack(junctions, dim=0)
    new_lines = torch.stack(new_lines, dim=0)
    lines_junc_idx = torch.stack(lines_junc_idx, dim=0)

    # Interpolate the new junction descriptors
    junc_descs = sample_descriptors_corner_conv(junctions, all_descs, s).mT

    return (
        junctions,
        junc_scores,
        junc_descs,
        connectivity,
        new_lines,
        lines_junc_idx,
        num_true_junctions,
    )


class WireframeExtractor(BaseModel):
    default_conf = {
        "point_extractor": {
            "name": None,
            "trainable": False,
            "dense_outputs": True,
            "max_num_keypoints": None,
            "force_num_keypoints": False,
        },
        "line_extractor": {
            "name": None,
            "trainable": False,
            "max_num_lines": None,
            "force_num_lines": False,
            "min_length": 15,
        },
        "wireframe_params": {
            "merge_points": True,
            "merge_line_endpoints": True,
            "nms_radius": 3,
        },
    }
    required_data_keys = ["image"]

    def _init(self, conf):
        self.point_extractor = get_model(self.conf.point_extractor.name)(
            self.conf.point_extractor
        )
        self.line_extractor = get_model(self.conf.line_extractor.name)(
            self.conf.line_extractor
        )

    def _forward(self, data):
        b_size, _, h, w = data["image"].shape
        device = data["image"].device

        if (
            not self.conf.point_extractor.force_num_keypoints
            or not self.conf.line_extractor.force_num_lines
        ):
            assert b_size == 1, "Only batch size of 1 accepted for non padded inputs"

        # Line detection
        pred = self.line_extractor(data)
        if pred["line_scores"].shape[-1] != 0:
            pred["line_scores"] /= pred["line_scores"].max(dim=1)[0][:, None] + 1e-8

        # Keypoint prediction
        pred = {**pred, **self.point_extractor(data)}
        assert (
            "dense_descriptors" in pred
        ), "The KP extractor should return dense descriptors"
        s_desc = data["image"].shape[2] // pred["dense_descriptors"].shape[2]

        # Remove keypoints that are too close to line endpoints
        if self.conf.wireframe_params.merge_points:
            line_endpts = pred["lines"].reshape(b_size, -1, 2)
            dist_pt_lines = torch.norm(
                pred["keypoints"][:, :, None] - line_endpts[:, None], dim=-1
            )
            # For each keypoint, mark it as valid or to remove
            pts_to_remove = torch.any(
                dist_pt_lines < self.conf.wireframe_params.nms_radius, dim=2
            )
            if self.conf.point_extractor.force_num_keypoints:
                # Replace the points with random ones
                num_to_remove = pts_to_remove.int().sum().item()
                pred["keypoints"][pts_to_remove] = torch.rand(
                    num_to_remove, 2, device=device
                ) * pred["keypoints"].new_tensor([[w - 1, h - 1]])
                pred["keypoint_scores"][pts_to_remove] = 0
                for bs in range(b_size):
                    descrs = sample_descriptors_corner_conv(
                        pred["keypoints"][bs][pts_to_remove[bs]][None],
                        pred["dense_descriptors"][bs][None],
                        s_desc,
                    )
                    pred["descriptors"][bs][pts_to_remove[bs]] = descrs[0].T
            else:
                # Simply remove them (we assume batch_size = 1 here)
                assert len(pred["keypoints"]) == 1
                pred["keypoints"] = pred["keypoints"][0][~pts_to_remove[0]][None]
                pred["keypoint_scores"] = pred["keypoint_scores"][0][~pts_to_remove[0]][
                    None
                ]
                pred["descriptors"] = pred["descriptors"][0][~pts_to_remove[0]][None]

        # Connect the lines together to form a wireframe
        orig_lines = pred["lines"].clone()
        if (
            self.conf.wireframe_params.merge_line_endpoints
            and len(pred["lines"][0]) > 0
        ):
            # Merge first close-by endpoints to connect lines
            (
                line_points,
                line_pts_scores,
                line_descs,
                line_association,
                pred["lines"],
                lines_junc_idx,
                n_true_junctions,
            ) = lines_to_wireframe(
                pred["lines"],
                pred["line_scores"],
                pred["dense_descriptors"],
                s=s_desc,
                nms_radius=self.conf.wireframe_params.nms_radius,
                force_num_lines=self.conf.line_extractor.force_num_lines,
                max_num_lines=self.conf.line_extractor.max_num_lines,
            )

            # Add the keypoints to the junctions and fill the rest with random keypoints
            (all_points, all_scores, all_descs, pl_associativity) = [], [], [], []
            for bs in range(b_size):
                all_points.append(
                    torch.cat([line_points[bs], pred["keypoints"][bs]], dim=0)
                )
                all_scores.append(
                    torch.cat([line_pts_scores[bs], pred["keypoint_scores"][bs]], dim=0)
                )
                all_descs.append(
                    torch.cat([line_descs[bs], pred["descriptors"][bs]], dim=0)
                )

                associativity = torch.eye(
                    len(all_points[-1]), dtype=torch.bool, device=device
                )
                associativity[: n_true_junctions[bs], : n_true_junctions[bs]] = (
                    line_association[bs][: n_true_junctions[bs], : n_true_junctions[bs]]
                )
                pl_associativity.append(associativity)

            all_points = torch.stack(all_points, dim=0)
            all_scores = torch.stack(all_scores, dim=0)
            all_descs = torch.stack(all_descs, dim=0)
            pl_associativity = torch.stack(pl_associativity, dim=0)
        else:
            # Lines are independent
            all_points = torch.cat(
                [pred["lines"].reshape(b_size, -1, 2), pred["keypoints"]], dim=1
            )
            n_pts = all_points.shape[1]
            num_lines = pred["lines"].shape[1]
            n_true_junctions = [num_lines * 2] * b_size
            all_scores = torch.cat(
                [
                    torch.repeat_interleave(pred["line_scores"], 2, dim=1),
                    pred["keypoint_scores"],
                ],
                dim=1,
            )
            line_descs = sample_descriptors_corner_conv(
                pred["lines"].reshape(b_size, -1, 2), pred["dense_descriptors"], s_desc
            ).mT  # [B, n_lines * 2, desc_dim]
            all_descs = torch.cat([line_descs, pred["descriptors"]], dim=1)
            pl_associativity = torch.eye(n_pts, dtype=torch.bool, device=device)[
                None
            ].repeat(b_size, 1, 1)
            lines_junc_idx = (
                torch.arange(num_lines * 2, device=device)
                .reshape(1, -1, 2)
                .repeat(b_size, 1, 1)
            )

        del pred["dense_descriptors"]  # Remove dense descriptors to save memory
        torch.cuda.empty_cache()

        pred["keypoints"] = all_points
        pred["keypoint_scores"] = all_scores
        pred["descriptors"] = all_descs
        pred["pl_associativity"] = pl_associativity
        pred["num_junctions"] = torch.tensor(n_true_junctions)
        pred["orig_lines"] = orig_lines
        pred["lines_junc_idx"] = lines_junc_idx
        return pred

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, _pred, _data):
        return {}
