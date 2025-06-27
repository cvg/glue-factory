import torch
import numpy as np
from torch.utils.data.dataloader import default_collate

class HAFMencoder(object):
    def __init__(self, dis_th=10, ang_th=0):
        self.dis_th = dis_th
        self.ang_th = ang_th

    def __call__(self, annotations):
        targets = []
        metas = []
        batch_size = annotations['batch_size']
        stride = annotations['stride']
        for batch_id in range(batch_size):
            junctions = annotations['junctions'][batch_id].clone()[:, [1, 0]] / float(stride)

            width = annotations['width'] // stride
            height = annotations['height'] // stride
            edge_indices = annotations['line_map'][batch_id].triu().nonzero()

            t, m = self.encoding_single_image(junctions, edge_indices, height, width)

            targets.append(t)
            metas.append(m)

        return default_collate(targets), metas

    def adjacent_matrix(self, n, edges, device):
        mat = torch.zeros(n + 1, n + 1, dtype=torch.bool, device=device)
        if edges.size(0) > 0:
            mat[edges[:, 0], edges[:, 1]] += True
            mat[edges[:, 1], edges[:, 0]] += True
        return mat

    def lines2hafm(self, lines, height, width):
        device = lines.device
        if lines.shape[0] == 0:
            hafm_ang = torch.zeros((3, height, width), device=device)
            hafm_dis = torch.zeros((1, height, width), device=device)
            hafm_mask = torch.zeros((1, height, width), device=device)
            return torch.zeros((3, height, width), device=device), torch.zeros((1, height, width),
                                                                               device=device), torch.zeros(
                (1, height, width), device=device)

        lmap, _, _ = _C.encodels(lines, height, width, height, width, lines.size(0))
        dismap = torch.sqrt(lmap[0] ** 2 + lmap[1] ** 2)[None]

        def _normalize(inp):
            mag = torch.sqrt(inp[0] * inp[0] + inp[1] * inp[1])
            return inp / (mag + 1e-6)

        md_map = _normalize(lmap[:2])
        st_map = _normalize(lmap[2:4])
        ed_map = _normalize(lmap[4:])
        st_map = lmap[2:4]
        ed_map = lmap[4:]

        md_ = md_map.reshape(2, -1).t()
        st_ = st_map.reshape(2, -1).t()
        ed_ = ed_map.reshape(2, -1).t()
        Rt = torch.cat(
            (torch.cat((md_[:, None, None, 0], md_[:, None, None, 1]), dim=2),
             torch.cat((-md_[:, None, None, 1], md_[:, None, None, 0]), dim=2)), dim=1)
        R = torch.cat(
            (torch.cat((md_[:, None, None, 0], -md_[:, None, None, 1]), dim=2),
             torch.cat((md_[:, None, None, 1], md_[:, None, None, 0]), dim=2)), dim=1)
        # Rtst_ = torch.matmul(Rt, st_[:,:,None]).squeeze(-1).t()
        # Rted_ = torch.matmul(Rt, ed_[:,:,None]).squeeze(-1).t()
        Rtst_ = torch.bmm(Rt, st_[:, :, None]).squeeze(-1).t()
        Rted_ = torch.bmm(Rt, ed_[:, :, None]).squeeze(-1).t()
        swap_mask = (Rtst_[1] < 0) * (Rted_[1] > 0)
        pos_ = Rtst_.clone()
        neg_ = Rted_.clone()
        temp = pos_[:, swap_mask]
        pos_[:, swap_mask] = neg_[:, swap_mask]
        neg_[:, swap_mask] = temp

        pos_[0] = pos_[0].clamp(min=1e-9)
        pos_[1] = pos_[1].clamp(min=1e-9)
        neg_[0] = neg_[0].clamp(min=1e-9)
        neg_[1] = neg_[1].clamp(max=-1e-9)

        mask = (dismap.view(-1) <= self.dis_th).float()

        pos_map = pos_.reshape(-1, height, width)
        neg_map = neg_.reshape(-1, height, width)

        md_angle = torch.atan2(md_map[1], md_map[0])
        pos_angle = torch.atan2(pos_map[1], pos_map[0])
        neg_angle = torch.atan2(neg_map[1], neg_map[0])

        mask *= (pos_angle.reshape(-1) > self.ang_th * np.pi / 2.0)
        mask *= (neg_angle.reshape(-1) < -self.ang_th * np.pi / 2.0)

        pos_angle_n = pos_angle / (np.pi / 2)
        neg_angle_n = -neg_angle / (np.pi / 2)
        md_angle_n = md_angle / (np.pi * 2) + 0.5
        mask = mask.reshape(height, width)

        hafm_ang = torch.cat((md_angle_n[None], pos_angle_n[None], neg_angle_n[None],), dim=0)
        hafm_dis = dismap.clamp(max=self.dis_th) / self.dis_th
        mask = mask[None]
        return hafm_ang, hafm_dis, mask

    def encoding_single_image(self, junctions, edge_indices, height, width):
        device = junctions.device

        # jmap = torch.zeros((height,width),device=device)
        # joff = torch.zeros((2,height,width),device=device,dtype=torch.float32)
        jmap = np.zeros((height, width), dtype=np.float32)
        joff = np.zeros((2, height, width), dtype=np.float32)

        dx, dy = np.meshgrid(np.arange(width), np.arange(height))
        # gaussian = np.exp(-(dx**2+dy**2)/2.0/2.0**2)

        if junctions.shape[0] > 0:
            junctions_np = junctions.cpu().numpy()
            xint, yint = junctions_np[:, 0].astype(np.int32), junctions_np[:, 1].astype(np.int32)
            off_x = junctions_np[:, 0] - np.floor(junctions_np[:, 0]) - 0.5
            off_y = junctions_np[:, 1] - np.floor(junctions_np[:, 1]) - 0.5

            jmap[yint, xint] = 1  # = jmap[yint,xint] + 1
            joff[0, yint, xint] = off_x
            joff[1, yint, xint] = off_y

            lines = junctions[edge_indices].reshape(-1, 4)
            pos_mat = self.adjacent_matrix(junctions.size(0), edge_indices, device)
            labels = torch.ones((lines.shape[0],), device=device)
        else:
            lines = torch.empty((0, 4), device=device)
            pos_mat = None
            labels = None
        # for _x,_y in junctions.cpu().numpy():
        #     _map = np.exp(-((dx-_x)**2+(dy-_y)**2)/2.0/8.0**2)
        #     _map /= _map.max()
        #     jmap = np.maximum(jmap,_map)
        # import matplotlib.pyplot as plt
        # import pdb; pdb.set_trace()
        jmap = torch.from_numpy(jmap).to(device)
        joff = torch.from_numpy(joff).to(device)
        hafm_ang, hafm_dis, hafm_mask = self.lines2hafm(lines, height, width)

        target = {
            'jloc': jmap[None],
            'joff': joff,
            'md': hafm_ang,
            'dis': hafm_dis,
            'mask': hafm_mask
        }

        meta = {
            'junc': junctions,
            'lines': lines,
            'Lpos': pos_mat,
            'lpre': lines,
            'lpre_label': labels
        }
        return target, meta
