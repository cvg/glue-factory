import numpy as np
import torch
from gluefactory.models.extractors.jpldd.line_utils import merge_lines


def create_line_candidates(keypoints: torch.Tensor) -> torch.Tensor:
    junctions = torch.zeros_like(keypoints).to(keypoints.device)
    junctions[:, 0], junctions[:, 1] = keypoints[:, 1].clone(), keypoints[:, 0].clone()
    lines = torch.hstack([
        torch.cartesian_prod(junctions[:, 0], junctions[:, 0]),
        torch.cartesian_prod(junctions[:, 1], junctions[:, 1]),
    ]).reshape((-1, 2, 2))
    return lines


def fine_line_filter(
    lines: torch.Tensor, points: torch.Tensor,angles: torch.Tensor,direction:torch.Tensor, df_thresh:int=5, 
    inlier_thresh:float=0.7, a_diff_thresh:float=np.pi/20,a_std_thresh:float=np.pi/30,
    a_inlier_thresh:float=0.5,merge:bool=False,merge_thresh:int=4
) -> torch.Tensor:
    '''
    lines: (N, 2, 2) -> each of the N elements: [[x1, x2], [y1, y2]]
    '''

    # each point with distance field lower than threshold is an inlier
    inlier_indices = points < df_thresh
    inlier_ratio = inlier_indices.sum(dim=-1).float() / inlier_indices.shape[-1]
    
    # same thing just with angles
    valid_angles = angles
    inlier_angle_indices = torch.remainder(torch.abs(valid_angles[0] - direction.unsqueeze(-1)), np.pi).unsqueeze(0) < a_diff_thresh
    angle_inlier_ratio = inlier_angle_indices.sum(dim=-1).float() / inlier_angle_indices.shape[-1]
    
    # Most time is spent on finding local minimum. The higher the ratio is the more points it should take into consideration
    # The higher the sample rate, the longer it takes

    crit1 = points.mean(dim=-1) < df_thresh
    crit2 = valid_angles.std(dim=-1) < a_std_thresh
    crit3 = inlier_ratio > inlier_thresh
    crit4 = torch.remainder(torch.abs(valid_angles.mean(dim=-1) - direction), np.pi) < a_diff_thresh
    crit5 = angle_inlier_ratio > a_inlier_thresh

    validity = crit1 & crit2 & crit3 & crit4 & crit5
    lines = lines.unsqueeze(0)[validity]
    
    if merge:
        new_lines = lines.mT
        merged_lines = merge_lines(new_lines, thresh=merge_thresh, overlap_thresh=0.).float()
        lines = merged_lines.mT
    start_x = lines[:, 1, 0]
    start_y = lines[:, 0, 0]
    end_x = lines[:, 1, 1]
    end_y = lines[:, 0, 1]

    lines = torch.stack([torch.stack([start_x, start_y], dim=1), torch.stack([end_x, end_y], dim=1)], dim=1)
    return lines

def coarse_line_filter(
    lines: torch.Tensor, df:torch.Tensor,af:torch.Tensor,n_samples:int=10, df_thresh:int=5,
    a_diff_thresh:float=np.pi/20,a_std_thresh:float=np.pi/30,
    a_inlier_thresh:float=0.5,min_len:int=5,max_len:int=100
) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    '''
    lines: (N, 2, 2) -> each of the N elements: [[x1, x2], [y1, y2]]
    '''

    sq_line_len = (lines[:, 0, 0] - lines[:, 0, 1])**2 + (lines[:, 1, 0] - lines[:, 1, 1])**2
    # It seems like this criterion is not needed because in these cases the line length would be 0 anyway
    #validity = (lines[:, 0, 0] != lines[:, 0, 1]) & (lines[:, 1, 0] != lines[:, 1, 1])
    validity = (sq_line_len >= (min_len ** 2)) & (sq_line_len <= (max_len ** 2))
    lines = lines[validity]

    offsets = torch.linspace(0, 1, n_samples).view(1, 1, -1).to(lines.device)

    # (n_points,n_samples) for x and y respectively
    xs = lines[:, 0, :1] + (lines[:, 0, 1:] - lines[:, 0, :1]) * offsets
    xs = torch.round(xs).long()
    ys = lines[:, 1, :1] + (lines[:, 1, 1:] - lines[:, 1, :1]) * offsets
    ys = torch.round(ys).long()
    
    # Move sample points to respective point in grid with lowest line distance field value
    points = df[xs, ys]
    angles = af[xs, ys]

    # Get line direction
    slope = (lines[:, 0, 1] - lines[:, 0, 0]) / (lines[:, 1, 1] - lines[:, 1, 0] + 1e-10)
    direction = torch.remainder(torch.atan(slope), torch.pi)

    # each point with distance field lower than threshold is an inlier
    
    # same thing just with angles
    valid_angles = angles
    inlier_angle_indices = torch.remainder(torch.abs(valid_angles[0] - direction.unsqueeze(-1)), np.pi).unsqueeze(0) < a_diff_thresh
    angle_inlier_ratio = inlier_angle_indices.sum(dim=-1).float() / inlier_angle_indices.shape[-1]
    
    # Most time is spent on finding local minimum. The higher the ratio is the more points it should take into consideration
    # The higher the sample rate, the longer it takes

    crit1 = points.mean(dim=-1) < df_thresh
    crit2 = valid_angles.std(dim=-1) < a_std_thresh
    #crit3 = inlier_ratio > inlier_thresh
    crit4 = torch.remainder(torch.abs(valid_angles.mean(dim=-1) - direction), np.pi) < a_diff_thresh
    crit5 = angle_inlier_ratio > a_inlier_thresh

    validity = crit1 & crit2 & crit4 & crit5 
    lines = lines.unsqueeze(0)[validity]
    points = points[validity].unsqueeze(0)
    angles = angles[validity].unsqueeze(0)
    direction = direction.unsqueeze(0)[validity]
    return lines,points,angles,direction


def detect_jpldd_lines(
    df: np.array, af: np.array, keypoints: np.array,img_size,n_samples:int=10, df_thresh:int=2, 
    inlier_thresh:float=0.7, a_diff_thresh:float=np.pi/20,a_std_thresh=np.pi/30,a_inlier_thresh=0.5,
    min_len:int=10,merge:bool=False,merge_thresh:int=4,
):
    line_candidates = create_line_candidates(keypoints)
    prelim_valid_lines,points,angles,direction = coarse_line_filter(
        line_candidates,df,af,n_samples=n_samples,df_thresh=df_thresh + 0.5,a_diff_thresh=a_diff_thresh * 2,
        a_std_thresh=a_std_thresh * 3,a_inlier_thresh=a_inlier_thresh / 3,min_len=min_len,max_len=0.25*min(img_size)
    )
    valid_lines = fine_line_filter(
        prelim_valid_lines, points, angles,direction,df_thresh=df_thresh,inlier_thresh=inlier_thresh,a_diff_thresh=a_diff_thresh,
        a_std_thresh=a_std_thresh,a_inlier_thresh=a_inlier_thresh,
        merge=merge,merge_thresh=merge_thresh
    )

    return valid_lines