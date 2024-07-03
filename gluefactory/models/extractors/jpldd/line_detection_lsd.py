from gluefactory.geometry.line_utils import merge_lines, filter_outlier_lines
from gluefactory.utils.deeplsd_utils import preprocess_angle
from pytlsd import lsd
import numpy as np


def detect_afm_lines(img: np.array, df: np.array, line_level: np.array, filtering='normal',
        merge=False, grad_thresh=3, grad_nfa=True):
        """ Detect lines from the line distance and angle field.
            Offer the possibility to ignore line in high DF values,
            and to merge close-by lines. """
        gradnorm = np.maximum(5 - df, 0).astype(np.float64)
        angle = line_level.astype(np.float64) - np.pi / 2
        angle = preprocess_angle(angle, img, mask=True)[0]
        angle[gradnorm < grad_thresh] = -1024
        lines = lsd(
            img.astype(np.float64), scale=1., gradnorm=gradnorm,
            gradangle=angle, grad_nfa=grad_nfa)[:, :4].reshape(-1, 2, 2)

        # Optionally filter out lines based on the DF and line_level
        if filtering:
            if filtering == 'strict':
                df_thresh, ang_thresh = 1., np.pi / 12
            else:
                df_thresh, ang_thresh = 1.5, np.pi / 9
            angle = line_level - np.pi / 2
            lines = filter_outlier_lines(
                img, lines[:, :, [1, 0]], df, angle, mode='inlier_thresh',
                use_grad=False, inlier_thresh=0.5, df_thresh=df_thresh,
                ang_thresh=ang_thresh)[0][:, :, [1, 0]]

        # Merge close-by lines together
        if merge:
            lines = merge_lines(lines, thresh=4,
                                overlap_thresh=0).astype(np.float32)

        return lines