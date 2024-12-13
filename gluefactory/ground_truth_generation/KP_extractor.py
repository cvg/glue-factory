import os
import h5py
import numpy as np
import argparse
from scipy.ndimage import maximum_filter
from tqdm import tqdm

class KPExtractor:
    def __init__(self, config):
        self.threshold_type = config.get('threshold_type', 'nms')
        self.threshold_value = config.get('threshold_value', 0.015)
        self.max_keypoints = config.get('max_keypoints', None)  # Default to None if not provided

    def extract_keypoints(self, file_path):
        with h5py.File(file_path, 'r') as f:
            # Ensure dataset exists and handle missing keys gracefully
            if 'superpoint_heatmap' not in f:
                raise KeyError(f"'superpoint_heatmap' dataset not found in {file_path}")
            
            heatmap = f['superpoint_heatmap'][()]
            
            nms = (heatmap == maximum_filter(heatmap, size=3)) & (heatmap > self.threshold_value)
            keypoints = np.argwhere(nms)  # (row, col)
            scores = heatmap[nms]  # Extract scores

            if self.max_keypoints is not None and len(keypoints) > self.max_keypoints:
                idx = np.argsort(scores)[::-1][:self.max_keypoints]
                keypoints = keypoints[idx]

            return keypoints

class GTGenerator:
    def __init__(self, config):
        self.source_dir = config['source_dir']
        self.output_dir = config['output_dir']
        self.file_format = config.get('file_format', 'hdf5')
        self.output_format = config.get('output_format', 'numpy')
        self.extractor = KPExtractor(config.get('extractor_config', {}))

    def get_hdf5_files(self):
        files = []
        for root, _, filenames in os.walk(self.source_dir):
            files.extend([os.path.join(root, f) for f in filenames if f.endswith(self.file_format)])
        return files

    def save_keypoints(self, keypoints, file_path):
        # Modify output file name to remove .hdf5 extension
        rel_path = os.path.relpath(file_path, self.source_dir)
        output_file = os.path.join(self.output_dir, rel_path).replace('.hdf5', '.npy')
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        np.save(output_file, keypoints)

    def run(self):
        files = self.get_hdf5_files()
        for file_path in tqdm(files, desc="Processing files"):
            try:
                keypoints = self.extractor.extract_keypoints(file_path)
                self.save_keypoints(keypoints, file_path)
            except KeyError as e:
                print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate keypoints from SuperPoint heatmaps.")
    parser.add_argument('--source_dir', required=True, help="Path to the directory containing input HDF5 files.")
    parser.add_argument('--output_dir', required=True, help="Path to the directory to save output keypoints.")
    parser.add_argument('--file_format', default='hdf5', help="File format of the input files (default: hdf5).")
    parser.add_argument('--output_format', default='numpy', help="Output format for keypoints (default: numpy).")
    parser.add_argument('--threshold_type', default='nms', help="Thresholding type for keypoints (default: nms).")
    parser.add_argument('--threshold_value', type=float, default=0.015, help="Threshold value for keypoint detection (default: 0.015).")
    parser.add_argument('--max_keypoints', type=int, default=None, help="Maximum number of keypoints to keep (default: None).")

    args = parser.parse_args()

    config = {
        'source_dir': args.source_dir,
        'output_dir': args.output_dir,
        'file_format': args.file_format,
        'output_format': args.output_format,
        'extractor_config': {
            'threshold_type': args.threshold_type,
            'threshold_value': args.threshold_value,
            'max_keypoints': args.max_keypoints
        }
    }

    generator = GTGenerator(config)
    generator.run()
