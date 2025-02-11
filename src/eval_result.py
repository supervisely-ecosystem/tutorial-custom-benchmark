from collections import defaultdict
from pathlib import Path
from typing import Optional

import supervisely as sly
from supervisely.nn.benchmark.base_evaluator import BaseEvalResult


class MyEvalResult(BaseEvalResult):

    def _read_files(self, path: str) -> None:  # ⬅︎ This method is required
        """This method should LOAD evaluation metrics from disk."""
        save_path = Path(path) / "eval_data.json"  # path to the saved evaluation metrics
        self.eval_data = sly.json.load_json_file(str(save_path))

    def _prepare_data(self) -> None:  # ⬅︎ This method is required
        """This method should PREPARE data to allow easy access to the data."""

        gt = self.eval_data.get("gt_stats", {})
        pred = self.eval_data.get("pred_stats", {})

        # class statistics (class names as keys and number of objects as values)
        self._objects_per_class = self._get_objects_per_class(gt, pred)

        # GT metrics
        gt_obj_num = self._get_total_objects_count(gt)
        gt_cls_num = self._get_num_of_used_classes(gt)
        gt_cls_most_freq = self._get_most_frequent_class(gt)

        # Prediction metrics
        pred_obj_num = self._get_total_objects_count(pred)
        pred_cls_num = self._get_num_of_used_classes(pred)
        pred_cls_most_freq = self._get_most_frequent_class(pred)

        self._key_metrics = {
            "Objects Count": [gt_obj_num, pred_obj_num],
            "Found Classes": [gt_cls_num, pred_cls_num],
            "Classes with Max Figures": [gt_cls_most_freq, pred_cls_most_freq],
        }

    # ---------------- ⬇︎ Properties to access the data easily ⬇︎ ----------------- #
    @property
    def key_metrics(self):
        """Return key metrics as a dictionary."""
        return self._key_metrics.copy()

    @property
    def objects_per_class(self):
        """Return the number of objects per class."""
        return self._objects_per_class.copy()

    # ------- ⬇︎ Utility methods (you can create any methods you need) ⬇︎ --------- #
    def _get_most_frequent_class(self, stats: dict):
        name = max(stats.get("objects_count", {}).items(), key=lambda x: x[1])[0]
        return f"{name} ({stats['objects_count'][name]})"

    def _get_total_objects_count(self, stats: dict):
        return sum(stats.get("objects_count", {}).values())

    def _get_objects_per_class(self, gt: dict, pred: dict):
        gt_img_stats = gt.get("objects_count", {})
        pred_img_stats = pred.get("objects_count", {})

        images_per_class = defaultdict(dict)
        for name, gt_images_count in gt_img_stats.items():
            pred_images_count = pred_img_stats.get(name, 0)
            images_per_class[name] = [gt_images_count, pred_images_count]

        return images_per_class

    def _get_num_of_used_classes(self, stats: dict):
        return len(stats.get("images_count", {}))
