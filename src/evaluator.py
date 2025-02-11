from collections import defaultdict
from pathlib import Path

import supervisely as sly
from supervisely.nn.benchmark.base_evaluator import BaseEvaluator

from src.eval_result import MyEvalResult


class MyEvaluator(BaseEvaluator):
    eval_result_cls = MyEvalResult  # we will implement this class in the next step

    def evaluate(self):
        """This method should CALCULATE evaluation metrics and DUMP them to disk."""

        # For example, let's iterate over all datasets and calculate some statistics
        gt_project = sly.Project(self.gt_project_path, sly.OpenMode.READ)
        pred_project = sly.Project(self.pred_project_path, sly.OpenMode.READ)

        gt_stats = {"images_count": defaultdict(int), "objects_count": defaultdict(int)}
        pred_stats = {"images_count": defaultdict(int), "objects_count": defaultdict(int)}

        for ds_1 in gt_project.datasets:
            ds_2 = pred_project.datasets.get(ds_1.name)
            ds_1: sly.Dataset
            for name in ds_1.get_items_names():
                ann_1 = ds_1.get_ann(name, gt_project.meta)
                ann_2 = ds_2.get_ann(name, pred_project.meta)

                gt_founded_classes = set()
                pred_founded_classes = set()
                for label in ann_1.labels:
                    if label.obj_class.geometry_type != sly.Polygon:
                        continue
                    if label.obj_class.name == "person":
                        continue
                    class_name = label.obj_class.name
                    gt_founded_classes.add(class_name)
                    gt_stats["objects_count"][class_name] += 1
                for label in ann_2.labels:
                    if label.obj_class.geometry_type != sly.Bitmap:
                        continue
                    if label.obj_class.name == "person":
                        continue
                    class_name = label.obj_class.name
                    pred_founded_classes.add(class_name)
                    pred_stats["objects_count"][class_name] += 1

                for class_name in gt_founded_classes:
                    gt_stats["images_count"][class_name] += 1
                for class_name in pred_founded_classes:
                    pred_stats["images_count"][class_name] += 1

        # save the evaluation results
        self.eval_data = {"gt_stats": gt_stats, "pred_stats": pred_stats}

        # Dump the eval_data to disk (to be able to load it later)
        save_path = Path(self.result_dir) / "eval_data.json"
        sly.json.dump_json_file(self.eval_data, save_path)
        return self.eval_data
