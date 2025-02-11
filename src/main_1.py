"""
Example 1:
This script is a part of the custom benchmark example.

In this example, we evaluate the model without inference. Assume that the predictions are already available.
Input data: 2 local projects (Ground Truth and Prediction) in Supervisely format.
We will evaluate the model using BaseEvaluator class.

The script performs the following steps:
1. Download the Ground Truth and Prediction projects.
2. Initialize the Evaluator object.
3. Run the evaluation.
4. Initialize the EvalResult object.
5. Initialize the Visualizer object and visualize the results.
6. Upload the results to the Supervisely Team Files.

Please note that to open the visualizations in the web interface, 
you need to upload the visualizations to the Supervisely Team Files.
"""

import os

import supervisely as sly
from dotenv import load_dotenv

from src.evaluator import MyEvaluator
from src.visualizer import MyVisualizer

if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("local.env")

api = sly.Api()

team_id = sly.env.team_id()
gt_project_id = 73
pred_project_id = 159


workdir = sly.app.get_data_dir()
gt_path = os.path.join(workdir, "gt_project")
pred_path = os.path.join(workdir, "pred_project")
eval_result_dir = os.path.join(workdir, "evaluation")
vis_result_dir = os.path.join(workdir, "vizualizations")

# 0. Download projects
for project_id, path in [(gt_project_id, gt_path), (pred_project_id, pred_path)]:
    if not sly.fs.dir_exists(path):
        sly.download_project(
            api,
            project_id,
            path,
            log_progress=True,
            save_images=False,
            save_image_info=True,
        )


# 1. Initialize Evaluator
evaluator = MyEvaluator(gt_path, pred_path, eval_result_dir)

# 2. Run evaluation
evaluator.evaluate()

# 3. Initialize EvalResult object
eval_result = evaluator.get_eval_result()

# 4. Initialize visualizer and visualize
visualizer = MyVisualizer(api, [eval_result], vis_result_dir)
visualizer.visualize()

# 5. Upload to Supervisely Team Files
remote_dir = "/model-benchmark/custom_benchmark"
api.file.upload_directory(team_id, evaluator.result_dir, remote_dir + "/evaluation")
# ⬇︎ required to open visualizations in the web interface
visualizer.upload_results(team_id, remote_dir + "/visualizations/")
