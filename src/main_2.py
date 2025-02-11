"""
Example 2:
This script is a part of the custom benchmark example.

In this example, we will evaluate the model using the CustomBenchmark class.
Input data:
- Option 1: GT and Prediction projects IDs from Supervisely.
- Option 2: GT project ID and deployed model session ID from Supervisely.

The CustomBenchmark class is a wrapper around the BaseEvaluator class.
It has all the functionality for preparing the data, running inference, 
running the evaluation, visualizing the results, and uploading them to the Supervisely Team Files. 


The script performs the following steps:
1. Initialize the benchmark.
2. Run the evaluation.
3. Generate charts and dashboards.
4. Upload the results to the Supervisely Team Files.

Important:
- `run_evaluation` method is used to evaluate the model with inference.
- `evaluate` method is used to evaluate the model without inference.

Please note that to open the visualizations in the web interface, 
you need to upload the visualizations to the Supervisely Team Files.
"""

import os

import supervisely as sly
from dotenv import load_dotenv

from src.benchmark import CustomBenchmark

if sly.is_development():
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv("local.env")

api = sly.Api()

team_id = 8
gt_project_id = 73
pred_project_id = 159
model_session_id = 1234

# 1. Initialize benchmark
bench = CustomBenchmark(api, gt_project_id, output_dir=sly.app.get_data_dir())

# 2. Run evaluation
bench.evaluate(pred_project_id)  # ⬅︎ evaluate without inference
# bench.run_evaluation(model_session_id)    # ⬅︎ evaluate with inference

# 3. Generate charts and dashboards
bench.visualize()

# 4. Upload to Supervisely Team Files
remote_dir = f"/model-benchmark/custom_benchmark/{model_session_id}"
bench.upload_eval_results(remote_dir + "/evaluation/")
# ⬇︎ required to open visualizations in the web interface
bench.upload_visualizations(remote_dir + "/visualizations/")
