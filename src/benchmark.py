from supervisely.nn.benchmark.base_benchmark import BaseBenchmark
from supervisely.nn.task_type import TaskType

from src.evaluator import MyEvaluator
from src.visualizer import MyVisualizer


class CustomBenchmark(BaseBenchmark):
    visualizer_cls = MyVisualizer  # ⬅︎ the visualizer class

    @property
    def cv_task(self) -> str:
        return TaskType.OBJECT_DETECTION  # ⬅︎ the visualizer class

    def _get_evaluator_class(self) -> type:
        return MyEvaluator  # ⬅︎ the visualizer class
