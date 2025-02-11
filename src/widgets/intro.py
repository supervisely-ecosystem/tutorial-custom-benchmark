from datetime import datetime

from supervisely.nn.benchmark.object_detection.base_vis_metric import BaseVisMetric
from supervisely.nn.benchmark.visualization.widgets import MarkdownWidget


class Intro(BaseVisMetric):

    def get_header(self, user_login: str) -> MarkdownWidget:
        current_date = datetime.now().strftime("%d %B %Y, %H:%M")

        header_text = (
            "<h1>Pretrained YOLOv11 Model</h1>"
            "<div class='model-info-block'>"
            f"   <div>Created by <b>{user_login}</b></div>"
            f"   <div><i class='zmdi zmdi-calendar-alt'></i><span>{current_date}</span></div>"
            "</div>"
        )
        header = MarkdownWidget("markdown_header", "Header", text=header_text)
        return header

    @property
    def md(self) -> MarkdownWidget:
        text = "## Overview \n- **Task type**: Object Detection\n"
        md = MarkdownWidget(name="intro", title="Intro", text=text)
        md.is_info_block = True
        md.width_fit_content = True
        return md
