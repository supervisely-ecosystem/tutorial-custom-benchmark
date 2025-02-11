from supervisely.nn.benchmark.object_detection.base_vis_metric import BaseVisMetric
from supervisely.nn.benchmark.visualization.widgets import ChartWidget, MarkdownWidget


class CustomMetric(BaseVisMetric):

    @property
    def md(self) -> MarkdownWidget:
        text = (
            "## Number of Objects per Class\n"
            " In this section, you can explore the number of objects per class"
            " in the GT and predictions projects."
        )
        return MarkdownWidget(name="custom_metric", title="Custom Metric", text=text)

    @property
    def chart(self) -> ChartWidget:
        import plotly.graph_objects as go

        x = list(self.eval_result.objects_per_class.keys())
        y1, y2 = zip(*self.eval_result.objects_per_class.values())

        fig = go.Figure()
        fig.add_trace(go.Bar(y=y1, x=x, name="GT"))
        fig.add_trace(go.Bar(y=y2, x=x, name="Predictions"))

        fig.update_layout(barmode="group", bargap=0.15, bargroupgap=0.05)
        fig.update_xaxes(title_text="Class")
        fig.update_yaxes(title_text="Images")

        return ChartWidget(name="images_chart", figure=fig)
