from IPython.display import HTML, Markdown, display


class JupyterMixin:
    def __init__(self) -> None:
        self.headers = False

    @staticmethod
    def display_obj(obj):
        display(obj)

    def render_header_if_needed(self, state, header_text, ds=""):
        sample_size = state.get("sample_size", {}).get(ds, None)
        if self.headers:
            sample_info = "" if sample_size is None else f" (sample size: {sample_size})"
            header = f"**{header_text}{sample_info}**"
            self.render_markdown(header)

    @staticmethod
    def render_markdown(md):
        display(Markdown(md))


class JupyterTools:
    def fix_tabs_scrolling(self):
        """
        Helper utility to fix Jupyter styles for Tab widgets. This enforces the Tab element to fully expand and not use scrollbar.
        See the issue: https://github.com/jupyter-widgets/ipywidgets/issues/1791
        """
        style = """
            <style>
               .jupyter-widgets-output-area .output_scroll {
                    height: unset !important;
                    border-radius: unset !important;
                    -webkit-box-shadow: unset !important;
                    box-shadow: unset !important;
                }
                .jupyter-widgets-output-area  {
                    height: auto !important;
                }
            </style>
            """
        display(HTML(style))
