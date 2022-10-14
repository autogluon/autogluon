from IPython.display import display, HTML


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
