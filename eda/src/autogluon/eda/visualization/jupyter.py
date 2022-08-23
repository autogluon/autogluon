from IPython.display import display, HTML


class JupyterMixin:

    def display_obj(self, obj):
        display(obj)

    def render_text(self, text, text_type=None):
        if text_type in [f'h{r}' for r in range(1, 7)]:
            display(HTML(f"<{text_type}>{text}</{text_type}>"))
        else:
            print(text)
