from typing import Callable, Dict, List, Optional, Union

from IPython.display import display
from ipywidgets import HBox, Layout, Output, Tab

from .. import AnalysisState
from .base import AbstractVisualization
from .jupyter import JupyterMixin

__all__ = [
    "MarkdownSectionComponent",
    "SimpleVerticalLinearLayout",
    "SimpleHorizontalLayout",
    "PropertyRendererComponent",
    "TabLayout",
]

from ..state import is_key_present_in_state


class SimpleVerticalLinearLayout(AbstractVisualization):
    """
    Renders facets in a sequential order (facets will appear in a vertical layout).
    """

    def __init__(
        self,
        facets: Union[AbstractVisualization, List[AbstractVisualization]],
        namespace: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(namespace, **kwargs)
        if not isinstance(facets, list):
            facets = [facets]
        self.facets = facets
        self._kwargs = kwargs

    def can_handle(self, state: AnalysisState) -> bool:
        return True

    def _render(self, state: AnalysisState) -> None:
        for facet in self.facets:
            facet.render(state)


class SimpleHorizontalLayout(SimpleVerticalLinearLayout):
    """
    Render components horizontally using `HBox` widget.
    See `HBox widget <https://ipywidgets.readthedocs.io/en/stable/examples/Widget%20List.html#HBox>`_ documentation for details.
    """

    def _render(self, state: AnalysisState) -> None:
        outs = [Output() for _ in range(len(self.facets))]
        for out, facet in zip(outs, self.facets):
            with out:
                facet.render(state)

        display(HBox(outs, layout=Layout(flex="row wrap")))


class TabLayout(SimpleVerticalLinearLayout):
    """
    Render components using `Tab` widget.
    See `HBox widget <https://ipywidgets.readthedocs.io/en/stable/examples/Widget%20List.html#Tabs>`_ documentation for details.

    """

    def __init__(self, facets: Dict[str, AbstractVisualization], namespace: Optional[str] = None, **kwargs) -> None:
        self.facet_tab_names = list(facets.keys())
        super().__init__(list(facets.values()), namespace, **kwargs)

    def _render(self, state: AnalysisState) -> None:
        outs = [Output() for _ in self.facets]
        tab = Tab(children=outs, width=400)
        for i, name in enumerate(self.facet_tab_names):
            tab.set_title(i, name)
        display(tab)
        for out, facet in zip(outs, self.facets):
            with out:
                facet.render(state)


class MarkdownSectionComponent(AbstractVisualization, JupyterMixin):
    """
    Render provided string as a Markdown cell.
    See `Jupyter Markdown cell <https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Working%20With%20Markdown%20Cells.html>`_
    documentation for details.

    Parameters
    ----------
    markdown: str
        markdown text to render
    condition_fn: Optional[Callable], default = None
        if specified, call the provided function with `state` arg. The function expected to return bool value
    namespace: Optional[str], default = None
        namespace to use; can be nested like `ns_a.ns_b.ns_c`

    """

    def __init__(
        self, markdown: str, condition_fn: Optional[Callable] = None, namespace: Optional[str] = None, **kwargs
    ) -> None:
        """

        Parameters
        ----------
        markdown
        condition_fn
        namespace
        kwargs
        """
        super().__init__(namespace, **kwargs)
        self.markdown = markdown
        self.condition_fn = condition_fn

    def can_handle(self, state: AnalysisState) -> bool:
        return True if self.condition_fn is None else self.condition_fn(state)

    def _render(self, state: AnalysisState) -> None:
        self.render_markdown(self.markdown)


class PropertyRendererComponent(AbstractVisualization, JupyterMixin):
    """
    Render component stored in `state`'s dot-separated path to property (i.e. `a.b` results in `state.a.b`)

    Parameters
    ----------
    property: str
        dot-separated path to property (i.e. `a.b` results in `state.a.b`)
    transform_fn: Optional[Callable], default = None
        if specified, call the provided function with the object extracted from `state`'s `property`.
        Returned transformation is then passed into render function
    namespace: Optional[str], default = None
        namespace to use; can be nested like `ns_a.ns_b.ns_c`
    """

    def __init__(
        self, property: str, transform_fn: Optional[Callable] = None, namespace: Optional[str] = None, **kwargs
    ) -> None:
        super().__init__(namespace, **kwargs)
        self.property = property
        self.transform_fn = transform_fn

    def can_handle(self, state: AnalysisState) -> bool:
        return is_key_present_in_state(state, self.property)

    def _render(self, state: AnalysisState) -> None:
        obj = state
        for p in self.property.split("."):
            obj = obj[p]
        if self.transform_fn is not None:
            obj = self.transform_fn(obj)
        self.display_obj(obj)
