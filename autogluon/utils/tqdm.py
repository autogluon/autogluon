import sys
from .miscs import in_ipynb

from tqdm import tqdm as base

__all__ = ['tqdm']


if True:  # pragma: no cover
    # import IPython/Jupyter base widget and display utilities
    IPY = 0
    IPYW = 0
    try:  # IPython 4.x
        import ipywidgets
        IPY = 4
        try:
            IPYW = int(ipywidgets.__version__.split('.')[0])
        except AttributeError:  # __version__ may not exist in old versions
            pass
    except ImportError:  # IPython 3.x / 2.x
        IPY = 32
        import warnings
        with warnings.catch_warnings():
            ipy_deprecation_msg = "The `IPython.html` package" \
                                  " has been deprecated"
            warnings.filterwarnings('error',
                                    message=".*" + ipy_deprecation_msg + ".*")
            try:
                import IPython.html.widgets as ipywidgets
            except Warning as e:
                if ipy_deprecation_msg not in str(e):
                    raise
                warnings.simplefilter('ignore')
                try:
                    import IPython.html.widgets as ipywidgets  # NOQA
                except ImportError:
                    pass
            except ImportError:
                pass

    try:  # IPython 4.x / 3.x
        if IPY == 32:
            from IPython.html.widgets import IntProgress, HBox, HTML, VBox
            IPY = 3
        else:
            from ipywidgets import IntProgress, HBox, HTML, VBox
    except ImportError:
        try:  # IPython 2.x
            from IPython.html.widgets import IntProgressWidget as IntProgress
            from IPython.html.widgets import ContainerWidget as HBox
            from IPython.html.widgets import HTML
            IPY = 2
        except ImportError:
            IPY = 0

    try:
        from IPython.display import display  # , clear_output
    except ImportError:
        pass

    # HTML encoding
    try:  # Py3
        from html import escape
    except ImportError:  # Py2
        from cgi import escape


class mytqdm(base):
    @staticmethod
    def status_printer(_, total=None, desc=None, ncols=None, img=None):
        """
        Manage the printing of an IPython/Jupyter Notebook progress bar widget.
        """
        # Fallback to text bar if there's no total
        # DEPRECATED: replaced with an 'info' style bar
        # if not total:
        #    return super(mytqdm, mytqdm).status_printer(file)

        # fp = file

        # Prepare IPython progress bar
        try:
            if total:
                pbar = IntProgress(min=0, max=total)
            else:  # No total? Show info style bar with no progress tqdm status
                pbar = IntProgress(min=0, max=1)
                pbar.value = 1
                pbar.bar_style = 'info'
        except NameError:
            # #187 #451 #558
            raise ImportError(
                "IntProgress not found. Please update jupyter and ipywidgets."
                " See https://ipywidgets.readthedocs.io/en/stable"
                "/user_install.html")

        if desc:
            pbar.description = desc
            if IPYW >= 7:
                pbar.style.description_width = 'initial'
        # Prepare status text
        ptext = HTML()
        timg = HTML()
        if img:
            timg.value = "<br>%s<br>" % img
        # Only way to place text to the right of the bar is to use a container
        container = VBox([HBox(children=[pbar, ptext]), timg])
        # Prepare layout
        if ncols is not None:  # use default style of ipywidgets
            # ncols could be 100, "100px", "100%"
            ncols = str(ncols)  # ipywidgets only accepts string
            try:
                if int(ncols) > 0:  # isnumeric and positive
                    ncols += 'px'
            except ValueError:
                pass
            pbar.layout.flex = '2'
            container.layout.width = ncols
            container.layout.display = 'inline-flex'
            container.layout.flex_flow = 'row wrap'
        display(container)

        return container

    def display(self, msg=None, pos=None,
                # additional signals
                close=False, bar_style=None):
        # Note: contrary to native tqdm, msg='' does NOT clear bar
        # goal is to keep all infos if error happens so user knows
        # at which iteration the loop failed.

        # Clear previous output (really necessary?)
        # clear_output(wait=1)

        if not msg and not close:
            msg = self.__repr__()

        tbar, timg = self.container.children
        pbar, ptext = tbar.children
        pbar.value = self.n

        if self.img:
            timg.value = "<br>%s<br>" % self.img

        if msg:
            # html escape special characters (like '&')
            if '<bar/>' in msg:
                left, right = map(escape, msg.split('<bar/>', 1))
            else:
                left, right = '', escape(msg)

            # remove inesthetical pipes
            if left and left[-1] == '|':
                left = left[:-1]
            if right and right[0] == '|':
                right = right[1:]

            # Update description
            pbar.description = left
            if IPYW >= 7:
                pbar.style.description_width = 'initial'

            # never clear the bar (signal: msg='')
            if right:
                ptext.value = right

        # Change bar style
        if bar_style:
            # Hack-ish way to avoid the danger bar_style being overridden by
            # success because the bar gets closed after the error...
            if not (pbar.bar_style == 'danger' and bar_style == 'success'):
                pbar.bar_style = bar_style

        # Special signal to close the bar
        if close and pbar.bar_style != 'danger':  # hide only if no error
            try:
                self.container.close()
            except AttributeError:
                self.container.visible = False

    def set_svg(self, img):
        self.img = img

    def __init__(self, *args, **kwargs):
        # Setup default output
        file_kwarg = kwargs.get('file', sys.stderr)
        if file_kwarg is sys.stderr or file_kwarg is None:
            kwargs['file'] = sys.stdout  # avoid the red block in IPython

        # Initialize parent class + avoid printing by using gui=True
        kwargs['gui'] = True
        kwargs.setdefault('bar_format', '{l_bar}{bar}{r_bar}')
        kwargs['bar_format'] = kwargs['bar_format'].replace('{bar}', '<bar/>')
        super(mytqdm, self).__init__(*args, **kwargs)
        if self.disable or not kwargs['gui']:
            return

        # Get bar width
        self.ncols = '100%' if self.dynamic_ncols else kwargs.get("ncols", None)

        # Replace with IPython progress bar display (with correct total)
        unit_scale = 1 if self.unit_scale is True else self.unit_scale or 1
        total = self.total * unit_scale if self.total else self.total
        self.img = None
        self.container = self.status_printer(
            self.fp, total, self.desc, self.ncols, self.img)
        self.sp = self.display

        # Print initial bar state
        if not self.disable:
            self.display()

    def __iter__(self, *args, **kwargs):
        try:
            for obj in super(mytqdm, self).__iter__(*args, **kwargs):
                # return super(tqdm...) will not catch exception
                yield obj
        # NB: except ... [ as ...] breaks IPython async KeyboardInterrupt
        except:  # NOQA
            self.sp(bar_style='danger')
            raise

    def update(self, *args, **kwargs):
        try:
            super(mytqdm, self).update(*args, **kwargs)
        except Exception as exc:
            # cannot catch KeyboardInterrupt when using manual tqdm
            # as the interrupt will most likely happen on another statement
            self.sp(bar_style='danger')
            raise exc

    def close(self, *args, **kwargs):
        super(mytqdm, self).close(*args, **kwargs)
        # If it was not run in a notebook, sp is not assigned, check for it
        if hasattr(self, 'sp'):
            # Try to detect if there was an error or KeyboardInterrupt
            # in manual mode: if n < total, things probably got wrong
            if self.total and self.n < self.total:
                self.sp(bar_style='danger')
            else:
                if self.leave:
                    self.sp(bar_style='success')
                else:
                    self.sp(close=True)

    def moveto(self, *args, **kwargs):
        # void -> avoid extraneous `\n` in IPython output cell
        return


tqdm = mytqdm if in_ipynb() else base
