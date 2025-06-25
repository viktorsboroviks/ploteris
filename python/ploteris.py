import json
import subprocess
import typing
import pandas as pd
import vplot


# helpers


def paths_dict(path_prefix: str, path_postfix) -> dict[int, str]:
    def _substring_between(string: str, prefix: str, postfix: str) -> str:
        start = string.find(prefix) + len(prefix)
        end = string.find(postfix, start)
        return string[start:end]

    output = subprocess.check_output(["find", ".", "-type", "f"], text=True)
    output = output.strip().split("\n")
    paths = {}
    for filepath in output:
        if path_prefix in filepath and path_postfix in filepath:
            i = int(_substring_between(filepath, path_prefix, path_postfix))
            assert i not in paths
            paths[i] = filepath

    return paths


def data_from_path(
    path: str, index_col: str = None, parse_dates: bool = False
) -> pd.DataFrame:
    assert isinstance(path, str)
    data = pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)
    if parse_dates:
        assert isinstance(data.index, pd.DatetimeIndex)
    return data


def data_from_paths(
    paths: dict[int, str], index_col: str, parse_dates: bool = False
) -> dict[pd.DataFrame]:
    assert isinstance(paths, dict)
    data_dict = {}
    for key, path in paths.items():
        assert isinstance(key, int)
        assert isinstance(path, str)
        assert key not in data_dict
        data_dict[key] = data_from_path(path, index_col)
    return data_dict


def get_data(config_json: str) -> dict[str, pd.DataFrame]:
    assert isinstance(config_json, list)
    data_dict = {}
    for entry in config_json:
        key = entry["key"]
        path = entry.get("path", None)
        path_prefix = entry.get("path_prefix", None)
        path_postfix = entry.get("path_postfix", None)
        index_col = entry.get("index_col", None)
        parse_dates = entry.get("parse_dates", False)

        if path:
            data_dict[key] = data_from_path(path, index_col, parse_dates)
        else:
            data_dict[key] = data_from_paths(
                paths_dict(path_prefix, path_postfix),
                index_col,
                parse_dates,
            )
    return data_dict


def get_idx(config_json: str, data: dict[str, pd.DataFrame], sort=True):
    idx = config_json.get("idx", None)
    if idx not in (None, ""):
        assert isinstance(idx, list)
        return sorted(idx)

    idx_data = config_json["idx_data"]
    assert idx_data in data
    if isinstance(data[idx_data], dict):
        idx = data[idx_data].keys()
    elif isinstance(data[idx_data], pd.DataFrame):
        idx = data[idx_data].index
    else:
        assert False
    if sort:
        idx = sorted(idx)
    return idx


def get_plot_filename(
    config_json: str, config_key: str, i: int = None, i_max: int = None
) -> str:
    prefix = config_json["plot"]["output"][config_key]["path_prefix"]
    postfix = config_json["plot"]["output"][config_key]["path_postfix"]
    assert (i is None) == (i_max is None)
    if i is None:
        i_str = ""
    else:
        i_max_len = len(str(i_max))
        assert len(str(i)) <= i_max_len
        i_str = str(i).zfill(i_max_len)
    return f"{prefix}{i_str}{postfix}"


# Ploteris


class Ploteris:
    config_json: str
    config_key: str
    data: dict
    idx: list = None

    def __init__(self, config_path, config_key: str):
        with open(config_path) as f:
            self.config_json = json.load(f)

        self.config_key = config_key

        # get data from files as tables to be used
        self.data = get_data(self.config_json["plot"]["data"])

        output_config_json = self.config_json["plot"]["output"][self.config_key]
        if "idx" in output_config_json and len(output_config_json["idx"]) > 0:
            self.idx = get_idx(output_config_json, self.data)

    # virtual
    def get_plot(self, i: int) -> vplot.PlotlyPlot:
        raise NotImplementedError()

    def plot_plot(self, i: int = None):

        assert (i is None) == (self.idx is None)
        if i is not None:
            i_max = self.idx[-1]
        else:
            i_max = None

        plot_filename = get_plot_filename(self.config_json, self.config_key, i, i_max)
        plot = self.get_plot(i)
        if ".html" in plot_filename:
            plot.width = None
            plot.height = None
        plot.to_file(plot_filename)

    def plot_plots(self):
        assert self.idx is not None
        assert len(self.idx) > 0

        for i in self.idx:
            assert isinstance(i, int)
            self.plot_plot(i)


# get vplot structures


def get_subplot_hist(
    col,
    row,
    data,
    mean=None,
    median=None,
    sd=None,
    bins=10,
    color=vplot.Color.BLUE,
    fill: typing.Literal[None, "solid", "transparent"] = None,
    legendgroup_name=None,
    y_title=None,
) -> vplot.Subplot:
    lines = []
    # histogram
    traces = [
        vplot.Histogram(
            is_probability_density=True,
            data=data,
            bins=bins,
            color=color,
            fill=fill,
            name="p density",
        ),
    ]

    # mean
    if mean is not None:
        traces += [
            vplot.Scatter(
                x=[mean],
                y=[0],
                color=vplot.Color.BLUE,
                dash=vplot.Dash.SOLID,
                hoverinfo="x+name",
                name="mean",
                showlegend=False,
            )
        ]
        lines += [
            vplot.Line(
                x=[mean],
                color=vplot.Color.BLUE,
                dash=vplot.Dash.SOLID,
            ),
        ]

    # median
    if median is not None:
        traces += [
            vplot.Scatter(
                x=[median],
                y=[0],
                color=vplot.Color.VIOLET,
                dash=vplot.Dash.SOLID,
                hoverinfo="x+name",
                name="median",
                showlegend=False,
            )
        ]
        lines += [
            vplot.Line(
                x=[median],
                color=vplot.Color.VIOLET,
                dash=vplot.Dash.SOLID,
            ),
        ]

    # sd
    if sd is not None:
        assert mean is not None
        assert lines is not None
        traces += [
            vplot.Scatter(
                x=[mean - sd],
                y=[0],
                color=vplot.Color.RED,
                dash=vplot.Dash.DASH,
                hoverinfo="x+name",
                name="sd",
                showlegend=False,
            ),
            vplot.Scatter(
                x=[mean + sd],
                y=[0],
                color=vplot.Color.RED,
                dash=vplot.Dash.DASH,
                hoverinfo="x+name",
                name="sd",
                # do not show legend for the 2nd sd to avoid duplication
            ),
        ]
        lines += [
            vplot.Line(
                x=[mean - sd, mean + sd],
                color=vplot.Color.RED,
                dash=vplot.Dash.DASH,
            ),
        ]
    return vplot.Subplot(
        col=col,
        row=row,
        traces=traces,
        lines=lines,
        legendgroup_name=legendgroup_name,
        y_title=y_title,
    )


def get_subplot_cdf(
    col,
    row,
    data,
    color=vplot.Color.BLUE,
    legendgroup_name=None,
    y_title=None,
) -> vplot.Subplot:
    # histogram
    traces = [
        vplot.CDF(
            data=data,
            color=color,
            name="cdf",
        ),
    ]

    return vplot.Subplot(
        col=col,
        row=row,
        traces=traces,
        legendgroup_name=legendgroup_name,
        y_title=y_title,
    )


def get_subplot_scatter3d(
    col,
    row,
    x_data: pd.Series,
    y_data: pd.Series,
    z_data: pd.Series,
    name: str = None,
    x_title: str = None,
    y_title: str = None,
    z_title: str = None,
) -> vplot.Subplot:

    traces = [
        vplot.Scatter3d(
            x=x_data,
            y=y_data,
            z=z_data,
            name=name,
        )
    ]

    return vplot.Subplot(
        col=col,
        row=row,
        traces=traces,
        x_title=f"x: {x_title}" if x_title else None,
        y_title=f"y: {y_title}" if y_title else None,
        z_title=f"z: {z_title}" if z_title else None,
    )


def get_subplot_heatmap2d(
    col,
    row,
    x_data: pd.Series,
    y_data: pd.Series,
    f_data: pd.Series,
    y_title: str = None,
    data_name: str = None,
    legendgroup_name: str = None,
) -> vplot.Subplot:

    showlegend = False
    if legendgroup_name:
        showlegend = False

    traces = [
        vplot.ScatterHeatmap2D(
            x=x_data,
            y=y_data,
            f=f_data,
            marker_size=2,
            showlegend=showlegend,
            name=data_name,
        )
    ]

    subplot = vplot.Subplot(
        col=col,
        row=row,
        traces=traces,
        log_y=False,
        y_title=y_title,
        legendgroup_name=legendgroup_name,
    )
    return subplot


def get_subplot_value(
    col,
    row,
    data: pd.Series,
    log_y: bool = False,
    plot_step: bool = False,
    color: vplot.Color = vplot.Color.BLUE,
    dash: vplot.Dash = vplot.Dash.SOLID,
    y_title: str = None,
    data_name: list[str] = None,
    legendgroup_name: str = None,
) -> vplot.Subplot:
    if plot_step:
        trace_function = vplot.Step
    else:
        trace_function = vplot.Scatter

    traces = []
    showlegend = False
    if legendgroup_name:
        showlegend = False

    if not isinstance(data, list):
        assert not isinstance(data, list)
        assert not isinstance(color, list)
        assert not isinstance(dash, list)
        assert not isinstance(data_name, list)
        traces += [
            trace_function(
                x=data.index,
                y=data,
                color=color,
                dash=dash,
                showlegend=showlegend,
                name=data_name,
            )
        ]
    elif len(data) == len(color):
        assert isinstance(data, list)
        assert isinstance(color, list)
        assert isinstance(dash, list)
        assert isinstance(data_name, list)
        assert len(data) == len(color) == len(dash) == len(data_name)
        for i in range(len(data)):
            traces += [
                trace_function(
                    x=data[i].index,
                    y=data[i],
                    color=color[i],
                    dash=dash[i],
                    showlegend=showlegend,
                    name=data_name[i],
                )
            ]
    else:
        assert isinstance(data, list)
        assert not isinstance(color, list)
        assert not isinstance(dash, list)
        assert not isinstance(data_name, list)
        for i in range(len(data)):
            traces += [
                trace_function(
                    x=data[i].index,
                    y=data[i],
                    color=color,
                    dash=dash,
                    showlegend=showlegend,
                    name=data_name,
                )
            ]

    subplot = vplot.Subplot(
        col=col,
        row=row,
        traces=traces,
        log_y=log_y,
        y_title=y_title,
        legendgroup_name=legendgroup_name,
    )
    return subplot
