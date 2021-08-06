import pandas as pd
import numpy as np
import altair as alt
from typing import Callable
from scipy.optimize import curve_fit


alt.data_transformers.disable_max_rows()


class RawData:
    """Parse raw sim data into a proper pd.Dataframe"""

    def __init__(self, filename: str):
        self.filename = filename
        self.df = self.parse_from_file(self.filename)

    @staticmethod
    def _add_origin(df: pd.DataFrame) -> pd.DataFrame:
        """Add origin data point in each model"""

        if df.epoch.min() > 0:
            # Borrow epoch == 1.0 as a dataframe for epoch = 0
            df_origin = df.loc[df.epoch == 1.0].copy()
            df_origin.score = 0
            df_origin.epoch = 0
            df_with_origin = pd.concat([df, df_origin], ignore_index=True)
            return df_with_origin.sort_values(
                by=["code_name", "cond", "epoch"]
            ).reset_index(drop=True)

        else:
            print("Already have origin, returning original df")
            return df

    def parse_from_file(self, filename: str) -> pd.DataFrame:
        """File parser for sims"""

        df = pd.read_csv(filename, index_col=0)
        df.rename(
            columns={
                "ID": "code_name",
                "Trial.Scaled": "epoch",
                "Hidden": "hidden_units",
                "PhoHid": "cleanup_units",
                "Pnoise": "p_noise",
                "Epsilon": "learning_rate",
                "Type": "cond",
                "Measure": "measure",
                "Score": "score",
                "Freq": "cond_freq",
                "Cons": "cond_cons",
            },
            inplace=True,
        )

        # We only use accuracy in this paper, so we only select that measure
        df = df.loc[df.measure == "Accuracy"]

        # Add origin data point (for nicer plots)
        df = self._add_origin(df)

        # Change type to word and nonword label
        df["type"] = df.cond.apply(
            lambda x: "word"
            if x in ["HF_CON", "HF_INC", "LF_CON", "LF_INC"]
            else "nonword"
        )

        return df

    def get(
        self,
        code_name: int = None,
        cond: str = None,
        measure: str = "Accuracy",
        epoch_less_than: float = None,
        remove_zero: bool = False,
    ) -> pd.DataFrame:
        """Convienient function for getting a subset of data"""
        df = self.df.copy()
        if code_name is not None:
            df = df.loc[df.code_name == code_name]
        if cond is not None:
            df = df.loc[df.cond == cond]
        if measure is not None:
            df = df.loc[df.measure == measure]
        if epoch_less_than is not None:
            df = df.loc[df.epoch <= epoch_less_than]
        if remove_zero:
            df = df.loc[df.score > 0]
        return df


class GrowthModel:
    def __init__(self, growth_function: Callable, xdata: list, ydata: list, name: str):
        """Fitting a growth function to data
        df: dataframe with score (y) and epoch (x)
        growth_function: growth function to fit to the data
        name: model name for labeling purpose
        """
        self.growth_function = growth_function
        self.xdata = xdata
        self.ydata = ydata
        self.name = name

        # These arguements' value are obtained from fit()
        self.params = None
        self.mse = None

    def fit(self, bounds: tuple = None, f_scale: float = 0.01):
        """Fitting the selected curve with robust method
        bounds: model constrain of parameters
        f_scale (only or robust = True): soft residual cutoff for outlier,
            used in scipy.optimize.least_squares()
        See https://scipy-cookbook.readthedocs.io/items/robust_regression.html
        for more details
        """
        self.params, _ = curve_fit(
            f=self.growth_function,
            xdata=self.xdata,
            ydata=self.ydata,
            bounds=(-np.inf, np.inf) if bounds == None else bounds,
            maxfev=10000,
            method="trf",  # Relatively robust method
            loss="soft_l1",  # More robust to outlier
            f_scale=f_scale,  # Soft boundary for outlier residual
        )

        self.pred = self.growth_function(self.xdata, *self.params)
        self.mse = np.mean(np.square(self.pred - self.ydata))

    def predict(self, x: list = None) -> list:
        """Use fitted equation to predict y (with mean squared error)"""
        if x is None:
            x = self.xdata

        try:
            pred = self.growth_function(x, *self.params)
        except TypeError:
            print("Run fit() first")

        return pred

    def make_plot_data(self) -> pd.DataFrame:
        """Compile plot data"""

        self.df_actual = pd.DataFrame(
            {"set": "actual", "epoch": self.xdata, "score": self.ydata}
        )

        self.df_pred = pd.DataFrame(
            {
                "set": "predicted",
                "epoch": self.xdata,
                "score": self.pred,
            }
        )

        return pd.concat([self.df_actual, self.df_pred], ignore_index=True)

    def plot(self) -> alt.Chart:
        """plot predicted versus actual value in the growth model"""
        df = self.make_plot_data()

        return (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                y=alt.Y("score", scale=alt.Scale(domain=(0, 1))),
                x=alt.X("epoch", scale=alt.Scale(domain=(0, 1))),
                color="set",
            )
            .properties(
                title=[
                    f"Model: {self.name}",
                    f"Parameters: {self.params.round(3)}",
                    f"Model MSE {self.mse:.3f}",
                ]
            )
        )


def alt_diagonal(
    x_label: str = "Word", y_label: str = "Nonword", color: str = "#D3D3D3"
) -> alt.Chart:
    """Make a diagonal line altair plot"""
    return (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(color=color)
        .encode(
            x=alt.X("x", axis=alt.Axis(title=x_label)),
            y=alt.Y("y", axis=alt.Axis(title=y_label)),
        )
    )


def apply_font_size(plot: alt.Chart, font_size: int = 18) -> alt.Chart:
    """Applying font size to altair chart"""
    return (
        plot.configure_axis(labelFontSize=font_size, titleFontSize=font_size)
        .configure_header(labelFontSize=font_size, titleFontSize=font_size)
        .configure_legend(labelFontSize=font_size, titleFontSize=font_size)
        .configure_title(
            fontSize=int(font_size * 1.3), font="serif", fontStyle="normal"
        )
    )


def control_space_heatmap(
    df,
    title,
    var,
    color_scheme,
    domain,
    font_size=18,
    epsilon_label="Epsilon",
    pnoise_label="P-noise",
    hide_legend=False,
):
    """Plotting a value (var) on control space as a heatmap, used in Figure 2 and Supplementary figures"""
    heatmap = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("p_noise:O", title=pnoise_label),
            y=alt.Y("hidden_units:O", sort="descending", title="Hidden"),
            column=alt.Row("learning_rate:O", sort="descending", title=epsilon_label),
            color=alt.Color(
                var,
                scale=alt.Scale(scheme=color_scheme, domain=domain),
                title=None,
            ),
        )
        .properties(title=title)
    )

    heatmap = apply_font_size(heatmap, font_size)

    if hide_legend:
        heatmap = heatmap.encode(
            color=alt.Color(
                var,
                scale=alt.Scale(scheme=scheme, domain=domain),
                title=color_label,
                legend=None,
            )
        )

    return heatmap


def parse_df(df: pd.DataFrame) -> pd.DataFrame:
    """Parsing the raw dataframe for plotting
    1. Mean within word (HF_INC, HF_CON, LF_INC, LF_CON) and nonword (NW_UN, NW_AMB)
    2. Add a cell code to indicate control space location
    """
    group_vars = [
        "code_name",
        "hidden_units",
        "p_noise",
        "learning_rate",
        "epoch",
        "type",
    ]

    df = df.groupby(group_vars).mean().reset_index()

    # Create a cell_code variable for later use in inteactive plot's selector
    df["cell_code"] = (
        "h"
        + df.hidden_units.astype(str)
        + "_p"
        + df.p_noise.astype(str)
        + "_l"
        + df.learning_rate.astype(str)
    )

    return df


def long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a long dataframe to wide format for plotting performance space
    Covnert word / nonword from row to column
    """
    index_vars = [
        "code_name",
        "cell_code",
        "hidden_units",
        "p_noise",
        "learning_rate",
        "epoch",
    ]
    wide_df = df.pivot_table(index=index_vars, columns="type").reset_index()

    wide_df.columns = wide_df.columns = [
        "_".join(c).strip("_") for c in wide_df.columns.values
    ]
    return wide_df


class InteractivePlot:
    def __init__(self, df: pd.DataFrame):

        # Parsed dataframes
        self.df = parse_df(df)
        self.wide_df = long_to_wide(self.df)
        self.wide_df = self._remove_extra_zeros(self.wide_df)

    @staticmethod
    def _remove_extra_zeros(df: pd.DataFrame) -> pd.DataFrame:
        """Keep only one zero data point to avoid LOESS regression overfitting on it
        (Only for wide format dataframe)
        """
        df = df.loc[(df.score_nonword > 0) | (df.epoch.isin([0, 0.05, 0.3]))]
        return df

    def plot(self, font_size: int = 18) -> alt.Chart:
        """Plotting the interactive plot"""
        max_epoch = self.df.epoch.max()

        ## I: Control Space ###
        select_control_space = alt.selection(
            type="multi",
            on="click",
            empty="none",
            fields=["cell_code"],
        )

        control_space = (
            alt.Chart(self.df.loc[self.df.epoch == self.df.epoch.max()])
            .mark_rect(stroke="white", strokeWidth=2)
            .encode(
                x=alt.X("p_noise:O", title="P-Noise"),
                y=alt.Y("hidden_units:O", sort="descending", title="Hidden"),
                column=alt.Column(
                    "learning_rate:O", sort="descending", title="Epsilon"
                ),
                color=alt.Color(
                    "mean(score):Q",
                    scale=alt.Scale(domain=(0, 1), scheme="redyellowgreen"),
                    title=f"Accuracy at {max_epoch:.1f}M",
                ),
                detail="cell_code:N",
                opacity=alt.condition(
                    select_control_space, alt.value(1), alt.value(0.1)
                ),
            )
            .add_selection(select_control_space)
        ).properties(title=f"Control space (Click cell to get plots)")

        ### II: Developmental Space ###

        development_space = (
            alt.Chart(self.df)
            .mark_line()
            .encode(
                y=alt.Y(
                    "mean(score):Q", title="Accuracy", scale=alt.Scale(domain=(0, 1))
                ),
                x=alt.X(
                    "epoch:Q",
                    title="Sample (M)",
                    scale=alt.Scale(domain=(0, max_epoch)),
                ),
                color=alt.Color(
                    "type:N",
                    legend=alt.Legend(orient="none", legendX=300, legendY=180),
                    title="Word type",
                ),
            )
            .properties(title="Developmental space")
            .transform_filter(select_control_space)
        )

        ### III: Performance Space ###

        # Points
        base = (
            alt.Chart(self.wide_df)
            .mark_circle(color="black", size=10)
            .encode(
                x=alt.X(f"score_word:Q", scale=alt.Scale(domain=(0, 1)), title="Word"),
                y=alt.Y(
                    f"score_nonword:Q", scale=alt.Scale(domain=(0, 1)), title="Nonword"
                ),
                tooltip=["epoch"],
            )
            .transform_filter(select_control_space)
        )

        # LOESS Curve
        loess = base.transform_loess(
            "score_word", "score_nonword", bandwidth=0.4
        ).mark_line(color="black")

        # Color points to indicate epoch
        color_points = (
            base.mark_circle(size=200)
            .encode(x="mean_x:Q", y="mean_y:Q", color=alt.Color("color:N", scale=None))
            .transform_aggregate(
                mean_x="mean(score_word)",
                mean_y="mean(score_nonword)",
                groupby=["epoch"],
            )
            .transform_calculate(
                color="if(datum.epoch===0.05, 'red', if(datum.epoch === 0.3, 'green', ''))"
            )
        )

        # Diagonal
        diagonal = alt_diagonal()

        performance_space = (base + color_points + loess + diagonal).properties(
            title=f"Performance space"
        )

        # Combined plot
        plot = control_space & (development_space | performance_space)

        return apply_font_size(plot, font_size)
