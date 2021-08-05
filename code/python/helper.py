import pandas as pd
import numpy as np
import altair as alt
from typing import Callable
from scipy.optimize import curve_fit


class RawData:
    """Parse raw sim data into a proper pd.Dataframe"""

    def __init__(self, filename:str):
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

    def parse_from_file(self, filename:str) -> pd.DataFrame:
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
        df = df.loc[df.measure=="Accuracy"]

        # Add origin data point (for nicer plots)
        df = self._add_origin(df)

        # Change type to word and nonword label
        df["type"] = df.cond.apply(
            lambda x: "word" if x in ["HF_CON", "HF_INC", "LF_CON", "LF_INC"] else "nonword"
        )

        return df

    def get(self, code_name:int=None, cond:str=None, measure:str="Accuracy", remove_zero:bool=False):
        """Convienient function for getting a subset of data"""
        df = self.df.copy()
        if code_name is not None:
            df = df.loc[df.code_name==code_name]
        if cond is not None:
            df = df.loc[df.cond==cond]
        if measure is not None:
            df = df.loc[df.measure==measure]
        if remove_zero:
            df = df.loc[df.score > 0] 
        return df




class GrowthModel:

    def __init__(self, growth_function: Callable, xdata:list, ydata:list, name:str):
        """ Fitting a growth function to data
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

    def fit(self, bounds:tuple=None, f_scale:float=0.01):
        """ Fitting the selected curve with robust method
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

    def predict(self, x:list=None):
        """ Use fitted equation to predict y (with mean squared error) 
        """
        if x is None:
            x = self.xdata

        try:
            pred = self.growth_function(x, *self.params) 
        except TypeError:
            print("Run fit() first")

        return pred

    def make_plot_data(self):
        """Compile plot data"""

        self.df_actual = pd.DataFrame(
            {
                "set": "actual",
                "epoch": self.xdata,
                "score": self.ydata
            }
        )

        self.df_pred = pd.DataFrame(
            {
                "set": "predicted",
                "epoch": self.xdata,
                "score": self.pred,
            }
        )

        return pd.concat([self.df_actual, self.df_pred], ignore_index=True)


    def plot(self):
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






class SimResults:
    """All sim results handler
    I: Selection:
    1. Control space h-param filter
    2. Control space region filter
    3. DVs Condition filter

    II: Plotting:
    1. Where are the selected model in the control space
    2. How's their average performance (in each cond / mean of all conds)
    3. Some basic descriptives in title
    """

    def __init__(self, df, word_conds=["HF_INC"], nonword_conds=["NW_UN"]):
        self.df = df.loc[
            df.measure == "Accuracy",
        ].copy()
        self.word_conds = word_conds
        self.nonword_conds = nonword_conds
        self._label_control_space()

        # Copy df for later resetting
        self.original_df = self.df

        # Reuseable plotting element
        self.diagonal = (
            alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
            .mark_line(color="#D3D3D3")
            .encode(
                x=alt.X("x", axis=alt.Axis(title="Word")),
                y=alt.Y("y", axis=alt.Axis(title="Nonword")),
            )
        )

    def checkpoint_df(self):
        self.original_df = self.df
        print(f"Saved current state of df. reset_df() will go back to this point")

    def reset_df(self):
        self.df = self.original_df
        print(
            f"df reset to last checkpoint, if no check point is set, \
                it will revert to the state during creation"
        )

    def count_model(self):
        return len(self.df.code_name.unique())

    def select_by_control(
        self,
        hidden_units=None,
        p_noise=None,
        learning_rate=None,
        cleanup_units=None,
        verbose=True,
    ):
        """Control space filter by h-params"""

        n_pre = self.count_model()
        if hidden_units is not None:
            self.df = self.df.loc[self.df.hidden_units.isin(hidden_units)]
        if p_noise is not None:
            self.df = self.df.loc[self.df.p_noise.isin(p_noise)]
        if learning_rate is not None:
            self.df = self.df.loc[self.df.learning_rate.isin(learning_rate)]
        if cleanup_units is not None:
            self.df = self.df.loc[self.df.cleanup_units.isin(cleanup_units)]

        n_post = self.count_model()

        if verbose:
            print(f"Selected {n_post} models from the original {n_pre} models")

    def select_by_region(self, region_name, verbose=True):
        """Control space filter by good/base/bad label"""

        n_pre = self.count_model()
        self.df = self.df.loc[self.df.control_region.isin(region_name)]
        n_post = self.count_model()
        if verbose:
            print(f"Selected {n_post} models from the original {n_pre} models")

    def select_by_cond(self, conds, verbose=True):
        """Filter DVs by condition"""

        n_pre = self.count_model()
        self.df = self.df.loc[self.df.cond.isin(conds)]
        n_post = self.count_model()

        if verbose:
            print(f"Selected {n_post} models from the original {n_pre} models")

    def reset_word_nonword(self, word_conds, nonword_conds):

        self.word_conds = word_conds
        self.nonword_conds = nonword_conds
        self.df = self.original_df

    ### Plotting ###

    def plot_control_space(self, color="count(code_name)", with_cleanup=False):
        """Plot selected models at control space"""
        pdf = (
            self.df.groupby(["cell_code", "control_region", "code_name"])
            .mean()
            .round(3)
            .reset_index()
        )

        self.select_control_space = alt.selection(
            type="multi",
            on="click",
            empty="none",
            fields=["cell_code"],
        )

        if with_cleanup:
            control_space = (
                alt.Chart(pdf)
                .mark_rect(stroke="white", strokeWidth=2)
                .encode(
                    x="p_noise:O",
                    y=alt.Y("hidden_units:O", sort="descending"),
                    column=alt.Column("learning_rate:O", sort="descending"),
                    row=alt.Row("cleanup_units:O", sort="descending"),
                    color=color,
                    detail="cell_code",
                    opacity=alt.condition(
                        self.select_control_space, alt.value(1), alt.value(0.2)
                    ),
                )
                .add_selection(self.select_control_space)
            )
        else:
            control_space = (
                alt.Chart(pdf)
                .mark_rect(stroke="white", strokeWidth=2)
                .encode(
                    x="p_noise:O",
                    y=alt.Y("hidden_units:O", sort="descending"),
                    column=alt.Column("learning_rate:O", sort="descending"),
                    color=color,
                    detail="cell_code",
                    opacity=alt.condition(
                        self.select_control_space, alt.value(1), alt.value(0.2)
                    ),
                )
                .add_selection(self.select_control_space)
            )

        return control_space

    def _interactive_dev(self, show_sd, baseline=None):
        """Plot the mean development of all selected models"""
        df = self.df.loc[self.df.cond.isin(self.word_conds + self.nonword_conds)]

        development_space_mean = (
            alt.Chart(df)
            .mark_line()
            .encode(
                y=alt.Y(
                    "mean(score):Q", title="Accuracy", scale=alt.Scale(domain=(0, 1))
                ),
                x=alt.X("epoch:Q", title="Sample (M)"),
                color=alt.Color(
                    "cond:N",
                    legend=alt.Legend(orient="none", legendX=300, legendY=180),
                    title="Condition",
                ),
            )
            .properties(title="Developmental space")
            .transform_filter(self.select_control_space)
        )

        if show_sd:
            development_space_sd = development_space_mean.mark_errorband(
                extent="stdev"
            ).encode(
                y=alt.Y("score:Q", title="Accuracy", scale=alt.Scale(domain=(0, 1)))
            )
            development_space_mean += development_space_sd

        if baseline is not None:
            development_space_mean += baseline

        return development_space_mean

    def make_wnw(self):

        word_conds = self.word_conds
        nonword_conds = self.nonword_conds

        variates = ["hidden_units", "p_noise", "learning_rate"]

        df_wnw = self.df.loc[
            (self.df.cond.isin(word_conds + nonword_conds)),
            variates + ["code_name", "cell_code", "epoch", "cond", "score"],
        ]

        df_wnw["pivot_cond"] = df_wnw.cond.apply(
            lambda x: "word" if x in word_conds else "nonword"
        )

        df_wnw = df_wnw.pivot_table(
            index=variates + ["epoch", "code_name", "cell_code"], columns="pivot_cond"
        ).reset_index()

        df_wnw.columns = df_wnw.columns = [
            "".join(c).strip() for c in df_wnw.columns.values
        ]
        df_wnw.rename(
            columns={
                "scoreword": "word_acc",
                "scorenonword": "nonword_acc",
            },
            inplace=True,
        )

        return df_wnw

    def _interactive_wnw(self, baseline=None, x_label="Word", y_label="Nonword"):
        """Private function for interactive plot: Performance space plot"""
        df = self.make_wnw()

        base_wnw = (
            alt.Chart(df)
            .mark_line(color="black")
            .encode(
                y=alt.Y("mean_nw:Q", scale=alt.Scale(domain=(0, 1)), title=y_label),
                x=alt.X("mean_w:Q", scale=alt.Scale(domain=(0, 1)), title=x_label),
                tooltip=["epoch", "mean_w:Q", "mean_nw:Q"],
            )
            .transform_filter(self.select_control_space)
            .transform_aggregate(
                mean_w="mean(word_acc)", mean_nw="mean(nonword_acc)", groupby=["epoch"]
            )
            .transform_calculate(
                color="if(datum.epoch===0.05, 'red', if(datum.epoch === 0.3, 'green', ''))"
            )
        )

        points = (
            base_wnw.mark_circle(size=200)
            .encode(color=alt.Color("color:N", scale=None))
            .add_selection(alt.selection_single())
        )

        base_wnw += points

        if baseline is not None:
            base_wnw += baseline

        return (self.diagonal + base_wnw).properties(
            title="Performance space (Nonword vs. Word)"
        )

    def plot_interactive(
        self,
        title=None,
        show_sd=True,
        base_dev=None,
        base_wnw=None,
        performance_x_label="Word",
        performance_y_label="Nonword",
    ):
        """Plot averaged developmental and performance space + interactive control space selection"""

        all_plot = (
            self.plot_control_space()
            & (
                self._interactive_dev(show_sd=show_sd, baseline=base_dev)
                | self._interactive_wnw(
                    baseline=base_wnw,
                    x_label=performance_x_label,
                    y_label=performance_y_label,
                )
            )
        ).properties(title=f"{title} (n = {len(self.df.code_name.unique())})")

        return all_plot

    def plot_dev_multiline(self, variable_of_interest, sort="ascending"):

        group_var = [variable_of_interest, "epoch", "cond"]
        df = self.df.groupby(group_var).mean().reset_index()

        base = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x="epoch:Q",
                y=alt.Y("score:Q", scale=alt.Scale(domain=(0, 1))),
                color=alt.Color(
                    variable_of_interest,
                    scale=alt.Scale(scheme="redyellowgreen"),
                    sort=sort,
                ),
                tooltip=[variable_of_interest, "score"],
            )
        )

        word = base.transform_filter(alt.datum.cond == "HF_INC").properties(
            title="Word"
        )
        nonword = base.transform_filter(alt.datum.cond == "NW_UN").properties(
            title="Nonword"
        )

        return word | nonword

    def plot_mean_dev(self, show_sd, by_cond=True, interactive=False, baseline=None):
        """Plot the mean development of all selected models
        interactive = True for plot_interactive() ONLY!
        baseline: Overlay a baseline plot
        show_sd: Show SD in plot or not
        by_cond: True: plot condition in separate line; False: Aggregate condition before plotting
        """

        if by_cond:
            pdf = self.df

        else:
            group_var = [
                "code_name",
                "hidden_units",
                "p_noise",
                "learning_rate",
                "epoch",
            ]
            pdf = self.df.groupby(group_var).mean().reset_index()

        development_space_mean = (
            alt.Chart(pdf)
            .mark_line()
            .encode(
                x="epoch:Q", y=alt.Y("mean(score):Q", scale=alt.Scale(domain=(0, 1)))
            )
        )

        development_space_sd = (
            alt.Chart(pdf)
            .mark_errorband(extent="stdev")
            .encode(
                y=alt.Y("score:Q", scale=alt.Scale(domain=(0, 1))),
                x="epoch:Q",
            )
        )

        if by_cond:
            development_space_sd = development_space_sd.encode(
                color=alt.Color(
                    "cond:N",
                    legend=alt.Legend(legendX=300, legendY=180),
                    title="Condition",
                )
            )

        if interactive:
            development_space_sd = development_space_sd.transform_filter(
                self.select_control_space
            )

        # Add Mean
        development_space_mean = development_space_sd.mark_line().encode(
            y="mean(score):Q"
        )

        if show_sd:
            development_space_mean += development_space_sd

        if baseline is not None:
            development_space_mean += baseline

        return development_space_mean

    def plot_mean_wnw(self, baseline=None):
        """Plot all perforamance space only"""

        df = self.make_wnw()
        df = df.groupby("epoch").mean().reset_index()

        base_wnw = (
            alt.Chart(df)
            .mark_line(color="black")
            .encode(
                y=alt.Y("nonword_acc:Q", scale=alt.Scale(domain=(0, 1))),
                x=alt.X("word_acc:Q", scale=alt.Scale(domain=(0, 1))),
                tooltip=["epoch", "word_acc:Q", "nonword_acc:Q"],
            )
            .transform_calculate(
                color="if(datum.epoch===0.05, 'red', if(datum.epoch === 0.3, 'green', ''))"
            )
        )

        points = base_wnw.mark_circle(size=200).encode(
            color=alt.Color("color:N", scale=None)
        )

        base_wnw += points

        if baseline is not None:
            base_wnw = baseline + base_wnw

        base_wnw = self.diagonal + base_wnw

        return base_wnw

    def plot_heatmap_wadv(self, mode="dev"):
        """Plot word advantage heatmap
        mode: (dev)elopment or (per)formance
        """
        assert mode == "per" or mode == "dev"

        if mode == "dev":
            x = "p_noise:O"
            col = "epoch:O"
        elif mode == "per":
            x = alt.X("word_acc:Q", bin=alt.Bin(maxbins=20))
            col = "p_noise:O"
        else:
            raise Exception("Only dev or per mode")

        df = self.make_wnw()

        plot = (
            alt.Chart(df)
            .mark_rect()
            .encode(
                x=x,
                y=alt.Y("hidden_units:O", sort="descending"),
                row=alt.Row("learning_rate:O", sort="descending"),
                column=col,
                tooltip=["epoch", "word_acc:Q", "nonword_acc:Q"],
                color=alt.Color(
                    "word_advantage",
                    scale=alt.Scale(scheme="redyellowgreen", domain=(-0.3, 0.3)),
                ),
            )
        )

        return plot

    def plot_performance_multiline(self, var, legend_title):

        df = self.make_wnw()
        df = df.groupby(["epoch"] + [var]).mean().reset_index()

        scale_sort = "ascending" if var == "p_noise" else "descending"

        base = alt.Chart(df).encode(
            y=alt.Y("nonword_acc:Q", scale=alt.Scale(domain=(0, 1))),
            x=alt.X("word_acc:Q", scale=alt.Scale(domain=(0, 1))),
        )

        line = base.mark_line().encode(
            strokeDash=alt.StrokeDash(var, type="ordinal", sort=scale_sort),
            color=alt.Color(
                var,
                type="nominal",
                legend=alt.Legend(title=legend_title),
                sort=scale_sort,
            ),
            tooltip=[var, "epoch", "word_acc", "nonword_acc"],
        )

        epoch_marking = (
            base.mark_circle(size=200)
            .encode(
                color=alt.Color(
                    "color:N",
                    scale=None,
                    title=None,
                    legend=alt.Legend(title=legend_title),
                )
            )
            .transform_calculate(
                color="if(datum.epoch===0.05, 'red', if(datum.epoch === 0.3, 'green', ''))"
            )
        )
        # return line
        return line + epoch_marking + self.diagonal

    def plot_performance_in_cell(self, h, p, l):
        df = self.make_wnw()
        df = df.loc[
            (df.hidden_units == h) & (df.p_noise == p) & (df.learning_rate == l)
        ]
        df = df.groupby(["epoch"]).mean().reset_index()

        base = (
            alt.Chart(df)
            .mark_line(color="black")
            .encode(
                y=alt.Y("nonword_acc:Q", scale=alt.Scale(domain=(0, 1))),
                x=alt.X("word_acc:Q", scale=alt.Scale(domain=(0, 1))),
                tooltip=["epoch", "word_acc", "nonword_acc"],
            )
        )

        epoch_marking = (
            base.mark_circle(size=200)
            .encode(color=alt.Color("color:N", scale=None))
            .transform_calculate(
                color="if(datum.epoch===0.05, 'red', if(datum.epoch === 0.3, 'green', ''))"
            )
        )

        return base + epoch_marking + self.diagonal
