"""Exploration functions"""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.gridspec import GridSpec
from scipy.stats import probplot
from utils import get_logger, transform_to_basic_type


def find_outliers_iqr(df: pd.DataFrame, col_name: str):
    """Find the outliers and log them

    Args:
        df (pd.DataFrame): the dataframe with all the data
        col_name (str): the column to analyze
    """
    feature = df[col_name]
    q1 = feature.quantile(0.25)
    q3 = feature.quantile(0.75)
    iqr = q3-q1
    outliers = feature[((feature < (q1-1.5*iqr)) | (feature > (q3+1.5*iqr)))]
    count_outliers = len(outliers)

    logger = get_logger()
    logger.info(
        "Outliers: %s / %s (%s%%)",
        count_outliers, len(feature),
        round((count_outliers * 100)/len(feature), 2)
    )
    logger.info("Lower Bound: %s", f'{q1-1.5*iqr:.4f}')
    logger.info("Upper Bound: %s", f'{q3+1.5*iqr:.4f}')


def numerical_exploration(
        df: pd.DataFrame, col_name: str, objective_column: str,
        stacked_buckets: float = 1.0, prob_plot: bool = True,
        sample_size: int = -1, random_state: int = 123):
    """The default numerical exploration

    Args:
        df (pd.DataFrame): the dataframe with the information
        col_name (str): the name of the column to analyze
        objective_column (str): the column which has the variable to learn
        stacked_buckets (float, optional): a number used to stack bins for the
        last graph. Defaults to 1.0.
        prob_plot (bool): boolean to plot probability plot. Defaults to True
        sample_size (int): the size of the sample
        random_state (int): the random state for the sample
    """
    logger = get_logger()
    logger.info('Description%s\n%s', '-'*20, df[col_name].describe())

    nan_count = df[col_name].isna().sum()
    logger.info(
        'Nan %s / %s (%s%%)', nan_count, len(df[col_name]),
        round(nan_count * 100 / len(df[col_name]), 2)
    )
    find_outliers_iqr(df, col_name)
    logger.info('-'*20)

    df_sample = df
    if sample_size > 0:
        df_sample = df.sample(
            sample_size,
            random_state=random_state
        )

    plot_numerical_graphs(
        df_sample, col_name, objective_variable=objective_column,
        stacked_buckets=stacked_buckets, prob_plot=prob_plot
    )


def plot_numerical_graphs(
        df: pd.DataFrame, col_name: str, stacked_buckets: float = 1.0,
        objective_variable: str = 'sale', prob_plot: bool = True):
    """The graphical numerical exploration

    Args:
        df (pd.DataFrame): the dataframe with the data
        col_name (str): the name of the column to explore
        stacked_buckets (float, optional): a number used to stack bins for the
        last graph. Defaults to 1.0.
        objective_variable (str, optional): the name of the variable to learn.
        Defaults to 'sale'.
        prob_plot (bool): boolean to plot probability plot. Defaults to True
    """
    logger = get_logger()
    logger.debug('Create subplots')
    fig = plt.figure(layout="tight", figsize=(18, 18))
    gs = GridSpec(6, 2, figure=fig)

    fig.suptitle('Plots for: '+col_name, fontsize=18)

    # Drop na for graphic
    columns_to_use = [col_name]
    if objective_variable:
        columns_to_use.append(objective_variable)
    df_no_na = df[columns_to_use].dropna()
    df_no_na = transform_to_basic_type(df_no_na)

    # Original data & plot probability
    logger.debug('Histogram')
    ax1 = fig.add_subplot(gs[0, :])
    sns.histplot(data=df_no_na, x=col_name, ax=ax1, kde=True)
    ax1.set_title('Histogram and density')

    logger.debug('Histogram without outliers')
    q1 = df_no_na[col_name].quantile(0.25)
    q3 = df_no_na[col_name].quantile(0.75)
    iqr = q3-q1
    ax1 = fig.add_subplot(gs[1, :])
    df_tmp = df_no_na[[col_name]].copy()
    df_tmp = df_tmp[
        (df_tmp[col_name] >= (q1-1.5*iqr))
        & (df_tmp[col_name] <= (q3+1.5*iqr))
    ]
    sns.histplot(data=df_tmp, x=col_name, ax=ax1, kde=True)
    ax1.set_title('Histogram without outliers')

    # boxplot
    logger.debug('Boxplot')
    ax2 = fig.add_subplot(gs[2, 0])
    sns.boxplot(data=df_no_na, x=col_name, ax=ax2, showfliers=True)
    ax2.set_title('Boxplot with outliers')

    # boxplot
    logger.debug('Boxplot without outliers')
    ax3 = fig.add_subplot(gs[2, 1])
    sns.boxplot(data=df_no_na, x=col_name, ax=ax3, showfliers=False)
    ax3.set_title('Boxplot without outliers')

    # prob plot
    if prob_plot:
        logger.debug('Quantile quantile plot')
        ax4 = fig.add_subplot(gs[3, 0])
        probplot(df_no_na[col_name], plot=ax4)
        ax4.set_title('Probability plot')

    # boxplot
    if not objective_variable:
        plt.tight_layout()
        return

    logger.debug('Boxplot over objective')
    colors_quantity = len(df_no_na[objective_variable].unique())
    if colors_quantity == 2:
        colors = ['#ff6961', '#8de5a1']
    else:
        colors = sns.color_palette("Set2", colors_quantity)

    ax5 = fig.add_subplot(gs[3, 1])
    split_category = df_no_na[objective_variable].astype('category')
    sns.boxplot(
        data=df_no_na, x=col_name, y=split_category,
        ax=ax5, showfliers=False, hue=split_category,
        palette=colors
    )
    ax5.set_title(f'Boxplot by {objective_variable}')

    # Stacked
    logger.debug('Stacked histogram')
    ax6 = fig.add_subplot(gs[4, :])
    flored_objective_stacked = np.floor(df_no_na[col_name]*stacked_buckets)
    df_plot = df_no_na.groupby(flored_objective_stacked / stacked_buckets)[
        objective_variable].value_counts(normalize=True).unstack(
        objective_variable)
    df_plot.plot.bar(ax=ax6, color=colors)
    ax6.set_title(f'Histogram stacked 100% by {objective_variable}')

    # Stacked
    logger.debug('Stacked histogram, only objective values')
    ax7 = fig.add_subplot(gs[5, :])
    highest_value = np.sort(df_plot.columns)[-1]
    df_plot[highest_value].fillna(0).plot.line(ax=ax7, color=colors[-1], style='-D')
    ax7.set_title(f'{objective_variable} %100 per stack')

    logger.debug('Plot on screen')
    plt.show()

    logger.info('Finished')


def categorical_exploration(
        df: pd.DataFrame, col_name: str, objective_variable: str,
        sample_size: int = -1, random_state: int = 123):
    """Make the default categorical exploration

    Args:
        df (pd.DataFrame): the dataframe with the data
        col_name (str): the name of the column to explore
        objective_variable (str): the name of the column that is going to be learned
        sample_size (int): the size of the sample
        random_state (int): the random state for the sample
    """
    logger = get_logger()
    logger.debug('Create subplots')
    figure, axis = plt.subplots(1, 3, figsize=(20, 5))
    figure.suptitle('Categorical features')
    col = col_name

    nan_count = df[col_name].isna().sum()
    logger.info(
        'Nan %s / %s (%s%%)', nan_count, len(df[col_name]),
        round(nan_count * 100 / len(df[col_name]), 2)
    )

    df_sample = df
    if sample_size > 0:
        df_sample = df.sample(
            sample_size,
            random_state=random_state
        )

    # Drop na for graphic
    columns_to_use = [col_name]
    if objective_variable:
        columns_to_use.append(objective_variable)
    df_no_na = df_sample[columns_to_use].dropna()

    total_unique_values = len(df_no_na[col].unique())
    df_graphs = df_no_na
    if total_unique_values > 15:
        logger.debug('Group not top categorical values')
        top_10_values = df_no_na[col].value_counts().head(10).index
        df_graphs[col] = df_no_na[col].apply(
            lambda x: x if x in top_10_values else 'Other')

    logger.debug('%s distribution', col_name)
    axis[0].set_title(f'Percentage of registers per  {col}')
    pie_data = df_graphs.groupby([col])[col].count().reset_index(name="count")
    logger.debug('Pie chart data: \n%s', pie_data.head().to_string())
    # piechart
    axis[0].pie(
        pie_data['count'], labels=pie_data[col],
        textprops={'fontsize': 8}
    )

    # barchart
    logger.debug('%s count', col_name)
    axis[1].set_title(f'Registers per {col}')
    sns.countplot(ax=axis[1], y=col, data=df_graphs)
    axis[1].tick_params(labelrotation=0)  # , fontsize=7

    if not objective_variable:
        figure.tight_layout()
        plt.show()
        logger.info('Finished')
        return

    # percentage barchart
    logger.debug('%s per %s', col_name, objective_variable)
    c = df_graphs.groupby([col, objective_variable]).size().rename('count')
    bar_percentage_data = (
        c / c.groupby(level=0).sum()
    ).reset_index(name='proportion')
    bar_percentage_data_pivoted = bar_percentage_data.pivot(
        index=col, columns=objective_variable,
        values='proportion').reset_index()

    logger.debug('Boxplot over objective')
    unique_classes = df_graphs[objective_variable].unique()
    colors_quantity = len(unique_classes)
    if colors_quantity == 2:
        colors = ['#ff6961', '#8de5a1']
    else:
        colors = sns.color_palette("Set2", colors_quantity)

    filter_columns = [objective_variable, col_name]
    filter_columns.extend(unique_classes.tolist())
    bar_percentage_data_pivoted.filter(items=filter_columns).plot(
        x=col,
        kind='barh',
        # stacked = True,
        title=f'{objective_variable} per {col}',
        mark_right=True,
        ax=axis[2],
        color=colors,
        fontsize=8)

    figure.tight_layout()
    plt.show()
    logger.info('Finished')


def plot_shap_values(shap_values: np.array, max_display: int = None):
    """Create the shap values bar and summary plot

    Args:
        shap_values (np.array): the result of a shap values analysis
        max_display (int, optional): the amount of row to display. Defaults to
        None
    """
    shap.plots.bar(shap_values, max_display=max_display)
    shap.summary_plot(shap_values, max_display=max_display)


def plot_shap_value_custom(
        shap_values_model: List, probas: np.array, real_labels: np.array,
        index: int, max_display: int = 20):
    """Show a waterfall plot for a record

    Args:
        shap_values_model (List): the result of the shap analysis
        probas (np.array): the probability predicted for this case
        real_labels (np.array): the real value
        index (int): the index of the record to plot
        max_display (int, optional): the amount of features to show in the
        waterfall plot. Defaults to 20.
    """
    plt.figure(facecolor='white')
    proba = probas[index]
    real_label = real_labels[index]
    p = shap.plots.waterfall(
        shap_values_model[index], show=False, max_display=max_display
    )
    p.set_xlabel(
        f"Valor real: {real_label} Probabilidad: {proba}", fontweight='bold')
    plt.show()


def plot_threshold_metrics(df: pd.DataFrame):
    """Plot a bar graph showing the  thresholds and the metrics precision,
    recall, f1

    Args:
        df (pd.DataFrame): the data frame with the results per threshold
    """
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))
    fig.suptitle('Model metrics per threshold: ', fontsize=18)

    # Precision
    df.plot(x='threshold', y=['precision'], ax=axes[0])
    axes[0].set_title('Precision')

    # Precision
    df.plot(x='threshold', y=['recall'], ax=axes[1])
    axes[1].set_title('Recall')

    # Precision
    df.plot(x='threshold', y=['f1'], ax=axes[2])
    axes[2].set_title('F1')

    plt.tight_layout()
    plt.show()


def plot_scores(df: pd.DataFrame, figsize: (int, int) = (18, 6)):
    """Plot the metrics graphs. This will plot score vs threshold, FP - TP ROC

    Args:
        df (pd.DataFrame): the dataframe with the metrics
    """
    tpr = df['tp'] / (df['tp'] + df['fn'])
    fpr = df['fp'] / (df['fp'] + df['tn'])

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    fig.suptitle('Scores')

    plt.figure()

    lw = 2  # line width

    axes[0].plot(fpr, tpr, color="tab:blue", lw=lw, label="ROC")

    axes[0].plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    axes[0].set_xlim([-0.05, 1.05])
    axes[0].set_ylim([-0.05, 1.05])
    axes[0].set_xlabel("FP Rate")
    axes[0].set_ylabel("TP Rate")
    axes[0].set_title("ROC")
    axes[0].legend(loc="lower right")

    axes[1].plot(
        df['threshold'],
        df['precision'],
        color="tab:blue", lw=lw, label="Precision")
    axes[1].plot(
        df['threshold'],
        df['recall'],
        color="tab:orange", lw=lw, label="Recall")
    axes[1].plot(
        df['threshold'],
        df['f1'],
        color="tab:green", lw=lw, label="F1")

    axes[1].plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    axes[1].set_xlim([-0.05, 1.05])
    axes[1].set_ylim([-0.05, 1.05])
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Score vs Threshold")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.show()


def shap_values_importance_percent(
        shap_values: np.array, feature_names: List[str]):
    """Plot the importance of each feature by the shap values analysis as a
    percent between all features

    Args:
        shap_values (np.array): the result of the shap values
        feature_names (List[str]): the names of the features put into the shap
        values analysis
    """
    # Compute feature importances
    feature_importances = np.abs(shap_values.values).mean(axis=0)

    # Calculate the total feature importance
    total_importance = np.sum(feature_importances)

    # Calculate the percentage of importance for each feature
    percentage_importance = feature_importances / total_importance * 100

    shap_importance = pd.DataFrame(
        list(zip(feature_names, percentage_importance)),
        columns=['col_name', 'feature_importance_vals']
    )
    shap_importance.sort_values(
        by=['feature_importance_vals'],
        ascending=False, inplace=True
    )

    # Plot the feature importances as a bar plot\
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=shap_importance['feature_importance_vals'].values,
        y=shap_importance['col_name'].values,
    )
    plt.xlabel('Percentage of Total Importance')
    plt.ylabel('Feature')
    plt.title('Global Feature Importance as Percentage')
    plt.show()


def plot_pivot_table(
        df: pd.DataFrame, values: str = 'test_precision',
        rows: str = 'over_sampling', columns: str = 'under_sampling'):
    """Plot the iterations of oversampling, undersampling

    Args:
        df (pd.DataFrame): the dataframe with the results of the iterations
        values (str, optional): the column name with the values to plot.
        Defaults to 'test_precision'
        rows (str, optional): the column name to use as rows. Defaults to 'over_sampling'.
        columns (str, optional): the column name to use as columns. Defaults to 'under_sampling'.
    """
    pivot_df = pd.pivot_table(
        df, values=values, index=[rows], columns=[columns], aggfunc='max'
    )
    subplots = plt.subplots(figsize=(18, 5))
    ax = subplots[1]
    sns.heatmap(
        pivot_df, annot=True, fmt='.1%', linewidths=.1, ax=ax,
        cmap=sns.color_palette("Blues", as_cmap=True)
    )
    ax.set_title(values.replace('_', ' ').title())


def lift_plot(
        df_lift: pd.DataFrame, figsize: (int, int) = (16, 8)):
    """Make a plot showing the distribution of users in each probability given
    by the prediction of a mode, the perfect distribution, and the real distribution

    Args:
        df_lift (pd.DataFrame): the dataframe used to plot
        bins (List[float]): the cuts for the probabilities group
        fig_size ((int, int), optional): the size of the plot. Defaults to (16, 8)
    """
    # Create plot
    subplots = plt.subplots(figsize=figsize)
    ax1 = subplots[1]

    # Plotting the bar plot on ax1
    bar_plot = sns.barplot(
        x='probability_group_label', y='predictions_count', data=df_lift,
        ax=ax1, label='Users', color='#00b1f6')
    # Set y-axis label for ax1
    ax1.set_ylabel('Users', color='#00b1f6')
    ax1.set_xlabel('Probability group')

    # Create a twin axis sharing the same x-axis
    ax2 = ax1.twinx()

    # Overlaying the line plot on ax2
    line_plot = sns.lineplot(
        x='probability_group_label', y='real_value_percent', data=df_lift,
        color='#1f3a6e', sort=False, ax=ax2, label='Real value', linewidth=3
    )
    # Set y-axis label for ax2
    ax2.set_ylabel('Real value', color='#1f3a6e')

    ax1.tick_params(axis='y', labelright=True, labelleft=False)
    ax2.tick_params(axis='y', labelright=False, labelleft=True)

    ax1.yaxis.set_label_position("right")
    ax1.yaxis.set_ticks_position("right")

    ax2.yaxis.set_label_position("left")
    ax2.yaxis.set_ticks_position("left")

    df_perfect = pd.DataFrame()
    df_perfect['x'] = df_lift['probability_group_label']
    df_perfect['y'] = df_lift['probability_group_number'] / 100
    ax2.plot(
        df_perfect['x'], df_perfect['y'], ls='--',
        c='red', label='Perfect model'
    )

    handles1, labels1 = bar_plot.get_legend_handles_labels()
    handles2, labels2 = line_plot.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    bar_plot.get_legend().remove()
    plt.legend(
        handles, labels, loc='upper center',
        bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=3
    )

    plt.title('DualSIM prediction')
    plt.xlabel('probability_group_label')

    # Rotating x-axis labels for better readability
    plt.xticks(rotation=45)

    plt.show()


def plot_shadowed_lift(
        df_lift: pd.DataFrame, df_shadow: pd.DataFrame, intervals: List[dict],
        figsize: (int, int) = (14, 10)):
    """Create the shadowed lift plot

    Args:
        df_lift (pd.DataFrame): the lift plot values
        df_shadow (pd.DataFrame): the lift plot values, but with the risk
        segment intervals
        intervals (List[dict]): the definition of the risk segment intervals
        figsize (int, int, optional): the size of the plot. Defaults to (14, 10).
    """
    # Create plot
    subplots = plt.subplots(figsize=figsize)
    ax1 = subplots[1]

    # Plotting the bar plot on ax1
    sns.barplot(
        x='probability_group_number', y='predictions_count', data=df_lift,
        ax=ax1, label='Users', color='#00b1f6')
    plt.xticks(
        range(len(df_lift['probability_group_label'])),
        df_lift['probability_group_label']
    )
    # Set y-axis label for ax1
    ax1.set_ylabel('Users')
    ax1.set_xlabel('Probability group')

    keys = list(intervals.keys())
    xmin, xmax = ax1.get_xlim()
    total_x = xmax - xmin
    for index, key in enumerate(keys):
        value = intervals[key]

        if index == 0:
            plt.axvspan(
                xmin, xmin + (value['max'] * total_x),
                color=value['color'], alpha=0.3, zorder=0
            )
            midpoint = xmin + (xmin + (value['max'] * total_x) / 2)
            plt.text(
                midpoint, df_lift['predictions_count'].max() * 0.9,
                df_shadow['percent_label'].values[index],
                fontsize=24
            )
        elif index == len(keys) - 1:
            plt.axvspan(
                xmin + (value['min'] * total_x), xmax,
                color=value['color'], alpha=0.3, zorder=0
            )
            midpoint = (xmin + (value['min'] * total_x)) \
                + ((xmax - (xmin + (value['min'] * total_x))) / 2) * 0.7
            plt.text(
                midpoint, df_lift['predictions_count'].max() * 0.9,
                df_shadow['percent_label'].values[index],
                fontsize=24
            )
        else:
            midpoint = (xmin + (value['min'] * total_x)) \
                + (((xmin + (value['max'] * total_x))
                    - (xmin + (value['min'] * total_x))) / 2) * 0.7
            plt.text(
                midpoint, df_lift['predictions_count'].max() * 0.9,
                df_shadow['percent_label'].values[index],
                fontsize=24
            )
            plt.axvspan(
                xmin + (value['min'] * total_x),
                xmin + (value['max'] * total_x),
                color=value['color'], alpha=0.3, zorder=0
            )
