"""
visualization - Módulo para visualización de datos

Incluye funciones para exploración de variables categóricas y numéricas.
"""

from .plots import (
    categorical_exploration,
    numerical_exploration,
    plot_shap_values,
    plot_threshold_metrics,
    plot_scores,
    lift_plot,
    plot_shadowed_lift,
    shap_values_importance_percent,
    plot_pivot_table,
    plot_shap_value_custom
)

__all__ = [
    'categorical_exploration',
    'numerical_exploration',
    'plot_shap_values',
    'plot_threshold_metrics',
    'plot_scores',
    'lift_plot',
    'plot_shadowed_lift',
    'shap_values_importance_percent',
    'plot_pivot_table',
    'plot_shap_value_custom'
]
