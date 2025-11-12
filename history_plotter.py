# plot_run_logs.py

"""
Plot convergence graphs for iterative algorithms from multiple run log files.

This script loads one or more Excel workbooks where each sheet contains
per‑iteration logs for a particular algorithm. You can specify which
columns to use for the x and y axes and provide custom labels for the
legend.  The resulting figure uses a palette of high‑contrast colours and
distinct line styles so it remains readable in grayscale.  The plot is
saved as a PNG with 300 dpi, suitable for scientific papers.
"""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_run_data(paths: List[str], x_col: str, y_col: str, sheet_nums: list[list[int]]|None = None) -> List[Tuple[pd.Series, pd.Series]]:
    """Load per‑iteration data from a list of Excel workbooks."""
    curves: List[Tuple[pd.Series, pd.Series]] = []
    for file_idx, path in enumerate(paths):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Excel file not found: {path}")
        xls = pd.ExcelFile(path)
        file_sheet_nums = sheet_nums[file_idx] if sheet_nums is not None else range(len(xls.sheet_names))
        for sheet_idx in file_sheet_nums:
            sheet = xls.sheet_names[sheet_idx]
            df = pd.read_excel(xls, sheet_name=sheet)
            if x_col not in df.columns or y_col not in df.columns:
                print(
                    f"Warning: missing '{x_col}' or '{y_col}' in sheet "
                    f"'{sheet}' of file '{path}'. Skipping this sheet."
                )
                continue
            x_series = pd.to_numeric(df[x_col], errors='coerce')
            y_series = pd.to_numeric(df[y_col], errors='coerce')
            curves.append((x_series, y_series))
    return curves


def plot_convergence(
    curves: List[Tuple[pd.Series, pd.Series]],
    legend_labels: List[str],
    x_label: str,
    y_label: str,
    max_x: int|float|None,
    output_file: str,
    color_style_cycle: list
) -> None:
    """Plot convergence curves and save to a PNG file."""
    if len(curves) != len(legend_labels):
        raise ValueError(
            f"Number of curves ({len(curves)}) does not match number of legend "
            f"labels ({len(legend_labels)})."
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (x_data, y_data) in enumerate(curves):
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        if max_x is not None:
            mask = x_data <= max_x
            x_data_cut, y_data_cut = x_data[mask], y_data[mask]
        else:
            x_data_cut, y_data_cut = x_data, y_data

        color_style = color_style_cycle[idx % len(color_style_cycle)]

        ax.plot(
            x_data_cut,
            y_data_cut,
            label=legend_labels[idx],
            color=color_style['color'],
            linestyle=color_style['linestyle'],
            marker=color_style['marker'],
            linewidth=2,
            markersize=4,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_yscale('log')
    # ax.grid(True, which='both', linestyle=':', linewidth=0.5)
    ax.legend()
    if (out_dir := os.path.dirname(output_file)) and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    # Paul Tol’s high‑contrast palette and varied line styles for grayscale printing
    color_cycle = [
        {'color': '#332288'},
        {'color': '#88CCEE'},
        {'color': '#DDCC77'},
        {'color': '#117733'},
        {'color': '#CC6677'},
        {'color': '#AA4499'},
    ]

    style_cycle = [
        {'linestyle': '-', 'marker': ''},
        {'linestyle': '--', 'marker': ''},
        {'linestyle': '-.', 'marker': ''},
        {'linestyle': ':', 'marker': ''},
        {'linestyle': '-', 'marker': 'o'},
        {'linestyle': '--', 'marker': 's'},
    ]
    # User configuration

    out_dir: str = '/home/sd/prj/thesis/PyProgs/MethodsCompare/storage/stats/MCIT-25-26-2025-11/talk_plots'

    # PseudoMono3 oper, MCIT-25-26
    # output_image: str = 'ex2_sn_2cases.png'
    # file_paths: List[str] = [
    #     '/home/sd/prj/thesis/PyProgs/MethodsCompare/storage/stats/MCIT-25-26-2025-11/PseudoMonotoneOperOne-2025-11-02_21-39-22/history-PseudoMonotoneOperOne-2025-11-02_21-39-22.xlsx',
    #     '/home/sd/prj/thesis/PyProgs/MethodsCompare/storage/stats/MCIT-25-26-2025-11/PseudoMonotoneOperOne-2025-11-02_21-54-37/history-PseudoMonotoneOperOne-2025-11-02_21-54-37.xlsx',
    #     '/home/sd/prj/thesis/PyProgs/MethodsCompare/storage/stats/MCIT-25-26-2025-11/PseudoMonotoneOperOne-2025-11-03_19-02-06/history-PseudoMonotoneOperOne-2025-11-03_19-02-06.xlsx'
    # ]

    # 4 - ||x|| oper, MCIT-25-26
    # output_image: str = 'ex1_sn_2cases.png'
    # file_paths: List[str] = [
    #     '/home/sd/prj/thesis/PyProgs/MethodsCompare/storage/stats/MCIT-25-26-2025-11/PseudoMonotoneOperAMinusNorm-2025-11-03_20-05-27/history-PseudoMonotoneOperAMinusNorm-2025-11-03_20-05-27.xlsx',
    #     '/home/sd/prj/thesis/PyProgs/MethodsCompare/storage/stats/MCIT-25-26-2025-11/PseudoMonotoneOperAMinusNorm-2025-11-03_20-08-22/history-PseudoMonotoneOperAMinusNorm-2025-11-03_20-08-22.xlsx',
    # ]

    # small TE problem, MCIT-25-26
    # output_image: str = 'traffic1_2cases.png'
    # file_paths: List[str] = [
    #     '/home/sd/prj/thesis/PyProgs/MethodsCompare/storage/stats/MCIT-25-26-2025-11/TrafficEquilibrium-2025-11-11_17-09-02/history-TrafficEquilibrium-2025-11-11_17-09-02.xlsx',
    #     '/home/sd/prj/thesis/PyProgs/MethodsCompare/storage/stats/MCIT-25-26-2025-11/TrafficEquilibrium-2025-11-11_17-09-27/history-TrafficEquilibrium-2025-11-11_17-09-27.xlsx',
    # ]

    # PseudoMono3 oper, MCIT-25-26 - with step increase also
    output_image: str = 'ex2_dn_3algs_inc.png'
    file_paths: List[str] = [
        '/home/sd/prj/thesis/PyProgs/MethodsCompare/storage/stats/MCIT-25-26-2025-11/PseudoMonotoneOperOne-2025-11-11_17-51-45/history-PseudoMonotoneOperOne-2025-11-11_17-51-45.xlsx'
    ]

    x_column: str = 'iterations'
    x_label: str = 'Iteration'

    # real_error or step_delta_norm
    y_column: str = 'real_error'

    # D_n or S_n
    y_label: str = '$D_n$'

    max_x = None
    # max_x = 400

    sheet_nums: List[int] = [[0, 1, 2]]
    legend_names: List[str] = ['Alg. 1, $\\alpha=0.2$', 'Alg. 2, $\\alpha_0=1, \  \\tau=0.49$',
                               'Alg. 2+, $\\alpha_0=1, \  \\tau=0.49$'
                               ]

    order = [0, 1, 2]
    # order = [0, 2, 1, 3]
    color_style_cycle = [
        {
            'color': color_cycle[0]['color'],
            'linestyle': style_cycle[1]['linestyle'],
            'marker': style_cycle[1]['marker']
        },
        {
            'color': color_cycle[2]['color'],
            'linestyle': style_cycle[2]['linestyle'],
            'marker': style_cycle[2]['marker']
        },
        {
            'color': color_cycle[1]['color'],
            'linestyle': style_cycle[0]['linestyle'],
            'marker': style_cycle[0]['marker']
        },
        {
            'color': color_cycle[3]['color'],
            'linestyle': style_cycle[3]['linestyle'],
            'marker': style_cycle[3]['marker']
        },
    ]

    output_image = os.path.join(out_dir, output_image)

    curves = load_run_data(file_paths, x_column, y_column, sheet_nums)

    curves = [curves[i] for i in order]
    legend_names = [legend_names[i] for i in order]

    plot_convergence(curves, legend_names, x_label, y_label, max_x, output_image, color_style_cycle)
    print(f"Plot saved to {output_image}")


if __name__ == '__main__':
    main()
