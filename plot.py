from math import inf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import toml
import os
from os import path

pgf_preamble = "\n".join([
    r"\usepackage[T1]{fontenc}",
    r"\usepackage{amsfonts}",
    r"\usepackage[cmintegrals]{newtxmath}",
    r"\usepackage{fontspec}",
    r"\usepackage{amsmath}",
    r"\setmainfont{Times New Roman}",
])

plt.rcParams.update({
    "figure.figsize": (7.5, 7.5),
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times New Roman",
    "font.size": 14.0,
    "pgf.texsystem": "xelatex",
    "pgf.rcfonts": False,
    "pgf.preamble": pgf_preamble,
    "xtick.color": "black",
    "ytick.color": "black",
    # "ytick.minor.visible": True,
    "axes.labelcolor": "black",
    "axes.labelsize": "x-large",
    "axes.edgecolor": "black",
    "lines.linewidth": 1.0,
    "legend.fancybox": False,
    "legend.framealpha": 1.0,
    "legend.fontsize": 10.75,
    "savefig.dpi": 1200,
})

d = ['(a)', '(b)', '(c)', '(d)']
markers = 'pso*d'
markersizes = [10, 6, 6, 8, 6]


def toml2dat(toml_name, input_dir, output_dir, algos, step=1):
    toml_path = path.join(input_dir, f'{toml_name}.toml')
    with open(toml_path, "r") as f:
        toml_obj = toml.load(f)
    output_dir = path.join(input_dir, output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for (graph_name, mancs) in toml_obj.items():
        graph_dir = os.path.join(output_dir, graph_name)
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)
        for algo in algos:
            algo_path = os.path.join(graph_dir, f'{algo}.dat')
            with open(algo_path, "w") as f:
                f.write('k\tAGC\n')
                for i, manc in enumerate(mancs[algo]):
                    f.write(f'{(i+1)*step}\t{manc}\n')


def compare_effects_optimum(toml_name: str,
                            input_dir: str,
                            output_dir: str,
                            algo_infos: dict[str, str],
                            scale_type: str,
                            step: int = 1):
    toml_path = path.join(input_dir, f'{toml_name}.toml')
    with open(toml_path, "r") as f:
        toml_obj = toml.load(f)
    fig, axs = plt.subplots(2, 2, layout='compressed')
    for i, (graph_name, mancs) in enumerate(toml_obj.items()):
        ax = axs[(i >> 1) & 1, i & 1]
        n = len(list(mancs.values())[0])
        x = np.arange(step, n * step + 1, step, dtype=np.int32)
        for j, (algo, info) in enumerate(algo_infos.items()):
            ax.plot(x,
                    mancs[algo],
                    label=info,
                    marker=markers[j],
                    ms=markersizes[j])
        ax.set_yscale(scale_type)
        ax.set_xlabel(r'$k$')
        ax.set_ylabel(r'$H\left(S\right)$')
        ax.text(0.02, 0.03, d[i], transform=ax.transAxes, fontsize='x-large')
        ax.legend(ncols=2, loc='upper right')
    plt.savefig(path.join(output_dir, f'{toml_name}.pdf'), backend='pgf')
    plt.close(fig)


def compare_effects(toml_name: str,
                    input_dir: str,
                    output_dir: str,
                    algo_infos: dict[str, str],
                    scale_type: str,
                    step: int = 1):
    toml_path = path.join(input_dir, f'{toml_name}.toml')
    with open(toml_path, "r") as f:
        toml_obj = toml.load(f)
    fig, axs = plt.subplots(2, 2, layout='compressed')
    for i, mancs in enumerate(toml_obj.values()):
        pos = ((i >> 1) & 1, i & 1)
        ax = axs[pos]
        n = len(list(mancs.values())[0])
        x = np.arange(step, n * step + 1, step, dtype=np.int32)
        for j, (algo, info) in enumerate(algo_infos.items()):
            ax.plot(x,
                    mancs[algo],
                    label=info,
                    marker=markers[j],
                    ms=markersizes[j])
        ax.set_yscale(scale_type)
        ax.set_xlabel(r'$k$')
        ax.set_ylabel(r'$H\left(S\right)$')
        ax.text(0.02, 0.03, d[i], transform=ax.transAxes, fontsize='x-large')
    fig.legend(
        handles=axs[0, 0].get_lines(),
        #    bbox_to_anchor=(0., 1.02, 1., .102),
        loc='outside upper right',
        # mode='expand',
        ncols=5)
    plt.savefig(path.join(output_dir, f'{toml_name}.pdf'), backend='pgf')
    plt.close(fig)


# toml2dat("compare_effects_exact", "outputs", "compare_effects/exact",
#          ["Top-Absorb", "Exact", "Top-Degree", "Approx", "Top-PageRank"], 5)

# compare_effects_optimum(
#     "compare_effects_optimum", "outputs", "outputs", {
#         "Exact": r"\textsc{Deter}",
#         "Approx": r"\textsc{Approx}",
#         "Optimum": r"\textsc{Optimum}",
#         "Random": r"\textsc{Random}",
#     }, 'linear')

# compare_effects(
#     "compare_effects_exact", "outputs", "outputs", {
#         "Exact": r"\textsc{Deter}",
#         "Approx": r"\textsc{Approx}",
#         "Top-Absorb": r"\textsc{Top-Absorb}",
#         "Top-PageRank": r"\textsc{Top-PageRank}",
#         "Top-Degree": r"\textsc{Top-Degree}",
#     }, 'linear', 5)

# toml2dat("compare_effects_optimum", "outputs", "compare_effects/optimum",
#          ["Exact", "Approx", "Optimum"])

# compare_effects("compare_effects_exact", "outputs", "images",
#                 ["Top-Absorb", "Exact", "Top-Degree", "Approx"])

# compare_effects_optimum("compare_effects_optimum", "outputs", "images",
#                         ["Exact", "Approx", "Optimum"])

# compare_effects("compare_effects_optimum", "outputs", "images",
#                 ["Exact", "Approx", "Optimum"])

# margin_errors("margin_errors", "outputs", "images", ["20", "50", "100", "200"],
#               r"$c_{JL}$")
