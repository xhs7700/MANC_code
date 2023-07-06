from math import inf
import numpy as np
import matplotlib.pyplot as plt
import toml
import os
from os import path

figsize = (8, 8)
d = {0: '(a)', 1: '(b)', 2: '(c)', 3: '(d)'}
markers = "so^+xD"


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


def compare_effects_optimum(toml_name, input_dir, output_dir, algos):
    toml_path = path.join(input_dir, f'{toml_name}.toml')
    with open(toml_path, "r") as f:
        toml_obj = toml.load(f)
    fig, axs = plt.subplots(1, 4, figsize=(16, 4), layout='tight')
    for i, (graph_name, mancs) in enumerate(toml_obj.items()):
        ax = axs[i]
        n = len(mancs[algos[0]])
        x = np.arange(1, n + 1)
        max_manc, min_manc = 0.0, inf
        for j, algo in enumerate(algos):
            max_manc = max(max_manc, max(mancs[algo]))
            min_manc = min(min_manc, min(mancs[algo]))
            ax.plot(x, mancs[algo], label=algo, marker=markers[j])
        ax.set_xlabel(r'Size $k$ of node set $S$', math_fontfamily='cm')
        ax.set_ylabel(r'MANC $H(S)$', math_fontfamily='cm')
        # ax.set_title(f'{d[i]} {graph_name}')
        ax.text(2, (max_manc - min_manc) * 0.6 + min_manc, graph_name)
        ax.legend()
    plt.savefig(path.join(output_dir, f'{toml_name}.eps'))
    plt.close(fig)


def compare_effects(toml_name, input_dir, output_dir, algos):
    toml_path = path.join(input_dir, f'{toml_name}.toml')
    with open(toml_path, "r") as f:
        toml_obj = toml.load(f)
    fig, axs = plt.subplots(2, 2, figsize=figsize, layout='tight')
    for i, (graph_name, mancs) in enumerate(toml_obj.items()):
        ax = axs[(i >> 1) & 1, i & 1]
        n = len(mancs[algos[0]])
        x = np.arange(1, n + 1)
        max_manc, min_manc = 0.0, inf
        for _, algo in enumerate(algos):
            max_manc = max(max_manc, max(mancs[algo]))
            min_manc = min(min_manc, min(mancs[algo]))
            ax.plot(x, mancs[algo], label=algo)
        ax.set_xlabel(r'Size $k$ of node set $S$', math_fontfamily='cm')
        ax.set_ylabel(r'MANC $H(S)$', math_fontfamily='cm')
        # ax.set_title(f'{d[i]} {graph_name}')
        ax.text((n + 1) / 2, (max_manc - min_manc) * 0.5 + min_manc,
                graph_name)
        ax.legend()
    plt.savefig(path.join(output_dir, f'{toml_name}.eps'))
    plt.close(fig)


def margin_errors(toml_name, input_dir, output_dir, factors, label_prefix):
    toml_path = path.join(input_dir, f'{toml_name}.toml')
    with open(toml_path, "r") as f:
        toml_obj = toml.load(f)
    fig, axs = plt.subplots(2, 2, figsize=figsize, layout='tight')
    x_pos = np.arange(6)
    xtick_label = [
        r'$(0,0.1]$', r'$(0.1,0.2]$', r'$(0.2,0.3]$', r'$(0.3,0.4]$',
        r'$(0.4,0.5]$', r'$(0.5,\infty]$'
    ]
    w = 0.2
    for i, (graph_name, errors) in enumerate(toml_obj.items()):
        ax = axs[(i >> 1) & 1, i & 1]
        for j, factor in enumerate(factors):
            ax.bar(x_pos + j * w,
                   height=errors[factor]['distribution'],
                   width=w,
                   label=f'{label_prefix} = {factor}')
        ax.set_xlabel(r'$error_u$')
        ax.set_ylabel('ratio')
        ax.set_xticks(x_pos + 1.5 * w, xtick_label, fontsize='small')
        ax.set_title(f'{d[i]} {graph_name}')
        ax.legend()
    plt.savefig(path.join(output_dir, f'{toml_name}.eps'))
    plt.close(fig)


plt.rc('font', family='serif', size=13)

toml2dat("compare_effects_exact", "outputs", "compare_effects/exact",
         ["Top-Absorb", "Exact", "Top-Degree", "Approx", "Top-PageRank"], 5)

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
