import numpy as np
import matplotlib.pyplot as plt
import toml
from os import path

figsize = (10, 10)


def compare_effects(toml_name, input_dir, output_dir, algos):
    toml_path = path.join(input_dir, f'{toml_name}.toml')
    with open(toml_path, "r") as f:
        toml_obj = toml.load(f)
    for graph_name, mancs in toml_obj.items():
        fig, ax = plt.subplots(figsize=figsize, layout='tight')
        n = len(mancs[algos[0]])
        x = np.arange(1, n + 1)
        for algo in algos:
            ax.plot(x, mancs[algo], label=algo)
        ax.set_xlabel(r'$k$')
        ax.set_ylabel(r'$M(S)$')
        ax.set_title(graph_name)
        ax.legend()
        plt.savefig(path.join(output_dir, f'{toml_name}_{graph_name}.svg'))
        plt.close(fig)


def margin_errors(toml_name, input_dir, output_dir, factors, label_prefix):
    toml_path = path.join(input_dir, f'{toml_name}.toml')
    with open(toml_path, "r") as f:
        toml_obj = toml.load(f)
    x_pos = np.arange(6)
    xtick_label = [
        r'$(0,0.1]$', r'$(0.1,0.2]$', r'$(0.2,0.3]$', r'$(0.3,0.4]$',
        r'$(0.4,0.5]$', r'$(0.5,\infty]$'
    ]
    w = 0.2
    for graph_name, errors in toml_obj.items():
        fig, ax = plt.subplots(figsize=figsize, layout='tight')
        for j, factor in enumerate(factors):
            ax.bar(x_pos + j * w,
                   height=errors[factor]['distribution'],
                   width=w,
                   label=f'{label_prefix} = {factor}')
        ax.set_xlabel('error')
        ax.set_ylabel('ratio')
        ax.set_xticks(x_pos + 1.5 * w, xtick_label)
        ax.set_title(graph_name)
        ax.legend()
        plt.savefig(path.join(output_dir, f'{toml_name}_{graph_name}.svg'))
        plt.close(fig)


# compare_effects("compare_effects_exact", "outputs", "images",
#                 ["rank", "exact", "degree", "absorb"])

margin_errors("margin_errors", "outputs", "images", ["20", "50", "100", "200"],
              r"$c_{JL}$")
