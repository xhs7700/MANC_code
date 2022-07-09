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


compare_effects("compare_effects_exact", "outputs", "images",
                ["rank", "exact", "degree", "absorb"])
