import ast
import re
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def from_project_root(path) -> str:
    root_path = Path(__file__).absolute().parent.parent
    return str(root_path.joinpath(path))


def plot_graphs(agent_losses,
                val_scores,
                max_loss=1.0,
                save_to_path=None,
                title=None,
                **kwargs):
    agent_losses = np.atleast_2d(agent_losses)
    val_scores = np.atleast_2d(val_scores)

    n_rows, n_cols = kwargs.get('grid') or (2, 1)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=kwargs.get('figsize') or (10, 8), facecolor='white')

    losses_df = pd.DataFrame(agent_losses).T
    first_non_zero_idx = losses_df.ne(0).prod(axis=1).idxmax()
    losses_df = losses_df.iloc[first_non_zero_idx:, :]

    losses_df.plot(
        use_index=True,
        # x='x',
        color='b',
        alpha=0.05,
        legend=False,
        xlabel='Episode',
        ylabel='Loss Value',
        ax=ax[0]
    )
    losses_df_mean = losses_df.mean(axis=1)
    losses_df_mean.plot(color='r', title=f'Loss Function', ax=ax[0])

    losses_df_stddev = losses_df.std(axis=1)
    losses_df_lower = losses_df_mean - losses_df_stddev
    losses_df_upper = losses_df_mean + losses_df_stddev

    ax[0].fill_between(
        np.arange(first_non_zero_idx, first_non_zero_idx + len(losses_df_lower)),
        losses_df_lower,
        losses_df_upper,
        facecolor='b',
        alpha=0.2
    )

    ax[0].set_xlabel('Episode')
    ax[0].set_ylim((0, max_loss))
    ax[0].set_title(f'Loss function')

    val_df = pd.DataFrame(val_scores).T
    val_x_start = 0 if kwargs.get('validate_at_start', False) else 1
    val_x = pd.DataFrame({'x': np.arange(val_x_start, len(val_scores[0])) * kwargs.get('validate_each', 1)})

    pd.concat(
        [val_df, val_x],
        axis=1
    ).plot(
        x='x',
        color='b',
        alpha=0.05,
        legend=False,
        xlabel='Episode',
        ylabel='Approximation Ratio',
        ax=ax[1]
    )
    val_df_mean = val_df.mean(axis=1)
    val_df_stddev = val_df.std(axis=1)
    val_df_lower = val_df_mean - val_df_stddev
    val_df_upper = val_df_mean + val_df_stddev

    print(val_df.to_numpy().min())
    print(val_df_lower.to_numpy().min())

    pd.concat([val_df_mean, val_x], axis=1).plot(x='x', color='r', legend=False, title=f'Validation scores', ax=ax[1])
    ax[1].fill_between(val_x.to_numpy().flatten(), val_df_lower, val_df_upper, facecolor='b', alpha=0.2)
    ax[1].set_xlabel('Episode')
    if max(val_df.to_numpy().min(), val_df_lower.to_numpy().min()) > 1:
        ax[1].set_ylim((1, None))
    ax[1].set_title('Validation scores')

    print(f'Min of avg validation score across episodes: {val_df_mean.min()}')

    # subtitle_keys = ['n_vertices', 'lr_config', 'batch_size']
    subtitle_keys = ['n_vertices', 'batch_size']
    subtitle = ', '.join([f'{k} = {kwargs[k]}' for k in subtitle_keys])
    title = title if title is not None \
        else f"{kwargs['problem'].upper()} - WDN"

    # fig.suptitle(f'{title}\n{subtitle}')
    fig.suptitle(title)
    plt.tight_layout()

    if save_to_path is not None:
        plt.savefig(save_to_path, facecolor='white', transparent=False)
    plt.show()


def replay_graphs(problem: str,
                  experiment_idx: int,
                  max_loss: float = 1.0,
                  save_img: bool = False,
                  extension: str = "png",
                  title: str = None,
                  figsize: tuple = None,
                  grid: tuple = None):
    problem = problem.lower()
    agent_losses, val_scores, config = load_experiment_files(problem, experiment_idx)

    print(f"{config=}")

    # Setup filename correctly
    filename_pattern = 'outputs-{}/run_{}{}.{}'.format(problem, experiment_idx, '{}', '{}')
    save_to_path = filename_pattern.format('', extension) if save_img else None

    plot_graphs(
        agent_losses,
        val_scores,
        max_loss=max_loss,
        save_to_path=save_to_path,
        title=title,
        n_vertices=config['n_vertices'],
        lr_config=config['lr_config'],
        batch_size=config['batch_size'],
        validate_each=config.get('validate_each', 1),
        validate_at_start=config.get('validate_at_start', False),
        problem=config['problem'],
        graph_type=config['graph_type'],
        figsize=figsize,
        grid=grid,
    )


def report_experiment_metrics(problem: str, experiment_idx: int):
    problem = problem.lower()
    agent_losses, val_scores, config = load_experiment_files(problem, experiment_idx)

    val_df = pd.DataFrame(val_scores).T
    val_df_mean = val_df.mean(axis=1)
    val_df_stddev = val_df.std(axis=1)

    idx = val_df_mean.argmin()
    smallest_mean = val_df_mean.iloc[idx]
    stddev_smallest_mean = val_df_stddev.iloc[idx]
    print("{} / {} / {} {}: {:.4f} ± {:.4f}".format(
        config['n_vertices'],
        'wdn',
        config['graph_params'],
        '/ ' + str(config.get('env_extra_params')) if len(config.get('env_extra_params', {})) > 0 else '',
        smallest_mean,
        stddev_smallest_mean
    ))


def load_experiment_files(problem: str, experiment_idx: int):
    problem = problem.lower()

    # Setup filename correctly
    filename_pattern = 'outputs-{}/run_{}{}.{}'.format(problem, experiment_idx, '{}', '{}')

    # Load losses
    with open(filename_pattern.format('_loss', 'log')) as f:
        agent_losses_str = f.read()
    agent_losses = ast.literal_eval(agent_losses_str)

    # Load approximation ratios
    with open(filename_pattern.format('_val', 'log')) as f:
        val_scores_str = f.read()
    val_scores = np.array(ast.literal_eval(val_scores_str))

    # Load config params
    with open(filename_pattern.format('', 'log')) as f:
        config_str = f.readlines()[0].strip()
    config_parsed = re.sub(r"('graph_type'): <[^\s]+:\s([^>]+)>", r"\1: \2", config_str)

    # print(f"{config_parsed=}")
    config = ast.literal_eval(config_parsed)

    return agent_losses, val_scores, config

# def plot_tsp_paths(graph, *paths, draw_all_edges=True) -> typing.Tuple[any, any]:
def plot_tsp_paths(graph, *paths, draw_all_edges=True, ax=None, **kwargs):
    """Plots the network and a set of paths if specified"""

    if ax is None:
        figsize = kwargs.get("figsize", (10, 8))
        fig, ax = plt.subplots(figsize=figsize)

    # Use coords attribute
    coords = {u: graph.nodes[u]["coords"] for u in graph.nodes}

    node_size = kwargs.get("node_size", 25)

    if not paths:
        nx.draw_networkx_nodes(graph, coords, node_size=node_size, ax=ax)
    else:
        nx.draw_networkx_nodes(graph, coords, node_size=node_size, node_color="#808080", ax=ax)
        all_nodes = set()
        for path in paths:
            for u in path:
                all_nodes.add(u)
        nx.draw_networkx_nodes(
            graph.subgraph(list(all_nodes)),
            coords,
            node_size=node_size,
            ax=ax
        )

    # nx.draw_networkx_labels(graph, coords, font_size=12, font_color="white")

    if paths is None and draw_all_edges:
        nx.draw_networkx_edges(graph, coords, edgelist=list(graph.edges), width=1, ax=ax)
    else:
        if draw_all_edges:
            nx.draw_networkx_edges(graph, coords, edgelist=list(graph.edges), width=0.1, ax=ax)
        for path_idx, path in enumerate(paths):
            edge_list = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            if len(path) > 0 and path[0] != path[-1]:
                edge_list.append((path[-1], path[0]))
            color_list = ['black', 'red']
            nx.draw_networkx_edges(
                graph,
                coords,
                edgelist=edge_list,
                width=1,
                edge_color=color_list[path_idx % len(color_list)],
                ax=ax
            )

    # plt.show()
