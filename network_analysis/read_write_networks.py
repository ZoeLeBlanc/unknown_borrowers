import pandas as pd
pd.options.mode.chained_assignment = None
import networkx as nx
from networkx.algorithms import bipartite
from igraph import *

def generate_nx_edgelist(G, delimiter=" ", data=True):
    """Copied from networkx documentation but had to rewrite this function because it kept mixing up the order of source and target nodes 😭"""
    try:
        part0 = [n for n, d in G.nodes.items() if d["bipartite"] == 0]
    except BaseException as e:
        raise AttributeError("Missing node attribute `bipartite`") from e
    if data is True or data is False:
        for n in part0:
            for e in G.edges(n, data=data):
                yield delimiter.join(map(str, e))
    else:
        for n in part0:
            for u, v, d in G.edges(n, data=True):
                e = [u, v]
                try:
                    e.extend(d[k] for k in data)
                except KeyError:
                    pass  # missing data for this edge, should warn?
                yield delimiter.join(map(str, e))


def get_edge_cols(bipartite_graph):
    """Dynamically get edge attributes but only need initial names"""
    keys = []
    for _, _, e in bipartite_graph.edges(data=True):
        keys = [*e]
        if len(keys) > 0:
            break
    return keys


def get_bipartite_edgelist(bipartite_graph):
    """Turn graph edgelist into pandas dataframe but keep bipartite groupings"""
    edges = []
    edge_cols = get_edge_cols(bipartite_graph)
    for line in generate_nx_edgelist(bipartite_graph, " ", data=edge_cols):
        data = line.split(' ')

        edges.append({'source': data[0], 'target': data[1], 'weight': data[2] if len(data) > 2 else 1})
    return pd.DataFrame(edges)


def generate_dataframes(graph, is_bipartite, is_networkx):
    """Generate dataframes from graph and write to file"""
    if is_networkx:
        nodes_df = pd.DataFrame.from_dict(
            dict(graph.nodes(data=True)), orient='index')
        nodes_df = nodes_df.reset_index(drop=True)
        if is_bipartite:
            edges_df = get_bipartite_edgelist(graph)
        else:
            edges_df = nx.to_pandas_edgelist(graph)
    else:
        nodes_df = graph.get_vertex_dataframe()
        nodes_df = nodes_df.reset_index()
        nodes_df = nodes_df.drop(columns=['_nx_name'])
        nodes_df = nodes_df.rename(columns={'vertex ID': 'node_id'})

        edges_df = graph.get_edge_dataframe()
        edges_df = edges_df.reset_index()
        edges_df = edges_df.drop(columns=['edge ID'])
    return (nodes_df, edges_df)


def write_dataframe(file_name, edges_df, nodes_df):
    """Write dataframes"""
    edges_df.to_csv(f'{file_name}_edgelist.csv', index=False)
    nodes_df.to_csv(f'{file_name}_nodelist.csv', index=False)

def combine_dataframes(graph_df, sco_df, columns, on_column, how_setting):
    joined_df = pd.merge(
        left=sco_df, right=graph_df[columns], on=on_column, how=how_setting)
    return joined_df

def write_graph(file_name, graph, is_networkx):
    """Write graph to file"""
    if is_networkx:
        nx.write_gexf(graph, f'{file_name}_graph.gexf')
    else:
        graph.write_graphml(f'{file_name}s_graph.graphml')
        graph_gml = nx.read_graphml(f'{file_name}_graph.graphml')
        nx.write_gexf(graph_gml, f'{file_name}_graph.gexf')
        if os.path.exists(f'{file_name}_graph.gexf'):
            os.remove(f'{file_name}_graph.graphml')
