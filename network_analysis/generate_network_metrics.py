import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
import networkx as nx
from networkx.algorithms import bipartite
import altair as alt
import numpy as np
import itertools
import collections
import warnings
warnings.filterwarnings("ignore")
from tqdm.notebook import trange, tqdm
tqdm.pandas()
from sknetwork.data import convert_edge_list
from sknetwork.clustering import Louvain, modularity, bimodularity
from sknetwork.ranking import Katz, PageRank
from sknetwork.linkpred import JaccardIndex
from igraph import *

import sys
sys.path.append("..")
from network_analysis.birankpy import BipartiteNetwork
from network_analysis.read_write_networks import * 

def get_bipartite_network_metrics(graph):
    '''Run network metrics for each node at various snapshots
    '''
    if nx.is_empty(graph):
        return graph
    top_nodes = [n for n in graph.nodes if graph.nodes[n]
                    ['group'] == 'members']
    bottom_nodes = [
        n for n in graph.nodes if graph.nodes[n]['group'] == 'books']
    # bottom_nodes, top_nodes = bipartite.sets(graph)
    graph_components = sorted(
        nx.connected_components(graph), key=len, reverse=True)
    #Degree

    print("calculating global degree")
    global_degree = bipartite.degree_centrality(graph, top_nodes)
    #Clustering
    print("calculating top global clustering")
    top_global_clustering = bipartite.clustering(graph, top_nodes)
    print("calculating bottom global clustering")
    bottom_global_clustering = bipartite.clustering(graph, bottom_nodes)
    #Closeness
    print("calculating global closeness")
    global_closeness = bipartite.closeness_centrality(graph, top_nodes)
    #Betweenness
    print("calculating global closeness")
    global_betweenness = bipartite.betweenness_centrality(graph, top_nodes)

    for index, component in enumerate(tqdm(graph_components)):
        subgraph = graph.subgraph(component)
        top_nodes = [
            n for n in subgraph.nodes if subgraph.nodes[n]['group'] == 'members']
        bottom_nodes = [
            n for n in subgraph.nodes if subgraph.nodes[n]['group'] == 'books']
        # bottom_nodes, top_nodes = bipartite.sets(subgraph)
        #Degree
        local_degree = bipartite.degree_centrality(subgraph, top_nodes)
        #Clustering
        top_local_clustering = bipartite.clustering(subgraph, top_nodes)
        bottom_local_clustering = bipartite.clustering(
            subgraph, bottom_nodes)
        #Closeness
        local_closeness = bipartite.closeness_centrality(
            subgraph, top_nodes)
        #Betweenness
        try:
            top_local_betweenness = bipartite.betweenness_centrality(
                subgraph, top_nodes)
        except ZeroDivisionError:
            top_local_betweenness = {k: 0 for k in top_nodes}

        try:
            bottom_local_betweenness = bipartite.betweenness_centrality(
                subgraph, bottom_nodes)
        except ZeroDivisionError:
            bottom_local_betweenness = {k: 0 for k in bottom_nodes}

        for d, v in subgraph.nodes(data=True):
            v['global_degree'] = global_degree[d]
            v['local_degree'] = local_degree[d]

            v['global_clustering'] = top_global_clustering[d] if 'members' in v['group'] else bottom_global_clustering[d]
            v['local_clustering'] = top_local_clustering[d] if 'members' in v['group'] else bottom_local_clustering[d]

            v['global_closeness'] = global_closeness[d]
            v['local_closeness'] = local_closeness[d]

            v['global_betweenness'] = global_betweenness[d]
            v['local_betweenness'] = top_local_betweenness[d] if 'members' in v['group'] else bottom_local_betweenness[d]

            v['node_title'] = d
            v['component'] = index

    return graph

def get_unipartite_network_metrics(graph):
    metrics = []
    global_degree = graph.degree()
    global_closeness = graph.closeness()
    global_betweenness = graph.betweenness()
    global_eigenvector = graph.eigenvector_centrality()
    global_clustering = graph.transitivity_local_undirected()
    global_radius = graph.radius()
    global_diameter = graph.diameter()

    for index, component in enumerate(tqdm(graph.decompose())):
        local_diameter = component.diameter()
        local_radius = component.radius()
        local_eigenvector = component.eigenvector_centrality()
        local_degree = component.degree()
        local_closeness = component.closeness()
        local_betweenness = component.betweenness()
        local_clustering = component.transitivity_local_undirected()

        for v in component.vs:
            value = {}
            value['index'] = v.index
            value['uri'] = v['uri']
            value['global_degree'] = global_degree[v.index]
            value['local_degree'] = local_degree[v.index]
            value['global_eigenvector'] = global_eigenvector[v.index]
            value['local_eigenvector'] = local_eigenvector[v.index]
            value['global_closeness'] = global_closeness[v.index]
            value['local_closeness'] = local_closeness[v.index]
            value['global_betweenness'] = global_betweenness[v.index]
            value['local_betweenness'] = local_betweenness[v.index]
            value['global_clustering'] = global_clustering[v.index]
            value['local_clustering'] = local_clustering[v.index]
            value['node_title'] = value['uri']
            value['global_graph_radius'] = global_radius
            value['global_diameter'] = global_diameter
            value['local_graph_radius'] = local_radius
            value['local_diameter'] = local_diameter
            value['component'] = index
            df = pd.DataFrame([value])
            metrics.append(df)
    dfs = pd.concat(metrics)
    for v in graph.vs:
        row = dfs.loc[dfs.uri == v['uri']]
        v['global_degree'] = row['global_degree'].values[0]
        v['local_degree'] = row['local_degree'].values[0]
        v['global_eigenvector'] = row['global_eigenvector'].values[0]
        v['local_eigenvector'] = row['local_eigenvector'].values[0]
        v['global_closeness'] = row['global_closeness'].values[0]
        v['local_closeness'] = row['local_closeness'].values[0]
        v['global_betweenness'] = row['global_betweenness'].values[0]
        v['local_betweenness'] = row['local_betweenness'].values[0]
        v['global_clustering'] = row['global_clustering'].values[0]
        v['local_clustering'] = row['local_clustering'].values[0]
        v['node_title'] = row['node_title'].values[0]
        v['global_graph_radius'] = row['global_graph_radius'].values[0]
        v['global_diameter'] = row['global_diameter'].values[0]
        v['local_graph_radius'] = row['local_graph_radius'].values[0]
        v['local_diameter'] = row['local_diameter'].values[0]
        v['component'] = row['component'].values[0]
    return graph

def get_katz(biadjacency, is_bipartite):
    print('calculating katz')
    katz = Katz()
    katz.fit_transform(biadjacency)
    if (len(katz.scores_) < 3) | (is_bipartite == False):
        values_row = katz.scores_
        values_col = katz.scores_
    else:
        values_row = katz.scores_row_ 
        values_col = katz.scores_col_ 
    return values_row, values_col

def get_louvain(biadjacency, is_bipartite):
    louvain = Louvain(modularity='dugue')
    louvain.fit_transform(biadjacency, force_bipartite=is_bipartite)
    if (len(louvain.labels_) == 1) | (is_bipartite == False):
        values_row = louvain.labels_
        values_col = louvain.labels_
    else:
        values_row = louvain.labels_row_
        values_col = louvain.labels_col_
    return values_row, values_col 

def get_pagerank(biadjacency, seeds=None):
    pagerank = PageRank()
    pagerank.fit(biadjacency, seeds=seeds)
    values_row = pagerank.scores_row_
    values_col = pagerank.scores_col_
    return values_row, values_col

def generate_sknetwork_metrics(edgelist, nodelist, metrics, is_bipartite, seeds=None):
    tuples = [tuple(x) for x in edgelist[['source', 'target','weight']].values]
    graph = convert_edge_list(tuples, bipartite=is_bipartite, named=True)
    biadjacency = graph.biadjacency if is_bipartite else graph.adjacency 
    
    for metric in metrics:
        values_row, values_col = get_louvain(biadjacency, is_bipartite) if metric == 'louvain' else (get_katz(biadjacency, is_bipartite) if metric == 'katz' else get_pagerank(biadjacency, seeds))
        nodelist[f"{metric}"] = np.nan
        if (len(tuples) > 1) and (is_bipartite):
            names_col = graph.names_col
            names_row = graph.names_row
            for label, node in zip(values_row, names_row):
                nodelist.loc[nodelist.uri == node, f"{metric}"] = label 
            for label, node in zip(values_col, names_col):
                nodelist.loc[nodelist.uri == node, f"{metric}"] = label
        else:
            names = graph.names
            node_col = 'uri' if is_bipartite else 'node_id'
            for label, node in zip(values_row, names):
                nodelist.loc[nodelist[node_col] == node, f"{metric}"] = label
    return nodelist

def generate_link_metrics(graph, edgelist, nodelist, metrics, is_bipartite):
    if is_bipartite:
        bn = BipartiteNetwork()
        bn.set_edgelist(
            edgelist,
            top_col='source', 
            bottom_col='target',
            weight_col='weight'
        )
        nodes_df = nodelist.copy()
        for m in metrics:
            source_birank_df, target_birank_df = bn.generate_birank(normalizer=f'{m}')
            source_birank_df = source_birank_df.rename(columns={'source_birank': m, 'source': 'uri'})
            target_birank_df = target_birank_df.rename(columns={'target_birank': m, 'target': 'uri'})
            rankings = pd.concat([source_birank_df, target_birank_df])
            nodes_df = pd.merge(nodes_df, rankings, on='uri', how='inner')
    else:
        pagerank = graph.pagerank()
        hubs = graph.hub_score()
        auth = graph.authority_score()

        for v in graph.vs:
            v['pagerank'] = pagerank[v.index]
            v['hubs'] = hubs[v.index]
            v['auth'] = auth[v.index]

        nd, _ = generate_dataframes(graph, False, False)
        nodes_df = pd.merge(nd[['uri', 'pagerank', 'hubs', 'auth']], nodelist, on='uri', how='inner')
    return nodes_df

def generate_local_metrics(graph, original_nodelist, sk_metrics, link_metrics, is_bipartite, is_networkx):
    if is_networkx:
        components = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    else:
        components = graph.decompose()
    nodes_df = []
    combined_metrics = sk_metrics + link_metrics
    for idx, g in enumerate(components, start=1):
        if is_bipartite:
            top_nodes = {n for n, d in g.nodes(data=True) if d["bipartite"] == 0}
            bottom_nodes = set(g) - top_nodes
            print(f'component {idx} - size {len(g)} - graph density ', bipartite.density(g, bottom_nodes))
        else:
            print(f'component {idx} - size {len(g.vs)} - graph density ', g.density())
        nodelist, edgelist = generate_dataframes(g, is_bipartite, is_networkx)
        if len(edgelist) > 0:
            updated_nodelist = generate_sknetwork_metrics(edgelist, nodelist, sk_metrics, is_bipartite)
            ranked_nodelist = generate_link_metrics(g, edgelist, updated_nodelist, link_metrics, is_bipartite)
        else:
            ranked_nodelist = nodelist.copy()
            for metric in combined_metrics:
                print(f'{metric} not calculated for component {idx}')
                ranked_nodelist[f'{metric}'] = None
        
        for metric in combined_metrics:
            ranked_nodelist = ranked_nodelist.rename(columns={metric : f'local_{metric}'})
            if is_bipartite:
                for d, v in graph.nodes(data=True):
                    node = ranked_nodelist.loc[ranked_nodelist.uri == v['uri']]
                    v['local_' + metric] = node['local_' + metric]
            else:
                for v in graph.vs:
                    node = ranked_nodelist.loc[ranked_nodelist.uri == v['uri']]
                    v['local_' + metric] = node['local_' + metric]
        nodes_df.append(ranked_nodelist)
    final_local_nodes = pd.concat(nodes_df)
    return final_local_nodes

def get_bipartite_link_prediction(edgelist, nodelist, pred_edge, is_bipartite=True):
    """
    From the sknetwork docs:
        If int i, return the similarities s(i, j) for all j.

        If list or array integers, return s(i, j) for i in query, for all j as array.

        If tuple (i, j), return the similarity s(i, j).

        If list of tuples or array of shape (n_queries, 2), return s(i, j) for (i, j) in query as array.
    """
    tuples = [tuple(x) for x in edgelist.values]
    graph = convert_edge_list(tuples, bipartite=is_bipartite)
    biadjacency = graph.biadjacency
    names = graph.names
    ji = JaccardIndex()
    ji.fit(biadjacency)
    ji_scores = ji.predict(list(pred_edge.values()))
    col_name = '_'.join(list(pred_edge.keys()))
    for name, score in zip(names, ji_scores):
        nodelist.loc[nodelist.uri == name, f"link_{col_name}"] = score 
    return nodelist

def get_node_redundancy(graph, nodelist):
    G = graph.copy()
    while any(len(G[node]) < 3 for node in G):
        remove = [node for node in G if len(G[node]) < 3]
        G.remove_nodes_from(remove)
    
    node_red = nx.bipartite.node_redundancy(G)
    bipartite_red = pd.DataFrame(node_red.items(), columns=['uri', 'redundancy'])
    nodelist = pd.merge(nodelist, bipartite_red, on='uri', how='left')
    return nodelist
    
def update_networkx_nodes(graph, nodelist):
    cols = nodelist.columns
    cols = [c for c in cols if ('global' in c) | ('local' in c)]
    for node,attr in graph.nodes(data=True):
        selected_node = nodelist[(nodelist.uri == node)]
        for col in cols:
            attr[col] = selected_node[col].values[0]


def update_igraph_nodes(graph, nodelist):
    cols = nodelist.columns
    cols = [c for c in cols if ('global' in c) | ('local' in c)]
    for node in graph.vs:
        selected_node = nodelist[(nodelist.uri == node['uri'])]
        for col in cols:
            node[col] = selected_node[col].values[0]

    

