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

import sys
sys.path.append("..")
from network_analysis.birankpy import BipartiteNetwork


def build_nodes(borrow_events, node_col):
    return borrow_events[[node_col]]


def build_edges(borrow_events, group_col, list_col):
    edges = []
    def create_edges(rows):
        if len(rows[f'list_{list_col}']) > 1:
            combos = list(itertools.combinations(rows[f'list_{list_col}'], 2))

            for c in combos:
                edge = {}
                edge['source'] = c[0]
                edge['target'] = c[1]
                edge[f'{group_col}'] = rows[group_col]
                edges.append(pd.DataFrame([edge]))

    borrow_events.groupby(f'{group_col}')[f'{list_col}'].apply(
        list).reset_index(name=f'list_{list_col}').progress_apply(create_edges, axis=1)
    final_edges = pd.concat(edges)
    grouped_edges = final_edges.groupby(
        ['source', 'target', f'{group_col}']).size().reset_index(name='counts')
    return grouped_edges

def get_attrs(dict_attrs, rows):
    updated_dict_attrs = dict_attrs.copy()
    for k,v in dict_attrs.items():
        updated_dict_attrs[k] = rows[v]
    return updated_dict_attrs

def add_nodes(rows, graph, node_attrs):
    updated_node_attrs = get_attrs(node_attrs, rows) if len(
        node_attrs) > 1 else node_attrs
    graph.add_nodes_from(rows, **updated_node_attrs)

def add_edges(rows, graph, edge_attrs):
    updated_edge_attrs = get_attrs(edge_attrs, rows)
    graph.add_edges_from([(rows.source, rows.target)], **updated_edge_attrs)

def create_unipartite_network(df, graph, node_attrs, edge_attrs, node_col, edge_col):
    '''Create a unipartite graph either members or books'''
    nodelist = build_nodes(df, node_col)
    edgelist = build_edges(df, edge_col, node_col)
    nodelist.apply(add_nodes, graph=graph, node_attrs=node_attrs)
    edgelist.apply(add_edges, graph=graph, edge_attrs=edge_attrs, axis=1)


def create_books_network(rows, graph, node_attrs, edge_attrs):
    '''Create a graph from all books using members who read the same book as an edges'''
    combos = list(itertools.combinations(rows.list_books, 2))

    updated_node_attrs = get_attrs(node_attrs, rows)
    updated_edge_attrs = get_attrs(edge_attrs, rows) if len(combos) == 0 else None
    if len(combos) > 0:
        graph.add_nodes_from(rows.list_books, **updated_node_attrs)
        graph.add_edges_from(combos, **updated_edge_attrs)
    if len(combos) == 0:
        graph.add_nodes_from(rows.list_books, **updated_node_attrs)

def create_bipartite_network(rows, graph, member_attrs, book_attrs, edge_attrs):
    updated_member_attrs = get_attrs(member_attrs, rows)
    updated_book_attrs = get_attrs(book_attrs, rows)
    updated_edge_attrs = get_attrs(edge_attrs, rows)

    tuples = [(rows.member_id, rows.item_uri)]
    graph.add_node(rows.member_id,**updated_member_attrs, group='members', bipartite=0)
    graph.add_node(rows.item_uri, group='books', bipartite=1, **updated_book_attrs)
    graph.add_edges_from(tuples, **updated_edge_attrs)

def get_bipartite_graph(df, member_attrs, book_attrs, edge_attrs):
    bipartite_graph = nx.Graph()
    
    df.apply(create_bipartite_network,graph=bipartite_graph, member_attrs=member_attrs, book_attrs=book_attrs, edge_attrs=edge_attrs, axis=1)
    print('connected?', nx.is_connected(bipartite_graph))
    print('bipartite?', nx.is_bipartite(bipartite_graph))
    components = [len(c) for c in sorted(
        nx.connected_components(bipartite_graph), key=len, reverse=True)]
    print(components)
    return bipartite_graph

def get_network_metrics(graph, is_bipartite=False):
    '''Run network metrics for each node at various snapshots
    
    Some on going questions:
    - Should we remove self loops?
    - Are there more accurate metrics?
    - Should we try a chain model to more accurately capture order of reading?
    - What are the behaviors we want to include with book borrowing? Obviously there's waiting for a book to be available but what about recommendations? Or genre?
    '''
    if nx.is_empty(graph):
        return graph
    elif is_bipartite:
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

    else:
        degree = nx.degree_centrality(graph)
        closeness = nx.closeness_centrality(graph)
        global_betweenness = nx.betweenness_centrality(graph)
        harmonic = nx.harmonic_centrality(graph)
        graph_components = sorted(
            nx.connected_components(graph), key=len, reverse=True)
        clustering = nx.clustering(graph)

        for index, component in enumerate(tqdm(graph_components)):
            subgraph = graph.subgraph(component)
            diameter = nx.diameter(subgraph) if subgraph.size() > 0 else 0
            radius = nx.radius(subgraph) if subgraph.size() > 0 else 0
            eigenvector = nx.eigenvector_centrality_numpy(
                subgraph, max_iter=600) if subgraph.size() > 1 else {k: 0 for k in subgraph.nodes}
            local_betweenness = nx.betweenness_centrality(subgraph)

            for d, v in subgraph.nodes(data=True):
                v['degree'] = degree[d]
                v['eigenvector'] = eigenvector[d]
                v['closeness'] = closeness[d]
                v['global_betweenness'] = global_betweenness[d]
                v['local_betweenness'] = local_betweenness[d]
                v['harmonic'] = harmonic[d]
                v['clustering'] = clustering[d]
                v['node_title'] = d
                v['graph_radius'] = radius
                v['diameter'] = diameter
                v['component'] = index
    return graph

def generate_edgelist(G, delimiter=" ", data=True):
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
    for _,_,e in bipartite_graph.edges(data=True):
        keys = [*e]
        if len(keys) > 0:
            break
    return keys

def get_bipartite_edgelist(bipartite_graph):
    """Turn graph edgelist into pandas dataframe but keep bipartite groupings"""
    edges = []
    edge_cols = get_edge_cols(bipartite_graph)
    for line in generate_edgelist(bipartite_graph, " ", data=edge_cols):
        data = line.split(' ')

        edges.append({'source': data[0], 'target': data[1], 'weight': data[2]})
    return pd.DataFrame(edges)

def generate_dataframes(graph, is_bipartite):
    """Generate dataframes from graph and write to file"""
    nodes_df = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient='index')
    nodes_df = nodes_df.reset_index(drop=True)
    if is_bipartite:
        edges_df = get_bipartite_edgelist(graph)
    else:  
        edges_df = nx.to_pandas_edgelist(graph)
    return (nodes_df, edges_df)

def write_dataframe(file_name, edges_df, nodes_df):
    """Write dataframes"""
    edges_df.to_csv(f'{file_name}_edgelist.csv', index=False)
    nodes_df.to_csv(f'{file_name}_nodelist.csv', index=False)
    return (nodes_df, edges_df)

def combine_dataframes(graph_df, sco_df, columns, on_column, how_setting):
    joined_df = pd.merge(left=sco_df, right=graph_df[columns], on=on_column, how=how_setting)
    return joined_df

def get_katz(biadjacency, is_bipartite):
    katz = Katz()
    katz.fit_transform(biadjacency)
    if (len(katz.scores_) == 1) | (is_bipartite == False):
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

def generate_sknetwork_metrics(edgelist, nodelist, metrics, is_bipartite=True, seeds=None):
    tuples = [tuple(x) for x in edgelist.values]
    graph = convert_edge_list(tuples, bipartite=is_bipartite)

    biadjacency = graph.biadjacency if is_bipartite else graph.adjacency 
    names_col = graph.names_col
    names_row = graph.names_row
    names = graph.names
    for metric in metrics:
        values_row, values_col = get_louvain(biadjacency, is_bipartite) if metric == 'louvain' else (get_katz(biadjacency, is_bipartite) if metric == 'katz' else get_pagerank(biadjacency, seeds))
        nodelist[f"{metric}"] = np.nan
        if (len(tuples) > 1) and (is_bipartite):
            for label, node in zip(values_row, names_row):
                nodelist.loc[nodelist.uri == node, f"{metric}"] = label 
            for label, node in zip(values_col, names_col):
                nodelist.loc[nodelist.uri == node, f"{metric}"] = label
        else:
            for label, node in zip(values_row, names):
                nodelist.loc[nodelist.uri == node, f"{metric}"] = label 
            
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
        pagerank = nx.pagerank(graph)
        hubs, auth = nx.hits(graph)

        for d, v in graph.nodes(data=True):
            v['pagerank'] = pagerank[d]
            v['hubs'] = hubs[d]
            v['auth'] = auth[d]

        nodes_df, _ = generate_dataframes(graph, False)
    
    return nodes_df

def generate_local_metrics(graph, original_nodelist, sk_metrics, link_metrics, is_bipartite):
    components = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    nodes_df = []
    for idx, g in enumerate(components, start=1):
        if is_bipartite:
            top_nodes = {n for n, d in g.nodes(data=True) if d["bipartite"] == 0}
            bottom_nodes = set(g) - top_nodes
            print(f'component {idx} - size {len(g)} - graph density ', bipartite.density(g, bottom_nodes))
        else:
            print(f'component {idx} - size {len(g)} - graph density ', nx.density(g))
        nodelist, edgelist = generate_dataframes(g, is_bipartite)
        updated_nodelist = generate_sknetwork_metrics(edgelist, nodelist, sk_metrics, is_bipartite)

        ranked_nodelist = generate_link_metrics(g, edgelist, updated_nodelist, link_metrics, is_bipartite)
        combined_metrics = sk_metrics + link_metrics
        for m in combined_metrics:
            ranked_nodelist.rename(columns={m : f'local_{m}'}, inplace=True)
        nodes_df.append(ranked_nodelist)
    final_local_nodes = pd.concat(nodes_df)
    return pd.merge(original_nodelist, final_local_nodes, on=original_nodelist.columns.tolist(), how='inner')

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
    
