
from igraph import *
import os
from tqdm.notebook import trange, tqdm
tqdm.pandas()

import warnings
warnings.filterwarnings("ignore")

import itertools

import networkx as nx
from networkx.algorithms import bipartite
import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)

import sys
sys.path.append("..")
from network_analysis.generate_network_metrics import *
from network_analysis.create_networks import *
from network_analysis.read_write_networks import * 


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

    borrow_events.groupby(f'{group_col}')[f'{list_col}'].apply(list).reset_index(name=f'list_{list_col}').progress_apply(create_edges, axis=1)
    final_edges = pd.concat(edges)
    grouped_edges = final_edges.groupby(
        ['source', 'target', f'{group_col}']).size().reset_index(name='counts')
    return grouped_edges

def get_attrs(dict_attrs, rows):
    updated_dict_attrs = dict_attrs.copy()
    for k, v in dict_attrs.items():
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
    nodelist = df.loc[:, [node_col]]
    edgelist = build_edges(df, edge_col, node_col)
    nodelist.apply(add_nodes, graph=graph, node_attrs=node_attrs, axis=1)
    edgelist.apply(add_edges, graph=graph, edge_attrs=edge_attrs, axis=1)

def create_bipartite_network(rows, graph, member_attrs, book_attrs, edge_attrs):
    
    updated_member_attrs = get_attrs(member_attrs, rows)
    updated_book_attrs = get_attrs(book_attrs, rows)
    updated_edge_attrs = get_attrs(edge_attrs, rows)

    tuples = [(rows.member_id, rows.item_uri)]
    graph.add_node(rows.member_id, **updated_member_attrs,
                   group='members', bipartite=0)
    graph.add_node(rows.item_uri, group='books',
                   bipartite=1, **updated_book_attrs)
    graph.add_edges_from(tuples, **updated_edge_attrs)

def get_unipartite_graph(borrow_events, grouped_events_df, member_attrs, book_attrs, original_node_attrs, original_edge_attrs, is_projected):
    if is_projected:
        bipartite_graph = get_bipartite_graph(
            grouped_events_df, member_attrs, book_attrs, original_edge_attrs)
        member_nodes = [
            n for n in bipartite_graph.nodes if bipartite_graph.nodes[n]['group'] == 'members']
        book_nodes = [
            n for n in bipartite_graph.nodes if bipartite_graph.nodes[n]['group'] == 'books']
        members_graph = bipartite.weighted_projected_graph(
            bipartite_graph, member_nodes)
        books_graph = bipartite.weighted_projected_graph(bipartite_graph, book_nodes)
    else:
        
        node_col = 'member_id'
        node_attrs = {'uri': node_col} | original_node_attrs
        edge_col = 'item_uri'
        edge_attrs = {'uri': edge_col, 'weight': 'counts'} | original_edge_attrs
        members_graph = nx.Graph()
        create_unipartite_network(borrow_events, members_graph,node_attrs, edge_attrs, node_col, edge_col)
        
        print('finished creating members graph')
        node_col = 'item_uri'
        node_attrs = {'uri': node_col} | original_node_attrs
        edge_col = 'member_id'
        edge_attrs = {'uri': edge_col,'weight': 'counts'} | original_edge_attrs
        books_graph = nx.Graph()
        create_unipartite_network(borrow_events, books_graph,node_attrs, edge_attrs, node_col, edge_col)
        print('finished creating books graph')
                            
    final_books_graph = Graph.from_networkx(books_graph)
    final_members_graph = Graph.from_networkx(members_graph)
    return final_members_graph, final_books_graph


def get_bipartite_graph(df, member_attrs, book_attrs, edge_attrs):
    bipartite_graph = nx.Graph()

    df.apply(create_bipartite_network, graph=bipartite_graph,
             member_attrs=member_attrs, book_attrs=book_attrs, edge_attrs=edge_attrs, axis=1)
    print('connected?', nx.is_connected(bipartite_graph))
    print('bipartite?', nx.is_bipartite(bipartite_graph))
    components = [len(c) for c in sorted(
        nx.connected_components(bipartite_graph), key=len, reverse=True)]
    print(components)
    return bipartite_graph


def build_bipartite_graphs(grouped_events_df, member_attrs, book_attrs, edge_attrs, should_process, write_to_file, file_name, sk_metrics, link_metrics, members_df, books_df):
    """Build Bipartite Graphs"""

    bipartite_graph = get_bipartite_graph(
        grouped_events_df, member_attrs, book_attrs, edge_attrs)

    top_nodes = {n for n, d in bipartite_graph.nodes(
        data=True) if d["bipartite"] == 0}

    bottom_nodes = set(bipartite_graph) - top_nodes
    print('graph density: ', bipartite.density(bipartite_graph, bottom_nodes))
    if should_process:
        processed_bipartite_graph = get_bipartite_network_metrics(
            bipartite_graph)
        if write_to_file:
            nx.write_gexf(processed_bipartite_graph, f'{file_name}_graph.gexf')
        processed_bipartite_graph = bipartite_graph
        processed_bipartite_nodelist, processed_bipartite_edgelist = generate_dataframes(
            processed_bipartite_graph, True, True)
        print(
            f"calculating local skmetrics: {' '.join(sk_metrics + link_metrics)}")
        local_nodelist = generate_local_metrics(
            processed_bipartite_graph, processed_bipartite_nodelist, sk_metrics, link_metrics, True, True)
        print(f"calculating global skmetrics: {' '.join(sk_metrics)}")
        updated_bipartite_nodelist = generate_sknetwork_metrics(
            processed_bipartite_edgelist, local_nodelist, sk_metrics, True)

        print(f"calculating global link metrics: : {' '.join(link_metrics)}")
        bipartite_nodelist = generate_link_metrics(
            processed_bipartite_graph, processed_bipartite_edgelist, updated_bipartite_nodelist, link_metrics, True)
        all_metrics = sk_metrics + link_metrics
        for m in all_metrics:
            bipartite_nodelist = bipartite_nodelist.rename(
                columns={m: f'global_{m}'})
        bipartite_edgelist = processed_bipartite_edgelist
        bipartite_graph = processed_bipartite_graph
    else:
        bipartite_nodelist, bipartite_edgelist = generate_dataframes(
            bipartite_graph, True, True)
    

    update_networkx_nodes(bipartite_graph, bipartite_nodelist)

    if write_to_file:
       write_dataframe(file_name, bipartite_edgelist, bipartite_nodelist)
       nx.write_gexf(bipartite_graph, f'{file_name}_graph.gexf')
    
    
    bipartite_members = bipartite_nodelist[bipartite_nodelist.group == 'members']

    bipartite_books = bipartite_nodelist[bipartite_nodelist.group == 'books']

    joined_members = combine_dataframes(
        bipartite_members, members_df, bipartite_members.columns.tolist(), 'uri', 'inner')
    joined_books = combine_dataframes(
        bipartite_books, books_df, bipartite_books.columns.tolist(), 'uri', 'inner')

    return (bipartite_graph, bipartite_nodelist, bipartite_edgelist, joined_members, joined_books)

## Reload Saved Graphs
def reload_saved_graphs(file_path, members_df, books_df):
    bipartite_graph = nx.read_gexf(f'{file_path}_graph.gexf')
    bipartite_nodelist = pd.read_csv(f'{file_path}_nodelist.csv')
    bipartite_edgelist = pd.read_csv(f'{file_path}_edgelist.csv')

    bipartite_members = bipartite_nodelist[bipartite_nodelist.group == 'members']

    bipartite_books = bipartite_nodelist[bipartite_nodelist.group == 'books']

    joined_members = combine_dataframes(
        bipartite_members, members_df, bipartite_members.columns.tolist(), 'uri', 'inner')
    joined_books = combine_dataframes(
        bipartite_books, books_df, bipartite_books.columns.tolist(), 'uri', 'inner')

    return (bipartite_graph, bipartite_nodelist, bipartite_edgelist, joined_members, joined_books)


def check_reload_build_bipartite_graphs(grouped_events_df, member_attrs, book_attrs, edge_attrs, should_process, write_to_file, file_name, sk_metrics, link_metrics, members_df, books_df):
    if os.path.exists(f'{file_name}_graph.gexf'):
        print(f"reloading saved graph: {file_name}")
        bipartite_graph, bipartite_nodelist, bipartite_edgelist, joined_members, joined_books = reload_saved_graphs(
            file_name, members_df, books_df)
    else:
        print(f"building graph: {file_name}")
        bipartite_graph, bipartite_nodelist, bipartite_edgelist, joined_members, joined_books = build_bipartite_graphs(
            grouped_events_df, member_attrs, book_attrs, edge_attrs, should_process, write_to_file, file_name, sk_metrics, link_metrics, members_df, books_df)

    return (bipartite_graph, bipartite_nodelist, bipartite_edgelist, joined_members, joined_books)


def build_unipartite_graphs(grouped_events_df, borrow_events, member_attrs, book_attrs, edge_attrs, node_attrs, should_process, write_to_file, file_name, sk_metrics, link_metrics, members_df, books_df, is_projected):

    members_graph, books_graph = get_unipartite_graph(
        borrow_events, grouped_events_df, member_attrs, book_attrs, node_attrs, edge_attrs, is_projected)

    if should_process:
        processed_members_graph = get_unipartite_network_metrics(members_graph)
        processed_members_graph.write_graphml(
            f'{file_name}_members_graph.graphml')
        processed_books_graph = get_unipartite_network_metrics(books_graph)
        processed_books_graph.write_graphml(f'{file_name}_books_graph.graphml')
        processed_members_nodelist, processed_members_edgelist = generate_dataframes(
            processed_members_graph, False, False)
        processed_books_nodelist, processed_books_edgelist = generate_dataframes(
            processed_books_graph, False, False)

        print(
            f"calculating local skmetrics members: {' '.join(sk_metrics + link_metrics)}")
        local_members_nodelist = generate_local_metrics(
            processed_members_graph, processed_members_nodelist, sk_metrics, link_metrics, False, False)
        print(
            f"calculating local skmetrics books: {' '.join(sk_metrics + link_metrics)}")
        local_books_nodelist = generate_local_metrics(
            processed_books_graph, processed_books_nodelist, sk_metrics, link_metrics, False, False)

        print(f"calculating global members skmetrics: {' '.join(sk_metrics)}")
        updated_members_nodelist = generate_sknetwork_metrics(
            processed_members_edgelist, local_members_nodelist, sk_metrics, False)
        print(f"calculating global books skmetrics: {' '.join(sk_metrics)}")
        updated_books_nodelist = generate_sknetwork_metrics(
            processed_books_edgelist, local_books_nodelist, sk_metrics, False)

        print(
            f"calculating global link members metrics: : {' '.join(link_metrics)}")
        members_nodelist = generate_link_metrics(
            processed_members_graph, processed_members_edgelist, updated_members_nodelist, link_metrics, False)
        print(f"calculating global link books metrics: : {' '.join(link_metrics)}")
        books_nodelist = generate_link_metrics(
            processed_books_graph, processed_books_edgelist, updated_books_nodelist, link_metrics, False)

        all_metrics = sk_metrics + link_metrics
        for m in all_metrics:
            members_nodelist = members_nodelist.rename(columns={m: f'global_{m}'})
            books_nodelist = books_nodelist.rename(columns={m: f'global_{m}'})
        members_edgelist = processed_members_edgelist
        books_edgelist = processed_books_edgelist
    else:
        processed_members_graph = members_graph
        processed_books_graph = books_graph
        members_nodelist, members_edgelist = generate_dataframes(
            processed_members_graph, False, False)
        books_nodelist, books_edgelist = generate_dataframes(
            processed_books_graph, False, False)
    
    update_igraph_nodes(processed_members_graph, members_nodelist)
    update_igraph_nodes(processed_books_graph, books_nodelist)

    if write_to_file:
        write_dataframe(file_name + '_members', members_edgelist, members_nodelist)
        members_graph.write_graphml(f'{file_name}_members_graph.graphml')
        members_gml = nx.read_graphml(f'{file_name}_members_graph.graphml')
        nx.write_gexf(members_gml, f'{file_name}_members_graph.gexf')
        if os.path.exists(f'{file_name}_members_graph.gexf'):
            os.remove(f'{file_name}_members_graph.graphml')

        write_dataframe(file_name + '_books', books_edgelist, books_nodelist)
        books_graph.write_graphml(f'{file_name}_books_graph.graphml')
        books_gml = nx.read_graphml(f'{file_name}_books_graph.graphml')
        nx.write_gexf(books_gml, f'{file_name}_books_graph.gexf')
        if os.path.exists(f'{file_name}_books_graph.gexf'):
            os.remove(f'{file_name}_books_graph.graphml')

    joined_members = combine_dataframes(
        members_nodelist, members_df, members_nodelist.columns.tolist(), 'uri', 'inner')
    joined_books = combine_dataframes(
        books_nodelist, books_df, books_nodelist.columns.tolist(), 'uri', 'inner')

    return (processed_members_graph, members_nodelist, members_edgelist, joined_members, processed_books_graph, books_nodelist, books_edgelist, joined_books)


def reload_saved_unipartite_graphs(file_path, members_df, books_df):
    members_graph = nx.read_gexf(f'{file_path}_members_graph.gexf')
    members_nodelist = pd.read_csv(f'{file_path}_members_nodelist.csv')
    members_edgelist = pd.read_csv(f'{file_path}_members_edgelist.csv')

    books_graph = nx.read_gexf(f'{file_path}_books_graph.gexf')
    books_nodelist = pd.read_csv(f'{file_path}_books_nodelist.csv')
    books_edgelist = pd.read_csv(f'{file_path}_books_edgelist.csv')

    joined_members = combine_dataframes(
        members_nodelist, members_df, members_nodelist.columns.tolist(), 'uri', 'inner')
    joined_books = combine_dataframes(
        books_nodelist, books_df, books_nodelist.columns.tolist(), 'uri', 'inner')

    return (members_graph, members_nodelist, members_edgelist, joined_members, books_graph, books_nodelist, books_edgelist,  joined_books)


def check_reload_build_unipartite_graphs(grouped_events_df, borrow_events, member_attrs, book_attrs, edge_attrs, node_attrs, should_process, write_to_file, file_name, sk_metrics, link_metrics, members_df, books_df, is_projected):
    if os.path.exists(f'{file_name}_members_graph.gexf'):
        print(f"reloading saved graph: {file_name}")
        members_graph, members_nodelist, members_edgelist, joined_members, books_graph, books_nodelist, books_edgelist, joined_books = reload_saved_unipartite_graphs(
            file_name, members_df, books_df)
    else:
        print(f"building graph: {file_name}")
        members_graph, members_nodelist, members_edgelist, joined_members, books_graph, books_nodelist, books_edgelist, joined_books = build_unipartite_graphs(grouped_events_df, borrow_events, member_attrs, book_attrs, edge_attrs, node_attrs, should_process, write_to_file, file_name, sk_metrics, link_metrics, members_df, books_df, is_projected)

    return (members_graph, members_nodelist, members_edgelist, joined_members, books_graph, books_nodelist, books_edgelist,  joined_books)
