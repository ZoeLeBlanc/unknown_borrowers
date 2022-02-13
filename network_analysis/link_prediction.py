from bigraph.predict import pa_predict, jc_predict, cn_predict, aa_predict, katz_predict
from network_analysis.load_datasets import get_updated_shxco_data
from network_analysis.create_networks import *
from network_analysis.birankpy import BipartiteNetwork
from bigraph.evaluation import evaluation
import sys
from IPython.display import display, Markdown, HTML
import warnings
from tqdm.notebook import trange, tqdm
import os
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import networkx as nx
import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)

tqdm.pandas()
warnings.filterwarnings("ignore")
sys.path.append("..")

bipartite_metrics = ['jc_prediction',
                     'pa_prediction', 'cn_prediction', 'aa_prediction']


def get_bipartite_link_predictions(graph, output_path):
    if os.path.exists(output_path):
        all_preds = pd.read_csv(output_path)
    else:
        print('Running jaccard link prediction')
        jc_preds = jc_predict(graph)
        jc_preds_df = pd.DataFrame(
            data=list(jc_preds.values()), index=jc_preds.keys()).reset_index()
        jc_preds_df.columns = ['member_id', 'item_uri', 'jc_prediction']
        print('Running preferential attachment link prediction')
        pa_preds = pa_predict(graph)
        pa_preds_df = pd.DataFrame(
            data=list(pa_preds.values()), index=pa_preds.keys()).reset_index()
        pa_preds_df.columns = ['member_id', 'item_uri', 'pa_prediction']
        print('Running common neighbors link prediction')
        cn_preds = cn_predict(graph)
        cn_preds_df = pd.DataFrame(
            data=list(cn_preds.values()), index=cn_preds.keys()).reset_index()
        cn_preds_df.columns = ['member_id', 'item_uri', 'cn_prediction']
        print('Running adamic adar link prediction')
        aa_preds = aa_predict(graph)
        aa_preds_df = pd.DataFrame(
            data=list(aa_preds.values()), index=aa_preds.keys()).reset_index()
        aa_preds_df.columns = ['member_id', 'item_uri', 'aa_prediction']

        # print('Running katz link prediction')
        # katz_preds = katz_predict(graph)
        # katz_preds_df = pd.DataFrame(
        #     data=list(katz_preds.values()), index=katz_preds.keys()).reset_index()
        # katz_preds_df.columns = ['member_id', 'item_uri', 'katz_prediction']

        all_preds = pd.merge(jc_preds_df, pa_preds_df, on=[
                            'member_id', 'item_uri'], how='outer')
        all_preds = pd.merge(all_preds, cn_preds_df, on=[
                            'member_id', 'item_uri'], how='outer')
        all_preds = pd.merge(all_preds, aa_preds_df, on=[
                            'member_id', 'item_uri'], how='outer')
        # all_preds = pd.merge(all_preds, katz_preds_df, on=['member_id', 'item_uri'], how='outer')
        all_preds.to_csv(output_path, index=False)
    return all_preds


def get_predictions_by_metric(row, metric, predictions_df, circulation_books, limit_to_circulation=True):
    if limit_to_circulation:
        subset_predictions = predictions_df[(predictions_df.member_id == row.member_id) & (
            predictions_df.item_uri.isin(circulation_books))].sort_values(by=f'{metric}', ascending=False)
    else:
        subset_predictions = predictions_df[(
            predictions_df.member_id == row.member_id)].sort_values(by=f'{metric}', ascending=False)

    return subset_predictions[['member_id', 'item_uri', f'{metric}']]


def get_specific_predictions(row, number_of_results, limit_to_circulation, events_df, borrow_events, members_df, books_df, relative_date, predict_group, output_path):

    print(
        f'Processing {row.member_id} with subscription {row.subscription_start}')
    grouped_col = 'item_uri' if predict_group == 'books' else 'member_id'
    index_col = 'member_id' if predict_group == 'books' else 'item_uri'
    seed_data = events_df[(events_df[index_col] == row[index_col]) & (
        events_df[grouped_col].isna() == False)]

    circulation_events = borrow_events[borrow_events.start_datetime.between(
        relative_date, row.subscription_endtime) | borrow_events.end_datetime.between(relative_date, row.subscription_endtime)]
    circulation_events = circulation_events[circulation_events[index_col]
                                            != row[index_col]]

    circulation_counts = circulation_events.groupby([grouped_col]).size().reset_index(
        name='counts').sort_values(['counts'], ascending=False)[0:number_of_results]
    circulating_items = circulation_counts[grouped_col].unique().tolist()

    # top_counts = len(circulation_events[(circulation_events[index_col] == row[index_col]) & (circulation_events[grouped_col].isna() == False)])

    graph_data = pd.concat([seed_data, circulation_events], axis=0)
    member_attrs = {'uri': 'member_id'}
    book_attrs = {'uri': 'item_uri'}
    edge_attrs = {'weight': 'counts'}
    should_process = True
    write_to_file = False
    sk_metrics = ['katz', 'louvain']
    link_metrics = ['HITS', 'CoHITS', 'BiRank', 'BGRM']
    circulation_events_grouped = graph_data.groupby(
        ['member_id', 'item_uri']).size().reset_index(name='counts')

    circulation_events_bipartite_graph, circulation_events_bipartite_nodelist, circulation_events_bipartite_edgelist, circulation_events_members, circulation_events_books = check_reload_build_bipartite_graphs(
        circulation_events_grouped, member_attrs, book_attrs, edge_attrs, should_process, write_to_file, 'test2', sk_metrics, link_metrics, members_df, books_df)

    remove = circulation_events_bipartite_nodelist[circulation_events_bipartite_nodelist.component != 0].uri.tolist(
    )
    circulation_events_bipartite_graph.remove_nodes_from(remove)

    predictions_df = get_bipartite_link_predictions(
        circulation_events_bipartite_graph)
    identified_top_predictions = {}
    metrics = ['jc_prediction', 'pa_prediction',
               'cn_prediction', 'aa_prediction']
    dfs = []
    for m in metrics:

        preds = get_predictions_by_metric(
            row, m, predictions_df, circulating_items, limit_to_circulation)
        identified_top_predictions[f'{index_col}'] = row[index_col]
        identified_top_predictions[f'predicted_values'] = preds[0:number_of_results][grouped_col].tolist(
        )

        identified_top_predictions[f'score'] = preds[0:number_of_results][m].tolist(
        )
        identified_top_predictions['metric'] = m
        dfs.append(pd.DataFrame.from_dict(
            identified_top_predictions, orient='columns'))

    df_final = pd.concat(dfs)

    if os.path.exists(output_path):
        df_final.to_csv(output_path, mode='a', header=False, index=False)
    else:
        df_final.to_csv(output_path, index=False, header=True)
    return df_final


def get_full_predictions(row, number_of_results, limit_to_circulation, predictions_df, events_df, predict_group, metrics, output_path):

    grouped_col = 'item_uri' if predict_group == 'books' else 'member_id'
    index_col = 'member_id' if predict_group == 'books' else 'item_uri'
    identified_top_predictions = {}

    # circulation_start = borrow_events.sort_values(by=['start_datetime'])[0:1].start_datetime.values[0]

    # circulation_events = borrow_events[borrow_events.start_datetime.between(
    #     circulation_start, row.subscription_endtime) | borrow_events.end_datetime.between(circulation_start, row.subscription_endtime)]
    # circulation_events = circulation_events[circulation_events[index_col]!= row[index_col]]
    circulation_events = events_df[(events_df.start_datetime < row.subscription_end) | (events_df.end_datetime < row.subscription_end)]

    circulation_counts = circulation_events.groupby([grouped_col]).size().reset_index(
        name='counts').sort_values(['counts'], ascending=False)
        # [0:number_of_results]
    # circulating_items = circulation_counts[grouped_col].unique().tolist()
    query_books = circulation_events.item_id.unique().tolist()
    member_book_ids = events_df[(events_df.item_id.notna()) & (events_df.member_id == row.member_id)].item_id.unique()
    query_books = list(set(query_books) - set(member_book_ids))

    dfs = []
    for idx, m in enumerate(metrics):


        preds = get_predictions_by_metric(
            row, m, predictions_df, query_books, limit_to_circulation)
        identified_top_predictions[f'{index_col}'] = row[index_col]
        identified_top_predictions[f'predicted_values'] = preds[grouped_col].tolist(
        )

        identified_top_predictions[f'score'] = preds[m].tolist(
        )
        identified_top_predictions['metric'] = m
        dfs.append(pd.DataFrame.from_dict(
            identified_top_predictions, orient='columns'))
    # df_final = pd.DataFrame.from_dict(
    #     identified_top_predictions, orient='columns')

    df_final = pd.concat(dfs)
    row_df = pd.DataFrame([row.to_dict()])
    df_final = df_final.merge(row_df, on=index_col, how='left')

    if os.path.exists(output_path):
        df_final.to_csv(output_path, mode='a', header=False, index=False)
    else:
        df_final.to_csv(output_path, index=False, header=True)

# Convert sparse matrix to tuple


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj, test_frac=.1, val_frac=.05, prevent_disconnect=True, verbose=False):
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.

    if verbose == True:
        print('preprocessing...')

    # Remove diagonal elements
    adj = adj - \
        sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    g = nx.from_scipy_sparse_matrix(adj)
    orig_num_cc = nx.number_connected_components(g)

    adj_triu = sp.triu(adj)  # upper triangular portion of adj matrix
    # (coords, values, shape), edges only 1 way
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]  # all edges, listed only once (not 2 ways)
    # edges_all = sparse_to_tuple(adj)[0] # ALL edges (includes both ways)
    # controls how large the test set should be
    num_test = int(np.floor(edges.shape[0] * test_frac))
    # controls how alrge the validation set should be
    num_val = int(np.floor(edges.shape[0] * val_frac))

    # Store edges in list of ordered tuples (node1, node2) where node1 < node2
    edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1]))
                   for edge in edges]
    all_edge_tuples = set(edge_tuples)
    train_edges = set(edge_tuples)  # initialize train_edges to have all edges
    test_edges = set()
    val_edges = set()

    if verbose == True:
        print('generating test/val sets...')

    # Iterate over shuffled edges, add to train/val sets
    np.random.shuffle(edge_tuples)
    for edge in edge_tuples:
        # print edge
        node1 = edge[0]
        node2 = edge[1]

        # If removing edge would disconnect a connected component, backtrack and move on
        g.remove_edge(node1, node2)
        if prevent_disconnect == True:
            if nx.number_connected_components(g) > orig_num_cc:
                g.add_edge(node1, node2)
                continue

        # Fill test_edges first
        if len(test_edges) < num_test:
            test_edges.add(edge)
            train_edges.remove(edge)

        # Then, fill val_edges
        elif len(val_edges) < num_val:
            val_edges.add(edge)
            train_edges.remove(edge)

        # Both edge lists full --> break loop
        elif len(test_edges) == num_test and len(val_edges) == num_val:
            break

    if (len(val_edges) < num_val or len(test_edges) < num_test):
        print("WARNING: not enough removable edges to perform full train-test split!")
        print("Num. (test, val) edges requested: (", num_test, ", ", num_val, ")")
        print("Num. (test, val) edges returned: (", len(
            test_edges), ", ", len(val_edges), ")")

    if prevent_disconnect == True:
        assert nx.number_connected_components(g) == orig_num_cc

    if verbose == True:
        print('creating false test edges...')

    test_edges_false = set()
    while len(test_edges_false) < num_test:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge not an actual edge, and not a repeat
        if false_edge in all_edge_tuples:
            continue
        if false_edge in test_edges_false:
            continue

        test_edges_false.add(false_edge)

    if verbose == True:
        print('creating false val edges...')

    val_edges_false = set()
    while len(val_edges_false) < num_val:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
                false_edge in test_edges_false or \
                false_edge in val_edges_false:
            continue

        val_edges_false.add(false_edge)

    if verbose == True:
        print('creating false train edges...')

    train_edges_false = set()
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false,
        # not in val_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
                false_edge in test_edges_false or \
                false_edge in val_edges_false or \
                false_edge in train_edges_false:
            continue

        train_edges_false.add(false_edge)

    if verbose == True:
        print('final checks for disjointness...')

    # assert: false_edges are actually false (not in all_edge_tuples)
    assert test_edges_false.isdisjoint(all_edge_tuples)
    assert val_edges_false.isdisjoint(all_edge_tuples)
    assert train_edges_false.isdisjoint(all_edge_tuples)

    # assert: test, val, train false edges disjoint
    assert test_edges_false.isdisjoint(val_edges_false)
    assert test_edges_false.isdisjoint(train_edges_false)
    assert val_edges_false.isdisjoint(train_edges_false)

    # assert: test, val, train positive edges disjoint
    assert val_edges.isdisjoint(train_edges)
    assert test_edges.isdisjoint(train_edges)
    assert val_edges.isdisjoint(test_edges)

    if verbose == True:
        print('creating adj_train...')

    # Re-build adj matrix using remaining graph
    adj_train = nx.adjacency_matrix(g)

    # Convert edge-lists to numpy arrays
    train_edges = np.array([list(edge_tuple) for edge_tuple in train_edges])
    train_edges_false = np.array([list(edge_tuple)
                                 for edge_tuple in train_edges_false])
    val_edges = np.array([list(edge_tuple) for edge_tuple in val_edges])
    val_edges_false = np.array([list(edge_tuple)
                               for edge_tuple in val_edges_false])
    test_edges = np.array([list(edge_tuple) for edge_tuple in test_edges])
    test_edges_false = np.array([list(edge_tuple)
                                for edge_tuple in test_edges_false])

    if verbose == True:
        print('Done with train-test split!')
        print('')

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false, \
        val_edges, val_edges_false, test_edges, test_edges_false


def run_link_prediction():
    members_df, books_df, borrow_events, events_df = get_updated_shxco_data(
        get_subscription=False)
    partial_df = pd.read_csv('../dataset_generator/data/partial_borrowers_collapsed.csv')
    partial_df['index_col'] = partial_df.index
    partial_members = ['raphael-france', 'hemingway-ernest',
                       'colens-fernand', 'kittredge-eleanor-hayden']

    # parse subscription dates so we can use them to identify circulating books
    partial_df['subscription_starttime'] = pd.to_datetime(
        partial_df['subscription_start'], errors='coerce')
    partial_df['subscription_endtime'] = pd.to_datetime(
        partial_df['subscription_end'], errors='coerce')

    # all_events = events_df[events_df.item_uri.isna() == False].copy()

    borrow_events = borrow_events[(borrow_events.start_datetime.isna() == False) & (
        borrow_events.end_datetime.isna() == False)]
    all_borrows = borrow_events[borrow_events.start_datetime <
                                '1942-01-01'].copy()
    metrics = ['jc_prediction', 'pa_prediction',
               'cn_prediction', 'aa_prediction']
    start_library = all_borrows.sort_values(
        by=['start_datetime'])[0:1].start_datetime.values[0]

    output_path = './data/partial_members_bipartite_circulation_events_predictions.csv'
    if os.path.exists(output_path):
        os.remove(output_path)
    partial_df[partial_df.member_id.isin(partial_members)].apply(get_specific_predictions, axis=1, number_of_results=10, limit_to_circulation=True, events_df=events_df,borrow_events=all_borrows, members_df=members_df, books_df=books_df, relative_date=start_library, predict_group='books', output_path=output_path)


if __name__ == '__main__':
    run_link_prediction()
