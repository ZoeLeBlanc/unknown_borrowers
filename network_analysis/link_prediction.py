import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
import networkx as nx
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import os

from tqdm.notebook import trange, tqdm
tqdm.pandas()
import warnings
warnings.filterwarnings("ignore")
from IPython.display import display, Markdown, HTML
import sys
sys.path.append("..")
from bigraph.predict import pa_predict, jc_predict, cn_predict,aa_predict, katz_predict
from bigraph.evaluation import evaluation
from network_analysis.birankpy import BipartiteNetwork

bipartite_metrics = ['jc_prediction', 'pa_prediction',
           'cn_prediction', 'aa_prediction']

def get_bipartite_link_predictions(graph):
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
    return all_preds


def get_predictions_by_metric(row, metric, predictions_df, circulation_books, limit_to_circulation=True):
    if limit_to_circulation:
        subset_predictions = predictions_df[(predictions_df.member_id == row.member_id) & (
            predictions_df.item_uri.isin(circulation_books))].sort_values(by=f'{metric}', ascending=False)
    else:
        subset_predictions = predictions_df[(
            predictions_df.member_id == row.member_id)].sort_values(by=f'{metric}', ascending=False)

    return subset_predictions[['member_id', 'item_uri', f'{metric}']]


def get_full_predictions(row, number_of_results, limit_to_circulation, predictions_df, relative_date, predict_group, output_path):
    grouped_col = 'item_uri' if predict_group == 'books' else 'member_id'
    index_col = 'member_id' if predict_group == 'books' else 'item_uri'
    identified_top_predictions = {}

    circulation_start = row.subscription_starttime - relative_date

    all_possible_circulations = events_df[(
        row.subscription_endtime >= events_df.end_datetime)]
    circulation_events = events_df[events_df.start_datetime.between(
        circulation_start, row.subscription_endtime) | events_df.end_datetime.between(circulation_start, row.subscription_endtime)]

    popular_all = all_possible_circulations.groupby([grouped_col]).size().reset_index(
        name='counts').sort_values(['counts'], ascending=False)[0:number_of_results]
    popular_current = circulation_events.groupby([grouped_col]).size().reset_index(
        name='counts').sort_values(['counts'], ascending=False)[0:number_of_results]

    circulation_all = all_possible_circulations[grouped_col].unique().tolist()
    circulation_subset = circulation_events[grouped_col].unique().tolist()

    identified_top_predictions[f'popular_all_{predict_group}'] = popular_all[grouped_col].tolist(
    )
    identified_top_predictions['popular_all_counts'] = popular_all.counts.tolist(
    )
    identified_top_predictions[f'popular_current_{predict_group}'] = popular_current[grouped_col].tolist(
    )
    identified_top_predictions['popular_current_counts'] = popular_current.counts.tolist(
    )

    for idx, m in enumerate(metrics):

        subset_all_predictions = get_predictions_by_metric(
            row, m, predictions_df, circulation_all, limit_to_circulation)
        subset_predictions = get_predictions_by_metric(
            row, m, predictions_df, circulation_subset, limit_to_circulation)
        identified_top_predictions[f'{m}_all'] = subset_all_predictions[0:number_of_results][grouped_col].tolist(
        )
        identified_top_predictions[f'{m}_subset'] = subset_predictions[0:number_of_results][grouped_col].tolist(
        )

        identified_top_predictions[f'{m}_all_scores'] = subset_predictions[0:number_of_results][m].tolist(
        )
        identified_top_predictions[f'{m}_subset_scores'] = subset_predictions[0:number_of_results][m].tolist(
        )
    df_final = pd.DataFrame.from_dict(
        identified_top_predictions, orient='columns')

    df_final[f'{index_col}'] = row[index_col]
    df_final['subscription_starttime'] = row.subscription_starttime
    df_final['subscription_endtime'] = row.subscription_endtime
    df_final['known_borrows'] = row.known_borrows

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
        print("Num. (test, val) edges returned: (", len(test_edges), ", ", len(val_edges), ")")

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
