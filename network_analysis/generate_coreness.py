import pandas as pd
import numpy as np
import time
import sys
sys.path.append("..")
from network_analysis.core_periphery_sbm import core_periphery as cp
from network_analysis.core_periphery_sbm import model_fit as mf
from network_analysis.read_write_networks import * 

def get_core_scores(G):
    # Initialize hub-and-spoke model and infer structure
    hub_time = time.time()
    print("Initializing hub-and-spoke model...")
    hubspoke = cp.HubSpokeCorePeriphery(n_gibbs=100, n_mcmc=10*len(G))
    hubspoke.infer(G)
    print("Done. Time taken: {:.2f} seconds".format(time.time() - hub_time))

    layer_time = time.time()
    print("Initializing layered model...")
    layered = cp.LayeredCorePeriphery(n_layers=4, n_gibbs=100, n_mcmc=10*len(G))
    layered.infer(G)
    print("Done. Time taken: {:.2f} seconds".format(time.time() - layer_time))

    # Get core and periphery assignments from hub-and-spoke model
    node2label_hs = hubspoke.get_labels(last_n_samples=50)

    # Get layer assignments from the layered model
    node2label_l = layered.get_labels(last_n_samples=50)
    nodelist, edgelist = generate_dataframes(G, False, True)

    # Get description length of hub-and-spoke model
    inf_labels_hs = hubspoke.get_labels(
        last_n_samples=50, prob=False, return_dict=False)
    mdl_hubspoke = mf.mdl_hubspoke(G, inf_labels_hs, n_samples=100000)

    # Get the description length of layered model
    inf_labels_l = layered.get_labels(
        last_n_samples=50, prob=False, return_dict=False)
    mdl_layered = mf.mdl_layered(G, inf_labels_l, n_layers=len(
        np.unique(inf_labels_l)), n_samples=100000)

    print("Hub-and-Spoke Model:", mdl_hubspoke, "Layered Model:", mdl_layered)
    sc_hub_spoke_df = pd.DataFrame.from_dict(
        node2label_hs, orient='index').reset_index()

    sc_hub_spoke_df.columns = ['node', 'global_hub_spoke']
    print('hubs and spokes', sc_hub_spoke_df.groupby('global_hub_spoke').size())

    sc_layered_df = pd.DataFrame.from_dict(
        node2label_l, orient='index').reset_index()

    sc_layered_df.columns = ['node', 'global_layer']
    print('layers', sc_layered_df.groupby('global_layer').size())

    node2coreness_l = layered.get_coreness(last_n_samples=50, return_dict=True)
    sc_layered_coreness_df = pd.DataFrame.from_dict(
        node2coreness_l, orient='index').reset_index()
    sc_layered_coreness_df.columns = ['node', 'global_coreness']

    merged_df = pd.merge(sc_hub_spoke_df, sc_layered_df, on='node')
    classified_nodes = pd.merge(merged_df, sc_layered_coreness_df, on='node')
    final_nodelist = pd.merge(classified_nodes, nodelist, left_on='node', right_on='label')
    return final_nodelist
    
