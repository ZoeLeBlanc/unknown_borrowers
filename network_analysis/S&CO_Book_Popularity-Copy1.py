#!/usr/bin/env python
# coding: utf-8

# # Book Popularity in S&co

# Jupyter notebook to explore the stability of book popularity over time at s&co

# ### Install Libraries and Data

# In[21]:


## Install packages and libraries 
#pip install pandas networkx altair vega numpy matplotlib tqdm


# In[10]:


import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
import os
import json
import networkx as nx
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import itertools
import collections
from tqdm.notebook import trange, tqdm


# In[12]:


#read in events and members csvs
from execeptional_metadata import *
members_df, books_df, events_df = get_shxco_exceptional_data()


# ### Reformat Events to Separate Out Subscriptions and Borrows

# In[3]:


# turn events into timestamped values
events_df['start_datetime'] = pd.to_datetime(events_df.start_date, format='%Y-%m-%d', errors='coerce')
events_df['end_datetime'] = pd.to_datetime(events_df.end_date, format='%Y-%m-%d', errors='coerce')
events_df['subscription_purchase_datetime'] = pd.to_datetime(events_df.subscription_purchase_date, format='%Y-%m-%d', errors='coerce')
events_df['index'] = events_df.index
events_df = events_df.reset_index(drop=True)


# In[4]:


# copy and subset only events related to subscriptions, and expand uri fields
subscription_events = events_df.copy()
subscription_events = subscription_events[subscription_events.event_type.isin(['Subscription', 'Renewal', 'Supplement'])]

subscription_events[['first_member_uri','second_member_uri']] = subscription_events.member_uris.str.split(';', expand=True)
subset_subscription_events = subscription_events[['subscription_purchase_datetime','start_datetime', 'event_type', 'member_names', 'end_datetime', 'index', 'first_member_uri','second_member_uri']]
member_subs = pd.merge(left=subset_subscription_events, right=members_df, left_on="first_member_uri", right_on="uri")
member_subs = member_subs[member_subs.has_card]


# In[5]:


# subset only events related to borrowing and create separate year and month columns for grouping
borrow_events = events_df[(events_df.event_type == 'Borrow') &(events_df.start_date.str.len() > 9) & (events_df.end_date.str.len() > 9)]
borrow_events.loc[:,'year'] = borrow_events.start_datetime.dt.year
borrow_events.loc[:,'month'] = borrow_events.start_datetime.dt.month
borrow_events[['first_member_uri','second_member_uri']] = borrow_events.member_uris.str.split(';', expand=True)


# #### Create a Subscription Active Field on Borrow Events

# In[6]:


# Merge borrow and subscription events to check if subscriptions where active when books where checked out
merged_borrows_subs = pd.merge(borrow_events, subset_subscription_events, on=['first_member_uri'], how='left')
merged_borrows_subs['subscription_active'] = np.where(((merged_borrows_subs.subscription_purchase_datetime_y >= merged_borrows_subs.start_datetime_x )|(merged_borrows_subs.start_datetime_x <= merged_borrows_subs.end_datetime_y )), True, False)


# In[7]:


# Clean out the merged dataframe to only include rows for borrow events
merged_borrows_subs = merged_borrows_subs[merged_borrows_subs.event_type_x=='Borrow']
merged_borrows_subs = merged_borrows_subs[['start_date', 'first_member_uri', 'subscription_active', 'item_title', 'index_x']]
merged_borrows_subs = merged_borrows_subs[merged_borrows_subs.item_title.isna() == False]
merged_borrows_subs = merged_borrows_subs[merged_borrows_subs.duplicated() == False]


# In[8]:


# Merge updated borrow events into original borrow events
merged_borrows_subs = merged_borrows_subs.rename(columns={'index_x':'index'})
updated_borrow_events = pd.merge(borrow_events, merged_borrows_subs, on=['index','start_date', 'first_member_uri', 'item_title'], how='inner')


# #### Subset borrows to prior to 1942 and define variables for years, months, and seasons

# In[9]:


# Assign updated borrow events back to variable and get list of years before 1942 and all months
borrow_events = updated_borrow_events[updated_borrow_events.year < 1942]
month_ranges = {12: 'Winter',1: 'Winter',2: 'Winter',3 :'Spring',4:'Spring',5:'Spring',6:'Summer',7:'Summer',8:'Summer',9:'Fall', 10:'Fall',11:'Fall'}
borrow_events['season'] = borrow_events.month.map(month_ranges)
years = borrow_events.year.unique().tolist()
months = borrow_events.month.unique().tolist()
seasons = borrow_events.season.unique().tolist()


# In[11]:


def first_member_name(row):
    return members_df[members_df.uri == row].name.values[0]  
    
borrow_events['first_name'] = borrow_events['first_member_uri'].apply(first_member_name)


# ### Calculate Book Popularity

# #### Calculate straight count-based popularity of top ten books

# In[152]:


# Top ten books signed 
popularity_counts = borrow_events.groupby(['item_title']).size().reset_index(name="overall_times_checked_out")
popularity_counts.sort_values(by=['overall_times_checked_out'], ascending=False)[0:10]


# #### Calculate book popularity seasonally and annually

# In[153]:


# Top ten books signed 
seasonal_counts = borrow_events.groupby(['year', 'season', 'item_title']).size().reset_index(name="seasonal_times_checked_out")
seasonal_counts.sort_values(by=['seasonal_times_checked_out'], ascending=False)[0:10]


# We can see from these two approaches that overall popularity surfaces more books whereas seasonal highlights periodicals

# In[187]:


popularity_metrics = pd.merge(popularity_counts, seasonal_counts, on=['item_title'])

season_ranges = {'Winter':'01', 'Spring':'04', 'Summer':'07','Fall':'10'}
popularity_metrics['month'] = popularity_metrics.season.map(season_ranges)
popularity_metrics['date'] = '01-' + popularity_metrics.month + '-' +popularity_metrics.year.astype(int).astype(str)
popularity_metrics['datetime'] = pd.to_datetime(popularity_metrics.date, format='%d-%m-%Y', errors='coerce')
# pd.cut(book_counts['total_counts'], bins=3, labels=['low', 'medium', 'high'])

field_x = 'overall_times_checked_out'
field_y = 'seasonal_times_checked_out'

corl = popularity_metrics[[field_x,field_y]].corr().iloc[0,1]

chart = alt.Chart(popularity_metrics).mark_circle().encode(
    x=field_x,
    y=field_y,
    color=alt.Color('year:O', scale=alt.Scale(scheme='redyellowblue')),
    tooltip=['item_title', 'overall_times_checked_out', 'seasonal_times_checked_out', 'yearmonth(datetime)'],
)
text = alt.Chart({'values':[{}]}).mark_text(
    align="left", baseline="top"
).encode(
    x=alt.value(5),  # pixels from left
    y=alt.value(5),  # pixels from top
    text=alt.value(f"r: {corl:.3f}"),
)
chart + text + chart.transform_regression(field_x, field_y).mark_line(color='black')


# We can see that generally there is some relationship between overall checkouts and frequency of checkout for some of the most popular titles but that for most popularity overall does not translate to consistency (which you would expect!)

# **So real question is how should we measure popularity here?**

# - Overall counts of popularity
# - Consistency seasonally or annually
# - Some metric combining these? 
# - Popularity subseted out by types of members (subscribers vs not, or exceptional members vs not) OR types of books (missing books vs not)

# In[37]:


def get_top_books_by_year(rows):
    sorted_rows = rows.sort_values(by=['times_checked_out'], ascending=False)[0:3]
    print(f'In {sorted_rows[0:1].season.values[0]} {sorted_rows[0:1].year.values[0]}, top books were: {dict(zip(sorted_rows.item_title.tolist(), sorted_rows.times_checked_out.tolist()))}')

book_counts.groupby(['year', 'season']).apply(get_top_books_by_year)


# But ideally we would like to view this as a graph showing the stability for each book over time in terms of its checkout rate

# In[62]:


season_ranges = {'Winter':'01', 'Spring':'04', 'Summer':'07','Fall':'10'}
book_counts['month'] = book_counts.season.map(season_ranges)
book_counts['date'] = '01-' + book_counts.month + '-' +book_counts.year.astype(int).astype(str)
book_counts['datetime'] = pd.to_datetime(book_counts.date, format='%d-%m-%Y', errors='coerce')
book_counts


# In[78]:


grouped_books = book_counts.groupby(['item_title']).size().reset_index(name='total_counts')
grouped_books.hist('total_counts')


# In[ ]:


book_counts = pd.merge(book_counts, grouped_books, on=['item_title'])


# In[145]:


book_counts[(book_counts.total_counts > 22) & (book_counts.times_checked_out == 0)]


# In[143]:


selection = alt.selection_multi(fields=['item_title'], bind='legend')
alt.Chart(book_counts[book_counts.total_counts > 22]).mark_line(point=True).encode(
    x='datetime:T',
    y='times_checked_out',
    color='item_title',
    opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
    tooltip=['item_title', 'datetime']
).add_selection(selection)


# In[135]:


book_counts['frequency']=pd.cut(book_counts['total_counts'], bins=3, labels=['low', 'medium', 'high'])


# In[141]:


field_1 = 'total_counts'
field_2 = 'times_checked_out'
base = alt.Chart(book_counts).encode(
    x=alt.X(f'{field_1}:Q', axis=alt.Axis(title='')),
    y=alt.Y(f'{field_2}:Q', axis=alt.Axis(title='')),
).properties(
    height=75,
    width=75,
)

# selection = alt.selection_single(fields=['node_title'])
chart = alt.layer(
    base.mark_circle().encode(
#         opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        tooltip=['item_title', f'{field_1}', f'{field_2}'],
        color=alt.Color('frequency:N', sort=['high','medium','low'])
    ),
    base.transform_regression(f'{field_1}', f'{field_2}').mark_line(color='black')
).facet(
    'yearmonth(datetime):N', columns=4
).properties(title='Comparing Graph Metrics for Each Member by Seasons and Year').resolve_scale(x='independent', y='independent').configure_legend(labelLimit= 0)
chart


# In[114]:


# selection = alt.selection_multi(fields=['item_title'], bind='legend')
alt.data_transformers.disable_max_rows()
alt.Chart(book_counts[book_counts.total_counts > 14]).mark_circle().encode(
    x='total_counts:Q',
    y='times_checked_out:Q',
    color=alt.Color('item_title:N', scale=alt.Scale(scheme='redyellowblue')),
#     opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
    tooltip=['item_title','total_counts', 'times_checked_out']
).properties(
    height=50,
    width=50
).facet(
    facet='yearmonth(datetime):N',
    columns=4
)
# .add_selection(selection)


# ### Calculate Uniqueness of Books By Borrows Each Season

# In[12]:


# Consistency of unique books over seasons
unique_books = borrow_events.groupby(['year', 'season'])['item_title'].apply(lambda x: list(np.unique(x))).reset_index(name='list_of_unique_books')
exploded_books = unique_books.explode('list_of_unique_books')
grouped_books = exploded_books.groupby('list_of_unique_books').size().reset_index(name='counts')
unique_book_counts = pd.merge(exploded_books, grouped_books, on=['list_of_unique_books'])
unique_book_counts.year = unique_book_counts.year.astype(int).astype(str) + '-01-01'
unique_book_counts.year = pd.to_datetime(unique_book_counts.year, format='%Y-%m-%d', errors='coerce')
unique_book_counts.hist('counts')


# In[ ]:





# In[20]:


unique_books = borrow_events.groupby(['year', 'season'])['item_title'].apply(lambda x: list(np.unique(x))).reset_index(name='list_of_unique_books')
unique_books


# In[19]:


unique_book_counts.sort_values(by=['counts', 'year', 'season'], ascending=False)


# In[13]:


# Group members and books by years and months
members_grouped = borrow_events.groupby([borrow_events.year,borrow_events.season,borrow_events.month, borrow_events.item_title])['first_member_uri'].apply(list).reset_index(name='list_members')

books_grouped1 = borrow_events.groupby([borrow_events.year,borrow_events.season,borrow_events.month, borrow_events.first_member_uri])['item_title'].apply(list).reset_index(name='list_books')



# In[16]:


len(members_df.uri.unique()), len(borrow_events.first_member_uri.unique())


# In[14]:


names = borrow_events[['first_member_uri', 'first_name']]
books_grouped = pd.merge(books_grouped1, names, on=['first_member_uri'])


# In[14]:


members_df[members_df.uri == 'https://shakespeareandco.princeton.edu/members/oerthel/']


# In[15]:


members_grouped[0:1].list_members.values


# In[391]:


def create_members_network(rows, graph):
    '''Create a graph from all members using shared books to connect members'''
    combos = list(itertools.combinations(rows.list_members, 2))
    if len(combos) > 0:
        graph.add_nodes_from(rows.list_members, item_title = rows.item_title, year=rows.year, month=rows.month, season=borrow_events.season)
        graph.add_edges_from(combos, item_title = rows.item_title, year=rows.year, month=rows.month, season=borrow_events.season)
    if len(combos) == 0:
        graph.add_nodes_from(rows.list_members, item_title = rows.item_title, year=rows.year, month=rows.month, season=borrow_events.season)

def create_books_network(rows, graph):
    '''Create a graph from all books using members who read the same book as an edges'''
    combos = list(itertools.combinations(rows.list_books, 2))
    if len(combos) > 0:
        graph.add_nodes_from(rows.list_books, member_name = rows.member_names, year=rows.year, month=rows.month, season=borrow_events.season)
        graph.add_edges_from(combos, member_name = rows.member_names, year=rows.year, month=rows.month, season=borrow_events.season)
    if len(combos) == 0:
        graph.add_nodes_from(rows.list_books, member_name = rows.member_names, year=rows.year, month=rows.month, season=borrow_events.season)


def create_bipartite_network(rows, graph):
    graph.add_node(rows.member_names, group='members', name=rows.member_names, year=rows.year, month=rows.month, season=rows.season)
    graph.add_node(rows.item_title, group='books', name=rows.item_title, year=rows.year, month=rows.month, season=rows.season)
    graph.add_edge(rows.member_names, rows.item_title, year=rows.year, month=rows.month, season=rows.season)


# In[395]:


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
        top_nodes = [n for n in graph.nodes if graph.nodes[n]['group'] == 'members']
        bottom_nodes = [n for n in graph.nodes if graph.nodes[n]['group'] == 'books']
#         print(top_nodes, bottom_nodes)
        graph_components = sorted(nx.connected_components(graph), key=len, reverse=True)
        #Degree
        global_degree = bipartite.degree_centrality(graph, top_nodes)
        #Clustering
        top_global_clustering = bipartite.clustering(graph, top_nodes)
        bottom_global_clustering = bipartite.clustering(graph, bottom_nodes)
        #Closeness
        global_closeness = bipartite.closeness_centrality(graph, top_nodes)
        #Betweenness
        global_betweenness = bipartite.betweenness_centrality(graph, top_nodes)
        
        for index, component in enumerate(graph_components):
            subgraph = graph.subgraph(component)
            top_nodes = [n for n in subgraph.nodes if subgraph.nodes[n]['group'] == 'members']
            bottom_nodes = [n for n in subgraph.nodes if subgraph.nodes[n]['group'] == 'books']
            #Degree
            local_degree = bipartite.degree_centrality(subgraph, top_nodes)
            #Clustering
            top_local_clustering = bipartite.clustering(subgraph, top_nodes)
            bottom_local_clustering = bipartite.clustering(subgraph, bottom_nodes)
            #Closeness
            local_closeness = bipartite.closeness_centrality(subgraph, top_nodes)
            #Betweenness
            try:
                top_local_betweenness = bipartite.betweenness_centrality(subgraph, top_nodes)
            except ZeroDivisionError:
                top_local_betweenness = {k: 0 for k in top_nodes}
                
            try:
                bottom_local_betweenness = bipartite.betweenness_centrality(subgraph, bottom_nodes)
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
        graph_components = sorted(nx.connected_components(graph), key=len, reverse=True)
        clustering = nx.clustering(graph)

        for index, component in enumerate(graph_components):
            subgraph = graph.subgraph(component)
            diameter = nx.diameter(subgraph) if subgraph.size() > 0 else 0
            radius = nx.radius(subgraph) if subgraph.size() > 0 else 0
            eigenvector = nx.eigenvector_centrality_numpy(subgraph, max_iter=600) if subgraph.size() > 1 else {k: 0 for k in subgraph.nodes}
            local_betweenness = nx.betweenness_centrality(subgraph)
            
            for d, v in subgraph.nodes(data=True):
                if 'member_name' in v:
                    check_sub_status = borrow_events[(borrow_events.item_title == d) & (borrow_events.member_names == v['member_name']) & (borrow_events.year == v['year']) & (borrow_events.month == v['month'])]
                if 'item_title' in v:
                    check_sub_status = borrow_events[(borrow_events.member_names == d) & (borrow_events.item_title == v['item_title']) & (borrow_events.year == v['year']) & (borrow_events.month == v['month'])]
                v['subscription_active'] = check_sub_status.subscription_active.values[0]
                v['degree'] = degree[d]
#                 print(d in eigenvector)
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


# In[396]:


from networkx.algorithms import bipartite
top_nodes = [n for n in bipartite_graph.nodes if bipartite_graph.nodes[n]['group'] == 'members']
bottom_nodes = [n for n in bipartite_graph.nodes if bipartite_graph.nodes[n]['group'] == 'books']


# In[415]:


# Process the borrowers graph through the network metrics and return dataframes for each month and each year
books_dfs = []
members_dfs = []
bipartite_dfs = []
for year in tqdm(years[0:4]):
    for season in seasons:
#         print(year, month)
        # Make books graph for each month range
        book_graph = nx.Graph()
        books_grouped[(books_grouped.year == year) & (books_grouped.season == season)].apply(create_books_network, graph=book_graph, axis=1)
        book_graph.remove_edges_from(nx.selfloop_edges(book_graph))
        books_graph_processed = get_network_metrics(book_graph)
        books_df = pd.DataFrame.from_dict(dict(books_graph_processed.nodes(data=True)), orient='index').reset_index(drop=True)
        books_df['season'] = list(month.keys())[0]
        books_dfs.append(books_df)

        # Make members graph for each month range
        member_graph = nx.Graph()
        members_grouped[(members_grouped.year == year) & (members_grouped.season == season)].apply(create_members_network, graph=member_graph, axis=1)
        member_graph.remove_edges_from(nx.selfloop_edges(member_graph))
        members_graph_processed = get_network_metrics(member_graph)
        members_df = pd.DataFrame.from_dict(dict(members_graph_processed.nodes(data=True)), orient='index').reset_index(drop=True)
        members_df['season'] = list(month.keys())[0]
        members_dfs.append(members_df)
        
        bp_graph = nx.Graph()
        borrow_events[(borrow_events.year == year) & (borrow_events.season == season)].apply(create_bipartite_network, graph=bp_graph, axis=1)
        bp_graph_processed = get_network_metrics(bp_graph, True)
        bp_df = pd.DataFrame.from_dict(dict(bp_graph_processed.nodes(data=True)), orient='index').reset_index(drop=True)
        bp_df['season'] = list(month.keys())[0]
        bipartite_dfs.append(bp_df)


# In[416]:


# Join the processed dataframes and then melt them 
joined_books_dfs = pd.concat(books_dfs)
# joined_books_dfs = joined_books_dfs.drop('index', axis=1)

joined_members_dfs = pd.concat(members_dfs)
# joined_members_dfs = joined_members_dfs.drop('index', axis=1)

joined_bps_dfs = pd.concat(bipartite_dfs)

melted_books_dfs = pd.melt(joined_books_dfs, id_vars=['member_name', 'node_title', 'year', 'month', 'subscription_active', 'season'])
melted_members_dfs = pd.melt(joined_members_dfs, id_vars=['item_title', 'node_title', 'year', 'month', 'subscription_active', 'season'])

melted_bps_dfs = pd.melt(joined_bps_dfs, id_vars=['group', 'name', 'year', 'month', 'node_title', 'season', 'component'])
melted_bps_dfs[0:1]
# joined_bps_dfs.sort_values(by=['year', 'season', 'month', 'local_degree'], ascending=[True, True, True, False]).groupby(['year', 'season']).head(20)


# In[417]:


melted_bps_dfs.year = melted_bps_dfs.year.astype(int).astype(str) + '-'+ melted_bps_dfs.month.astype(int).astype(str)+ '-01'
melted_bps_dfs.year = pd.to_datetime(melted_bps_dfs.year, format='%Y-%m-%d', errors='coerce')
melted_bps_dfs[0:1]


# In[418]:


joined_bps_dfs.sort_values(by=['year', 'season', 'month', 'global_degree'], ascending=[True, True, True, False]).groupby(['year', 'season', 'group']).head(2)


# In[411]:



selection_1 = alt.selection_multi(fields=['year'], bind='legend')
alt.Chart(melted_bps_dfs).mark_line(point=True).encode(
    x='year:T',
    y=alt.Y('value:Q', axis=alt.Axis(title='')),
    color=alt.Color('node_title:N', scale=alt.Scale(scheme='plasma')),
    opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
    column='variable:O',
).add_selection(
    selection
).properties(
    height=250,
    width=250,
    title='Each Metric of the Books Graph Averaged by Season and Colored by Year'
).resolve_scale(x='independent', y='independent')


# In[270]:


top_book_nodes = joined_books_dfs.sort_values(by=['year', 'season', 'month', 'closeness'], ascending=[True, True, True, False]).groupby(['year', 'season']).head(20)

top_book_nodes.year = top_book_nodes.year.astype(int).astype(str) + '-01-01'
top_book_nodes.year = pd.to_datetime(top_book_nodes.year, format='%Y-%m-%d', errors='coerce')
# top_book_nodes[0:10]


# In[271]:


test = top_book_nodes.groupby('node_title').size().reset_index(name='counts')
top_book_nodes = pd.merge(top_book_nodes, test, on=['node_title'])
top_book_nodes[0:1]


# In[274]:


selection = alt.selection_single(fields=['node_title'])

alt.Chart(top_book_nodes).mark_line(point=True).encode(
    x='year(year):T',
    y='closeness:Q',
    color='node_title',
#     tooltip=['node_title', 'closeness', 'year', 'season'],
    opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
).add_selection(selection)


# In[ ]:


# Average each of the metrics by season and compare rates by metric, members vs books, and across years. 
# Click on the legend years to highlight lines and hover on the points to get more info for each season
alt.data_transformers.disable_max_rows()
selection = alt.selection_multi(fields=['year'], bind='legend')
chart = alt.Chart(melted_members_dfs[melted_members_dfs.variable != 'subgraph']).mark_line(point=True).encode(
    x='seasons',
    y=alt.Y('mean(value):Q', axis=alt.Axis(title='')),
    color=alt.Color('year:N', scale=alt.Scale(scheme='plasma')),
    opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
    column='variable',
    tooltip=['seasons', 'variable', 'mean(value)', 'year']
).add_selection(
    selection
).properties(
    height=250,
    width=250,
    title='Each Metric of the Members Graph Averaged by Season and Colored by Year'
).resolve_scale(x='independent', y='independent')

selection_1 = alt.selection_multi(fields=['year'], bind='legend')
chart_1 = alt.Chart(melted_books_dfs[melted_books_dfs.variable != 'subgraph']).mark_line(point=True).encode(
    x='seasons',
    y=alt.Y('mean(value):Q', axis=alt.Axis(title='')),
    color=alt.Color('year:N', scale=alt.Scale(scheme='plasma')),
    opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
    column='variable',
    tooltip=['seasons', 'variable', 'mean(value)', 'year']
).add_selection(
    selection
).properties(
    height=250,
    width=250,
    title='Each Metric of the Books Graph Averaged by Season and Colored by Year'
).resolve_scale(x='independent', y='independent')

alt.vconcat(chart, chart_1)


# In[ ]:


# Delve into individual metrics per node for both books and members
# You can compare 
field_1 = 'closeness'
field_2 = 'betweenness'
base = alt.Chart(joined_members_dfs).encode(
    x=f'{field_1}:Q',
    y=f'{field_2}:Q',
).properties(
    height=50,
    width=50,
)

selection = alt.selection_single(fields=['node_title'])
chart = alt.layer(
    base.mark_circle().encode(
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        tooltip=['node_title', f'{field_1}', f'{field_2}']
    ),
    base.transform_regression(f'{field_1}', f'{field_2}').mark_line()
).add_selection(selection).facet(
    column='seasons',
    row='year'
).properties(title='Comparing Graph Metrics for Each Member by Seasons and Year').resolve_scale(x='independent', y='independent')

base_1 = alt.Chart(joined_books_dfs).encode(
    x=f'{field_1}:Q',
    y=f'{field_2}:Q',
).properties(
    height=50,
    width=50,
)

selection_1 = alt.selection_single(fields=['node_title'])
chart_1 = alt.layer(
    base_1.mark_circle().encode(
        opacity=alt.condition(selection_1, alt.value(1), alt.value(0.2)),
        tooltip=['node_title', f'{field_1}', f'{field_2}']
    ),
    base_1.transform_regression(f'{field_1}', f'{field_2}').mark_line()
).add_selection(selection_1).facet(
    column='seasons',
    row='year'
).properties(title='Comparing Graph Metrics for Each Book by Seasons and Year').resolve_scale(x='independent', y='independent')

alt.hconcat(chart, chart_1)


# In[ ]:



chart = alt.Chart(joined_members_dfs).mark_bar(size=2).encode(
    x='degree:Q',
    y=alt.Y('count():Q', axis=alt.Axis(title='')),
    color=alt.Color('subscription_active', scale=alt.Scale(scheme='plasma')),
    column='seasons',
    row='year'
).properties(
    height=50,
    width=50,
    title='The Degree Centrality of All Members Linked by Shared Books'
).resolve_scale(x='independent', y='independent')

chart_1 = alt.Chart(joined_books_dfs).mark_bar(size=2).encode(
    x='degree:Q',
    y=alt.Y('count():Q', axis=alt.Axis(title='')),
    color=alt.Color('subscription_active', scale=alt.Scale(scheme='plasma')),
    column='seasons',
    row='year'
).properties(
    height=50,
    width=50,
    title='The Degree Centrality of All Books Read and Linked by Shared Members'
).resolve_scale(x='independent', y='independent')
alt.hconcat(chart, chart_1)
# chart.save('books_degree_over_time_scale_dependent.png', scale_factor=2.0)


# In[ ]:


# book_graph = nx.Graph()
# books_grouped.apply(create_books_network, graph=book_graph, axis=1)
# degree_sequence = sorted([d for n, d in book_graph.degree()], reverse=True)  # degree sequence
# degreeCount = collections.Counter(degree_sequence)
# deg, cnt = zip(*degreeCount.items())

# fig, ax = plt.subplots()
# plt.bar(deg, cnt, width=0.80, color="b")

# plt.title("Degree Histogram for Entire Books Network")
# plt.ylabel("Count")
# plt.xlabel("Degree")
# ax.set_xticks([d + 0.4 for d in deg])
# ax.set_xticklabels(deg)

# # draw graph in inset
# plt.axes([0.4, 0.4, 0.5, 0.5])
# # Gcc = G.subgraph(sorted(nx.connected_components(book_graph), key=len, reverse=True)[0])
# pos = nx.spring_layout(book_graph)
# plt.axis("off")
# nx.draw_networkx_nodes(book_graph, pos, node_size=20)
# nx.draw_networkx_edges(book_graph, pos, alpha=0.4)
# plt.show()


# In[ ]:


# members_graph = nx.Graph()
# members_grouped.apply(create_members_network, graph=members_graph, axis=1)
# members_graph.remove_edges_from(nx.selfloop_edges(members_graph))
# degree_sequence = sorted([d for n, d in members_graph.degree()], reverse=True)  # degree sequence
# degreeCount = collections.Counter(degree_sequence)
# deg, cnt = zip(*degreeCount.items())

# fig, ax = plt.subplots()
# plt.bar(deg, cnt, width=0.80, color="b")

# plt.title("Degree Histogram for Entire Members Network")
# plt.ylabel("Count")
# plt.xlabel("Degree")
# ax.set_xticks([d + 0.4 for d in deg])
# ax.set_xticklabels(deg)

# # draw graph in inset
# plt.axes([0.4, 0.4, 0.5, 0.5])
# # Gcc = G.subgraph(sorted(nx.connected_components(members_graph), key=len, reverse=True)[0])
# pos = nx.spring_layout(members_graph)
# plt.axis("off")
# nx.draw_networkx_nodes(members_graph, pos, node_size=20)
# nx.draw_networkx_edges(members_graph, pos, alpha=0.4)
# plt.show()


# In[ ]:




