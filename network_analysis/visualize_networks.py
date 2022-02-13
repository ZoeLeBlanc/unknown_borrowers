import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
import altair as alt
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def scale_col(df, cols):
    for col in cols:
        df[col] = MinMaxScaler().fit_transform(df[col].values.reshape(-1, 1))
    return df

def update_borrow_count(df):
    df = df.drop('borrow_count', axis=1)
    return df


def get_member_borrow_counts(df, events_df):

    df = update_borrow_count(df)
    grouped_df = events_df.groupby(
        ['member_id']).size().reset_index(name='borrow_count')
    updated_df = pd.merge(df, grouped_df, on='member_id')
    return updated_df


def get_columns(df):
    original_columns = df.columns.to_list()
    columns = [c for c in original_columns if ('local' in c) | (
        'global' in c)]
    if 'borrow_count' in original_columns:
        columns = columns + ['borrow_count']
    if 'redundancy' in original_columns:
        columns = columns + ['redundancy']
    louvain = [c for c in original_columns if 'louvain' in c]
    if louvain:
        columns.remove(louvain[0])

    radius = [c for c in original_columns if 'radius' in c]
    if radius:
        columns.remove(radius[0])
    
    diameter = [c for c in original_columns if 'diameter' in c]
    if diameter:
        columns.remove(diameter[0])
    return columns


def get_correlation_df(df):
    columns = get_columns(df)
    df_copied = df[columns].copy()
    named_cols = [ 'BGRM', 'CoHITS', 'HITS', 'BiRank']
    final_cols = []
    for col in df_copied.columns.tolist():
        split_cols = col.split('_')
        remove_val = 'local' if 'local' in col else 'global'
        split_cols = [c for c in split_cols if (c != remove_val)]
        selected_cols = [string.capitalize() if string not in named_cols else string for string in split_cols] 
        selected_cols = ' '.join(selected_cols)
        final_cols.append(selected_cols)
        df_copied.rename(columns={col: selected_cols}, inplace=True)
    df_corr = df_copied.corr()
    
    return df_corr, final_cols

def generate_corr_chart(df, title):
    # data preparation
    corr_df, _ = get_correlation_df(df)
    pivot_cols = list(corr_df.columns)
    corr_df['cat'] = corr_df.index
    base = alt.Chart(corr_df).transform_fold(pivot_cols).encode(
        x=alt.X("cat:N", axis=alt.Axis(title='', labelAngle=-45)),  
        y=alt.Y('key:N', axis=alt.Axis(title=''))
    ).properties(height=300, width=300, title=title)
    boxes = base.mark_rect().encode(color=alt.Color(
        "value:Q", scale=alt.Scale(scheme="redyellowblue")))
    labels = base.mark_text(size=5, color="grey").encode(
        text=alt.Text("value:Q", format="0.1f"))
    chart = boxes + labels
    return chart


def get_melted_corr(df, df_type, node_type):
    corr_df, final_cols = get_correlation_df(df)
    corr_df['cat'] = corr_df.index
    melted_df = pd.melt(corr_df, id_vars=['cat'], value_vars=final_cols)
    melted_df['updated_variable'] = melted_df['cat'] + ' / ' + melted_df['variable']
    melted_df['type'] = df_type
    melted_df['node_type'] = node_type
    return melted_df


def compare_corr_chart(melted_df, melted_df2, x_type, y_type):
    concat_corr = pd.concat([melted_df, melted_df2])

    pivot_corr = pd.pivot(concat_corr, index=[
                          'updated_variable', 'cat', 'variable', 'node_type'], columns=['type'], values='value').reset_index()
    selection = alt.selection_multi(fields=['cat'], bind='legend')
    chart = alt.Chart(pivot_corr).mark_point(filled=True).encode(
        x=f'{x_type}:Q',
        y=f'{y_type}:Q',
        color=alt.Color('cat:N', scale=alt.Scale(scheme="redyellowblue"), legend=alt.Legend(title='Metric')),
        tooltip=['updated_variable', 'cat',
                    'variable', f'{y_type}:Q', f'{x_type}:Q'],
        opacity=alt.condition(selection, alt.value(1), alt.value(0.1)),
        shape='node_type:N'
    ).add_selection(
        selection
    ).resolve_scale(color='independent')
    return chart

def compare_node_variability(df, cols):
    abs_df = df.copy()
    ranked_items = []
    for c in cols:
        other_cols = [col for col in cols if c.split('_')[1] == col.split('_')[1]]
        other_cols.remove(c)
        comparison_col = c + '_' + other_cols[0]

        abs_df[comparison_col] = (abs_df[c] - abs_df[other_cols[0]]).abs()
        abs_df = abs_df.sort_values(by=comparison_col, ascending=False)
        top_dict = {'uri': abs_df.head(10).uri.tolist(), 'col_1': c, 'value_1': abs_df.head(10)[c].tolist(), 'col_2': other_cols[0], 'value_2': abs_df.head(10)[
            other_cols[0]].tolist(), 'ranking': 'top', 'abs_diff': abs_df.head(10)[comparison_col].tolist()}
        ranked_items.append(pd.DataFrame([top_dict]))

        abs_df = abs_df.sort_values(by=comparison_col, ascending=True)
        bottom_dict = {'uri': abs_df.head(10).uri.tolist(), 'col_1': c, 'value_1': abs_df.head(10)[c].tolist(), 'col_2': other_cols[0], 'value_2': abs_df.head(10)[
            other_cols[0]].tolist(), 'ranking': 'bottom', 'abs_diff': abs_df.head(10)[comparison_col].tolist()}
        ranked_items.append(pd.DataFrame([bottom_dict]))

    ranked_concat = pd.concat(ranked_items)
    ranked_exploded = ranked_concat.explode(
        ['uri', 'value_1', 'value_2', 'abs_diff'], ignore_index=True)
    selection = alt.selection_multi(fields=['uri'], bind='legend')
    chart = alt.Chart(ranked_exploded).mark_circle(size=100).encode(
        x='value_1',
        y='value_2',
        color=alt.Color('uri', scale=alt.Scale(scheme='plasma'), legend=alt.Legend(
            columns=4, symbolLimit=len(ranked_exploded.uri.unique().tolist()))),
        tooltip=['uri', 'col_1', 'value_1', 'col_2',
                 'value_2', 'ranking', 'abs_diff'],
        column='ranking',
        opacity=alt.condition(selection, alt.value(1), alt.value(0.1))
    ).add_selection(selection).properties(width=200).resolve_scale(x='independent', y='independent')
    return ranked_exploded, chart

def generate_scatter_regression_chart(df, x_type, y_type, color_col, facet_col, shape_col, title):
    selector = alt.selection_single(empty='all', fields=['uri'])
    base = alt.Chart(df).encode(
        x=alt.X(f'{x_type}:Q', axis=alt.Axis(title='')),
        y=alt.Y(f'{y_type}:Q', axis=alt.Axis(title='')),
    ).properties(
        height=150,
        width=200,
    )

    chart = alt.layer(
        base.mark_circle().encode(
            color=alt.Color(shape_col, scale=alt.Scale(
            scheme='set1'), sort=['members', 'books', 'killen', 'raphael-france']),
            tooltip=['uri', f'{x_type}:Q', f'{y_type}:Q'],
            opacity=alt.condition(selector, alt.value(1), alt.value(0.1)),
        ).add_selection(selector),
        base.transform_regression(f'{x_type}', f'{y_type}').mark_line(color='black', opacity=0.5)
    ).facet(facet=facet_col, columns=3).properties(title=title).resolve_scale(y='independent', x='independent')
    return chart

def visualize_node_variability(df, df1, df_type, df1_type, scaling, cols, index_col, subset_component, title, facet_col, color_col, shape_col):
    
    df_subset = df[df.component == 0][cols + index_col] if subset_component else df[cols + index_col]
    if scaling:
        df_subset = scale_col(df_subset, cols)
    df_subset = pd.melt(
        df_subset, id_vars=index_col, value_vars=cols, var_name=f'{df_type}_metric', value_name=f'{df_type}_value')
    df1_subset= df1[df1.component == 0][cols + index_col] if subset_component else df1[cols + index_col]
    if scaling:
        df1_subset = scale_col(df1_subset, cols)
    df1_subset= pd.melt(
        df1_subset, id_vars=index_col, value_vars=cols, var_name=f'{df1_type}_metric', value_name=f'{df1_type}_value')
    comparison_df = pd.merge(df_subset, df1_subset, on=index_col, how='outer')
    subset_df = comparison_df[(comparison_df[f'{df_type}_metric'].str.split('_').str[1] == comparison_df[f'{df1_type}_metric'].str.split('_').str[1])]
    subset_df['metric'] = subset_df[f'{df_type}_metric'].str.split('_').str[1]
    random_df = subset_df.sample(frac=0.5, random_state=42)
    random_df.loc[random_df.uri == 'killen', 'group'] = 'killen'
    random_df.loc[random_df.uri == 'raphael-france', 'group'] = 'raphael-france'
    chart = generate_scatter_regression_chart(random_df, f'{df1_type}_value', f'{df_type}_value', color_col, facet_col, shape_col, title)
    return chart

def final_network_stability_graph(melted_df, melted_df2, df_type, df_type2):
    melted_df['cat'] = melted_df.cat.str.split('_').str[1]
    melted_df['variable'] = melted_df.variable.str.split('_').str[1]
    melted_df.loc[melted_df.cat == 'count', 'cat'] = 'borrow frequency'
    melted_df.loc[melted_df.variable == 'count', 'variable'] = 'borrow frequency'
    melted_df['updated_variable'] = melted_df.cat + ' - ' + melted_df.variable
    

    melted_df2['cat'] = melted_df2.cat.str.split('_').str[1]
    melted_df2['variable'] = melted_df2.variable.str.split('_').str[1]
    melted_df2.loc[melted_df2.cat == 'count', 'cat'] = 'borrow frequency'
    melted_df2.loc[melted_df2.variable == 'count', 'variable'] = 'borrow frequency'
    melted_df2['updated_variable'] = melted_df2.cat + ' - ' + melted_df2.variable
    concat_corr = pd.concat([melted_df, melted_df2])

    pivot_corr = pd.pivot(concat_corr, index=[
                          'updated_variable', 'cat', 'variable'], columns='type', values='value').reset_index()
    selection = alt.selection_multi(fields=['cat'], bind='legend')
    chart = alt.Chart(pivot_corr).mark_circle().encode(
        x=f'{df_type}:Q',
        y=f'{df_type2}:Q',
        color=alt.Color('cat:N', scale=alt.Scale(scheme="redyellowblue"), legend=alt.Legend(title='Metric')),
        tooltip=['updated_variable', 'cat',
                 'variable', f'{df_type2}:Q', f'{df_type}:Q'],
        opacity=alt.condition(selection, alt.value(1), alt.value(0.1))
    ).add_selection(
        selection
    )
    return chart