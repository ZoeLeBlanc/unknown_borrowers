import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
import altair as alt


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
    # if is_members:
    #     df = get_member_borrow_counts(df, events_df)
    columns = get_columns(df)
    df_corr = df[columns].corr()
    
    return df_corr

def generate_corr_chart(df, title):
    # data preparation
    corr_df = get_correlation_df(df)
    pivot_cols = list(corr_df.columns)
    corr_df['cat'] = corr_df.index
    base = alt.Chart(corr_df).transform_fold(pivot_cols).encode(
        x="cat:N",  y='key:N').properties(height=300, width=300, title=title)
    boxes = base.mark_rect().encode(color=alt.Color(
        "value:Q", scale=alt.Scale(scheme="redyellowblue")))
    labels = base.mark_text(size=5, color="grey").encode(
        text=alt.Text("value:Q", format="0.1f"))
    chart = boxes + labels
    return chart


def get_melted_corr(df, df_type):
    corr_df = get_correlation_df(df)
    corr_df['cat'] = corr_df.index
    columns = get_columns(df)
    melted_df = pd.melt(corr_df, id_vars=['cat'], value_vars=columns)
    melted_df['updated_variable'] = melted_df['cat'] + \
        ' / ' + melted_df['variable']
    melted_df['type'] = df_type
    return melted_df


def compare_corr_chart(melted_df, melted_df2, df_type, df_type2):
    concat_corr = pd.concat([melted_df, melted_df2])

    pivot_corr = pd.pivot(concat_corr, index=[
                          'updated_variable', 'cat', 'variable'], columns='type', values='value').reset_index()
    selection = alt.selection_multi(fields=['cat'], bind='legend')
    chart = alt.Chart(pivot_corr).mark_circle().encode(
        x=f'{df_type}:Q',
        y=f'{df_type2}:Q',
        color=alt.Color('cat:N', scale=alt.Scale(scheme="redyellowblue")),
        tooltip=['updated_variable', 'cat',
                 'variable', f'{df_type2}:Q', f'{df_type}:Q'],
        opacity=alt.condition(selection, alt.value(1), alt.value(0.1))
    ).add_selection(
        selection
    ).resolve_scale(color='independent')
    return chart

def compare_node_variability(df):
    local_cols = [ col for col in df.columns if 'local' in col]
    global_cols = [ col for col in df.columns if 'global' in col]
    cols = df[global_cols + local_cols].columns.tolist()    
    cols.remove('global_louvain')
    cols.remove('local_louvain')
    abs_df = df.loc[df.component ==0][cols + ['uri']].copy()
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
    chart = alt.Chart(ranked_exploded).mark_circle(size=100).encode(
        x='value_1',
        y='value_2',
        color=alt.Color('uri', scale=alt.Scale(scheme='plasma'), legend=alt.Legend(
            columns=4, symbolLimit=len(ranked_exploded.uri.unique().tolist()))),
        tooltip=['uri', 'col_1', 'value_1', 'col_2',
                 'value_2', 'ranking', 'abs_diff'],
        column='ranking'
    ).properties(width=200).resolve_scale(x='independent', y='independent')
    return ranked_exploded, chart
