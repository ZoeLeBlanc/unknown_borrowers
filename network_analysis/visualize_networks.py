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
    columns = df.columns.to_list()
    columns = [c for c in columns if ('local' in c) | (
        'global' in c)] + ['borrow_count']
    columns.remove('local_louvain')
    columns.remove('global_louvain')
    if any('radius' in c for c in columns):
        columns.remove('global_graph_radius')
        columns.remove('local_graph_radius')
    if any('diameter' in c for c in columns):
        columns.remove('global_diameter')
        columns.remove('local_diameter')
    return columns


def get_correlation_df(df, events_df, is_members):
    if is_members:
        df = get_member_borrow_counts(df, events_df)
    columns = get_columns(df)
    df_corr = df[columns].corr()
    
    return df_corr

def generate_corr_chart(df, events_df, title, is_members):
    # data preparation
    corr_df = get_correlation_df(df, events_df, is_members)
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


def get_melted_corr(df, events_df, is_members, df_type):
    corr_df = get_correlation_df(df, events_df, is_members)
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
        color='cat:N',
        tooltip=['updated_variable', 'cat',
                 'variable', f'{df_type2}:Q', f'{df_type}:Q'],
        opacity=alt.condition(selection, alt.value(1), alt.value(0.1))
    ).add_selection(
        selection
    )
    return chart
