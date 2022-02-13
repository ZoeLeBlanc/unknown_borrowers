import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)

from sklearn import preprocessing

from ast import literal_eval

def split_cols(original_df):
    df = original_df[original_df.exceptional_types.isna() == False]
    df.exceptional_types = df.exceptional_types.apply(literal_eval)
    if 'exceptional_counts' in df.columns:
        df.exceptional_counts = df.exceptional_counts.apply(literal_eval)
        df = df.explode(['exceptional_types', 'exceptional_counts'])
    else:
        df = df.explode(['exceptional_types'])
    return df

def get_excpetional_counts(members_exploded, members_data):
    members_exceptional_counts = members_exploded.groupby(['member_id'])['exceptional_counts'].sum().reset_index()

    members_exceptional_types = members_exploded.groupby(['member_id'])['exceptional_types'].apply(lambda x: ', '.join(sorted(filter(None, x)))).reset_index()

    exceptional_processed = pd.merge(members_exceptional_counts, members_exceptional_types, on='member_id')
    members_data = members_data[['uri', 'name', 'sort_name', 'title', 'gender', 'is_organization',
        'has_card', 'birth_year', 'death_year', 'membership_years', 'viaf_url',
        'wikipedia_url', 'nationalities', 'addresses', 'postal_codes',
        'arrondissements', 'coordinates', 'notes', 'updated', 'member_id',
        'borrow_count', 'subscription_count','original_uri']]
    updated_members_data = pd.merge(members_data, exceptional_processed, on=['member_id'], how='left')
    updated_members_data[['exceptional_counts']] = updated_members_data[['exceptional_counts']].fillna(0)
    return updated_members_data

def get_alphabetical_names(row):
    split_name = row['name'].split(' ')
    title = row['title']
    processed_name = [name for name in split_name if name != title]
    first_name_letter = processed_name[0][0].lower()
    last_name_letter = processed_name[-1][0].lower()
    if ('(' == last_name_letter) or ('[' == last_name_letter):
        last_name_letter = processed_name[-2][0].lower()
    if first_name_letter == 'é':
        first_name_letter = 'e'
    if (first_name_letter == '[') and ('unclear' in row['name']):
        first_name_letter = 'unclear'
    if (first_name_letter == '[') and ('unnamed' in row['name']):
        first_name_letter = 'unnamed'
        
    return [first_name_letter, last_name_letter]

def encode_data(members_data):
    le = preprocessing.LabelEncoder()
    feature_cols = ['title', 'gender', 'is_organization',
    'has_card', 'birth_year', 'death_year', 'membership_years', 'viaf_url',
    'wikipedia_url', 'nationalities', 'addresses', 'postal_codes',
    'arrondissements', 'coordinates',
    'borrow_count', 'subscription_count', 'exceptional_types',
    'exceptional_counts', 'member_id']
    categorical_cols = ['title', 'gender', 'is_organization',
    'has_card', 'birth_year', 'death_year', 'membership_years', 'nationalities', 'addresses', 'postal_codes',
    'arrondissements', 'coordinates','exceptional_types','member_id', 'first_letter', 'last_letter']
    numerical_cols = ['borrow_count', 'subscription_count',
    'exceptional_counts', 'member_id']
    members_categorical = members_data[categorical_cols].set_index('member_id')
    members_categorical.index.name = None
    members_transformed = members_categorical.apply(le.fit_transform)
    members_numerical = members_data[numerical_cols].set_index('member_id')
    members_numerical.index.name = None
    members_merged = pd.merge(members_transformed, members_numerical, left_index=True, right_index=True)
    return members_merged

def get_famous_fields(members_data):
    famous_members = members_data[['viaf_url', 'wikipedia_url', 'member_id']]
    famous_members['has_viaf'] = 0
    famous_members['has_wikipedia'] = 0
    famous_members.loc[famous_members.viaf_url.isna() == False, 'has_viaf'] = 1
    famous_members.loc[famous_members.wikipedia_url.isna() == False, 'has_wikipedia'] = 1
    famous_members_transformed = famous_members[['member_id', 'has_viaf', 'has_wikipedia']].set_index('member_id')
    return famous_members_transformed

def get_membership_years(members_data, members_merged, binning):
    member_years = members_data[['member_id', 'membership_years']]
    member_years.membership_years = member_years.membership_years.str.split(';')
    exploded_member_years = member_years.explode('membership_years')
    exploded_member_years['value'] = 1
    if binning:
        exploded_member_years.membership_years.fillna(0, inplace=True)
        exploded_member_years.membership_years = exploded_member_years.membership_years.astype(int)
        exploded_member_years['bins'] = pd.qcut(exploded_member_years.membership_years, q=3)
        exploded_member_years = exploded_member_years.groupby(['member_id', 'bins'])['value'].sum().reset_index(name='value')
        exploded_member_years.bins = exploded_member_years.bins.astype(str).str.replace('(', '').str.replace(']', '').str.replace(', ', '-').str.replace('-0.001', 'null_year') + '_year_bins'
        year_cols = exploded_member_years.bins.unique().tolist()
        pivoted_member_years = pd.pivot(exploded_member_years, index='member_id', columns='bins', values='value').fillna(0)
    else:
        exploded_member_years['value'] = 1
        exploded_member_years.membership_years.fillna('null', inplace=True)
        exploded_member_years.membership_years = exploded_member_years.membership_years.astype(str)
        exploded_member_years.membership_years = exploded_member_years.membership_years + '_year'
        pivoted_member_years = exploded_member_years.pivot(index='member_id', columns='membership_years', values='value')
        
        year_cols = exploded_member_years.membership_years.unique().tolist()
    pivoted_member_years.fillna(0, inplace=True)
    pivoted_member_years = pivoted_member_years.reset_index()
    pivoted_member_years = pivoted_member_years[year_cols + ['member_id']].set_index('member_id')
    members_merged = pd.merge(members_merged, pivoted_member_years, left_index=True, right_index=True)
    return members_merged, year_cols

def get_subscription_volumes(events_df, members_merged):
    subscriptions_df = events_df[ (events_df.subscription_duration_days.isna() == False)]
    subscriptions_df = subscriptions_df.groupby(['member_id', 'subscription_volumes']).size().reset_index(name='subscription_volumes_count')
    subscriptions_df.subscription_volumes = subscriptions_df.subscription_volumes.astype(int).astype(str) + '_volumes'

    volumes = subscriptions_df.subscription_volumes.unique().tolist()
    pivot_subscriptions = pd.pivot(subscriptions_df, index='member_id', columns='subscription_volumes', values='subscription_volumes_count').fillna(0)
    pivot_subscriptions = pivot_subscriptions.reset_index()
    pivot_subscriptions = pivot_subscriptions[volumes + ['member_id']]
    pivot_subscriptions = pivot_subscriptions.rename_axis(None, axis=1)

    members_merged['member_id'] = members_merged.index
    merged_totals = pd.merge(members_merged, pivot_subscriptions, on='member_id', how='left')
    merged_totals[volumes] = merged_totals[volumes].fillna(0)

    merged_totals = merged_totals.set_index('member_id').reset_index()

    merged_totals.index.name = None

    return merged_totals, volumes

def get_subscription_length(events_df, members_merged, binning):
    subscriptions_df = events_df[ (events_df.subscription_duration_days.isna() == False)]
    subscriptions_df = subscriptions_df[['member_id', 'subscription_duration_days']]
    subscriptions_df['value'] = 1
    if binning:
        subscriptions_df['bins'] = pd.qcut(subscriptions_df.subscription_duration_days, q=3)
        subscriptions_df = subscriptions_df.groupby(['member_id', 'bins'])['value'].sum().reset_index(name='value')
        subscriptions_df.bins = subscriptions_df.bins.astype(str).str.replace('(', '').str.replace(']', '').str.replace(', ', '-') + '_subscription_duration_bins'
        days = subscriptions_df.bins.unique().tolist()
        pivot_subscriptions = pd.pivot(subscriptions_df, index='member_id', columns='bins', values='value').fillna(0)
    else: 
        subscriptions_df = subscriptions_df.groupby(['member_id', 'subscription_duration_days'])['value'].sum().reset_index(name='value')
        subscriptions_df.subscription_duration_days = subscriptions_df.subscription_duration_days.astype(int).astype(str)
        subscriptions_df.subscription_duration_days = subscriptions_df.subscription_duration_days + '_days'

        days = subscriptions_df.subscription_duration_days.unique().tolist()

        pivot_subscriptions = pd.pivot(subscriptions_df, index='member_id', columns='subscription_duration_days', values='value').fillna(0)

    pivot_subscriptions = pivot_subscriptions.reset_index()
    pivot_subscriptions = pivot_subscriptions[days + ['member_id']]
    pivot_subscriptions = pivot_subscriptions.rename_axis(None, axis=1)

    members_merged['member_id'] = members_merged.index
    merged_totals = pd.merge(members_merged, pivot_subscriptions, on='member_id', how='left')
    merged_totals[days] = merged_totals[days].fillna(0)

    merged_totals = merged_totals.set_index('member_id').reset_index()

    merged_totals.index.name = None

    return merged_totals, days


def build_feature_data(members_data, events_df, bin_membership_years, bin_subscription_lengths):
    ## Get exploded and exceptional counts
    members_exploded = split_cols(members_data)
    members_exceptional = get_excpetional_counts(members_exploded, members_data)

    ## Set class for whether member was a borrower or not
    members_exceptional['is_borrower'] = members_exceptional.has_card
    members_exceptional['is_borrower'] = members_exceptional.is_borrower.astype(int)
    # members_exceptional['member_activity'] = 0
    # members_exceptional.loc[members_exceptional.exceptional_counts > 0, 'member_activity'] = 1
    # members_exceptional.loc[members_exceptional.borrow_count > 0, 'member_activity'] = 1

    ## Get alphabetical names
    members_exceptional['name_letters'] =members_exceptional.apply(get_alphabetical_names, axis=1)
    members_exceptional[['first_letter','last_letter']] = pd.DataFrame(members_exceptional.name_letters.tolist(), index= members_exceptional.index)
    members_exceptional.drop('name_letters', axis=1, inplace=True)
    members_exceptional.loc[members_exceptional.first_letter == '"', 'first_letter'] = 'friend'
    members_exceptional.loc[members_exceptional.last_letter == 'т', 'last_letter'] = 't'

    ## Encode categorical data
    members_merged = encode_data(members_exceptional)

    ## Get famous fields
    famous_members_transformed = get_famous_fields(members_exceptional)
    famous_members_merged = pd.merge(members_merged, famous_members_transformed, left_index=True, right_index=True)

    ## Get membership years
    membership_merged, year_cols = get_membership_years(members_exceptional, famous_members_merged, binning=bin_membership_years)

    ## Get subscription volumes
    members_processed, volumes = get_subscription_volumes(events_df, membership_merged)

    ## Get subscription lengths
    members_totals, days = get_subscription_length(events_df, membership_merged, binning=bin_subscription_lengths)

    cols = list(set(members_totals.columns.tolist()) - set(members_processed.columns.tolist()))
    members_final = pd.merge(members_processed, members_totals[cols], left_index=True, right_index=True)

    return members_final, members_exceptional, days, year_cols, volumes