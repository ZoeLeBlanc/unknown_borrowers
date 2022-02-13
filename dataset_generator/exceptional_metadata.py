
import pandas as pd
pd.options.mode.chained_assignment = None
from datetime import datetime
from tqdm import tqdm
import sys
sys.path.append("..")
from dataset_generator.dataset import get_shxco_data

def get_longborrow_overides(events_df):
    borrow_overrides = pd.read_csv("../dataset_generator/data/long_borrow_overrides.csv")
    borrow_overrides["member_id"] = borrow_overrides.member_uris.apply(
        lambda x: x.split("/")[-2]
    )
    borrow_overrides["item_uri"] = borrow_overrides.item_uri.apply(
        lambda x: x.split("/")[-2] if pd.notna(x) else None
    )
    for borrow in borrow_overrides.itertuples():
        member_item_borrows = events_df[
            (events_df.event_type == "Borrow")
            & (events_df.member_uris == borrow.member_uris)
            & (events_df.item_uri == borrow.item_uri)
        ]
        if borrow.match_date == "start_date":
            # get the *index* of the row to update
            update_index = member_item_borrows.index[
                member_item_borrows.start_date == borrow.start_date
            ]
        elif borrow.match_date == "end_date":
            update_index = member_item_borrows.index[
                member_item_borrows.end_date == borrow.end_date
            ]

        # update with correct dates & borrow duration
        events_df.at[update_index, "start_date"] = borrow.start_date
        events_df.at[update_index, "end_date"] = borrow.end_date
        events_df.at[
            update_index, "borrow_duration_days"
        ] = borrow.borrow_duration_days
    return events_df

def load_data():
    """Load in initial datasets"""
    print("loading datasets")
    members_df, books_df, events_df = get_shxco_data()

    events_df = get_longborrow_overides(events_df)
    events_df['index_col'] = events_df.index
    # calculate borrow count for each member
    grouped_borrows = events_df[events_df.event_type == 'Borrow'].groupby(['member_id']).size().reset_index(name='borrow_count')
    members_df = members_df.merge(grouped_borrows, on='member_id', how='outer')
    members_df.borrow_count = members_df.borrow_count.fillna(0)
    # calculate subscription count for each member
    grouped_subscriptions = events_df[events_df.event_type.isin(['Subscription', 'Renewal', 'Supplement'])].groupby(['member_id']).size().reset_index(name='subscription_count')
    members_df = members_df.merge(grouped_subscriptions, on='member_id', how='outer')
    members_df.subscription_count = members_df.subscription_count.fillna(0)
    
    return members_df, books_df, events_df

def sunday_shoppers(events_df):
    """# members with in-shop events on sundays"""
    print('calculating sunday shoppers')
    # generate dataframe with all dates that would likely have brought a person into the store
    instore_events = events_df.copy()
    # limit to fields needed
    instore_events = instore_events[['event_type', 'start_date', 'end_date', 'member_uris','member_names', 'member_id', 'subscription_purchase_date', 'index_col', 'item_uri']]
    # for subscriptions (all types), purchase date is actual in-store date and may be different from subscription start
    # copy subscription purchase date (if set) or start date to date
    instore_events_start = instore_events.copy()
    instore_events_start['date'] = instore_events_start.apply(
        lambda x: x.subscription_purchase_date or x.start_date, axis=1)

    # only end dates for borrow events are an in-store event
    instore_events_end = instore_events[instore_events.event_type == 'Borrow'].rename(
        columns={'end_date': 'date'})
    # set event type to borrow end, in case we need to distinguish
    instore_events_end['event_type'] = 'Borrow end'
    instore_dates = instore_events_start.append(instore_events_end)
    # drop the columns we don't need anymore
    instore_dates = instore_dates.drop(
        columns=['start_date', 'end_date', 'subscription_purchase_date'])
    # limit to events with fully known dates so we can calculate day of week
    # - drop all unset values
    instore_dates = instore_dates.dropna()
    # - drop all partial dates
    instore_dates = instore_dates[instore_dates.date.str.len() == 10]
    # parse to datetime object
    instore_dates['dt'] = pd.to_datetime(instore_dates.date)
    # calculate day of week for each date
    instore_dates['weekday'] = instore_dates.dt.apply(lambda x: x.weekday())
    # filter to Sundays
    sunday_shoppers = instore_dates[instore_dates.weekday == 6]
    return sunday_shoppers


def post1942_events(events_df):
    """# members with activity after the shop officially closed"""
    print('calculating post-1942 events')
    # members with dates after official end date

    logbook_end = datetime(1941, 12, 12)
    # parse to datetime object
    events_df['dt'] = pd.to_datetime(events_df.start_date, errors='coerce')

    postshop_events = events_df[events_df.dt > logbook_end]
    return postshop_events


def missing_books(events_df):
    """# members who didn't return books

    ## missing books

    no return date, marked on the card as missing by Beach or her assistants
    """
    print('calculating missing books')
    missing_books = events_df[events_df.borrow_status == 'Missing']

    return missing_books


def unknown_borrow_status(events_df):
    """## unknown borrow end status

    we don't know what happened to the book, but no return date is documented
    """
    print('calculating unknown borrow status')
    unknown_borrows = events_df[events_df.borrow_status == 'Unknown']
    return unknown_borrows


def get_member_usage(row, subscription_events, borrow_events):
   # given a row with member uri and date,
   # calculate subscription volume & books out
    day = row.day
    # subset subscription and borrow events to this member
    # match uri on contains since we haven't split out multi-member accounts
    member_subs = subscription_events[subscription_events.member_uris.str.contains(
        row.uri)]
    member_books = borrow_events[borrow_events.member_uris.str.contains(
        row.uri)]
    vols = member_subs[(member_subs.start_datetime <= day) & (
        day < member_subs.end_datetime)].subscription_volumes.sum()
    books = len(member_books[(member_books.start_datetime <= day) & (
        member_books.end_datetime > day)])
    return vols, books


def overborrows(members_df, events_df):
    """# members who exceeded their subscription allowance"""
    print('calculating overborrows')
    # get list of members with cards, since only they will have borrow events
    card_members = members_df[members_df.has_card]

    """### separate subscription events and get date range"""


    # subset event data for only those with complete start and end dates
    date_events = events_df[(events_df.start_date.str.len() > 9) & (
        events_df.end_date.str.len() > 9)].copy()
    # turn start/end dates into datetimes
    date_events['start_datetime'] = pd.to_datetime(
        date_events.start_date, format='%Y-%m-%d', errors='ignore')
    date_events['end_datetime'] = pd.to_datetime(
        date_events.end_date, format='%Y-%m-%d', errors='ignore')

    # identify subscription events
    subscription_events = date_events[date_events.event_type.isin(
        ['Subscription', 'Renewal', 'Supplement'])]
    # get the earliest subscription start
    subs_start_date = subscription_events.start_datetime.min()
    # get the latest subscription end
    subs_end_date = subscription_events.end_datetime.max()

    """### separate out borrow events"""

    # borrow events
    borrow_events = date_events[date_events.event_type == 'Borrow']

    # generate list of days for the duration of the bookshop
    bookshop_dates = pd.date_range(start=subs_start_date, end=subs_end_date)

    """### calculate member volumes and books out at each borrow start"""

    # generate a dataframe with volume count and books out for each member with a card
    # on each day they checked out a book (borrow event start)

    all_member_usage = pd.DataFrame()

    for member_uri in tqdm(card_members.uri):
        # get borrow events for this member
        member_borrow = borrow_events[borrow_events.member_uris.str.contains(
            member_uri)]
        # get unique list of borrow start dates
        # -- instead of checking every day, just check each day with any new checkout:
        #   does the current checkouts put them over their subscription limit?
        member_borrow_dates = member_borrow.start_datetime.unique()

        # create dataframe for this member
        member_usage = pd.DataFrame(
            data={'day': member_borrow_dates, 'uri': member_uri})

        member_usage[["subscription_volumes", "books_out"]] = member_usage.apply(
            get_member_usage, axis=1, result_type='expand', args=(subscription_events, borrow_events))
        # append usage information for this member to the combined dataframe
        all_member_usage = all_member_usage.append(member_usage)

    # drop entries where subscription volumes are zero
    # (i.e., borrowing outside of a documented subscription)
    # since they are not part of the current analysis
    # [... but the zeroes may be interesting for other purposes]

    member_usage_withsub = all_member_usage[all_member_usage.subscription_volumes != 0]

    # calculate number of books out over allowed subsription volumes
    member_usage_withsub['excess_books_out'] = member_usage_withsub.apply(
        lambda x: max(x.books_out - x.subscription_volumes, 0), axis=1)
    # identify days when member had more books out than subscription allowed
    member_usage_withsub[member_usage_withsub.excess_books_out > 0]

    dfs = []
    for index, row in member_usage_withsub[member_usage_withsub.excess_books_out > 0].iterrows():
        df = borrow_events[(borrow_events.member_uris.str.contains(
            row.uri)) & (borrow_events.start_datetime == row.day)]
        df['excess_books_out'] = row.excess_books_out
        df['subscription_volumes'] = row.subscription_volumes
        df['books_out'] = row.books_out
        dfs.append(df)

    overborrows = pd.concat(dfs)
    return overborrows


def longborrows(events_df):
    """# long-term borrowers

    which members kept books out a year or longer?
    """
    print('calculating long borrows')
    longborrows = events_df[events_df.borrow_duration_days > 365]
    return longborrows


def group_types(row):
    """Group members or books with their types and counts"""
    row['exceptional_types'] = ','.join(row.type.values.tolist())
    row['exceptional_counts'] = ','.join(
        row.counts.astype(str).values.tolist())
    # row['exceptional_types'] = row.type.values.tolist()
    # row['exceptional_counts'] = row.counts.astype(str).values.tolist()
    return row

def calculate_exceptional_categories(write_to_csv):
    """Calculate the exceptional categories for members, books, and events. You can either write to csv with the write_to_csv or by default import this function into another script and get the updated dataframes as a result. """
    members_df, books_df, events_df = load_data()
    event_sets = {
        'sunday_shopers': sunday_shoppers(events_df),
        'post1942_events': post1942_events(events_df),
        'missing_events': missing_books(events_df),
        'unknown_borrows': unknown_borrow_status(events_df),
        'overborrows': overborrows(members_df, events_df),
        'longterm_borrows': longborrows(events_df),
    }
    event_dfs = []
    for label, events in event_sets.items():
        events['type'] = label
        event_dfs.append(events)

    concat_events = pd.concat(event_dfs)
    subset_events = concat_events[['index_col', 'item_uri', 'member_uris', 'start_date', 'type', 'books_out','excess_books_out', 'subscription_volumes', 'event_type', 'borrow_duration_days', 'subscription_duration_days', 'member_id', 'second_member_uri']]

    # subset_grouped = subset_events.groupby(['index_col'])['type'].transform(
    #     lambda x: ','.join(x)).reset_index(name='exceptional_types')

    subset_grouped = subset_events.groupby(['index_col'])['type'].apply(list).reset_index(name='exceptional_types')

    subset_grouped = subset_grouped.rename(columns={'index': 'index_col'})
    subset_merged = pd.merge(subset_events, subset_grouped, on=['index_col'], how='outer')

    events_copy = events_df.copy()
    exceptional_events = pd.merge(events_copy, subset_merged[['index_col', 'exceptional_types', 'books_out', 'excess_books_out']], on=['index_col'], how='left')
    exceptional_events = exceptional_events.drop_duplicates(subset='index_col')
    exceptional_events.exceptional_types.fillna('', inplace=True)

    """## Exceptional Books"""
    book_counts = subset_events[subset_events.item_uri.isna() == False][[
        'item_uri', 'type']].groupby(['item_uri', 'type']).size().reset_index(name='counts')
    book_counts = book_counts.rename(
        columns={'item_uri': 'id'})
    # book_counts.to_csv('book_counts.csv')
    # grouped_books = book_counts.groupby('id').apply(group_types)
    grouped_books = book_counts.groupby('id').apply(lambda x: [list(x['type']), list(x['counts'])]).apply(pd.Series).reset_index()
    # grouped_books.to_csv('grouped_books.csv')
    grouped_books.columns = ['id', 'exceptional_types', 'exceptional_counts']
    # deduped_books = grouped_books[['id', 'exceptional_types', 'exceptional_counts']].drop_duplicates()
    exceptional_books = pd.merge(books_df, grouped_books, on=['id'], how='outer')
    exceptional_books.exceptional_types.fillna('', inplace=True)
    exceptional_books.exceptional_counts.fillna(0, inplace=True)
    exceptional_books['original_uri'] = exceptional_books.uri
    exceptional_books['uri'] = exceptional_books.id

    """## Exceptional Members"""

    subset_members = subset_events.copy()
    grouped_members = subset_members.groupby(['member_id', 'type']).size().reset_index(name='counts')

    # grouped_dupes = grouped_members.groupby('member_id').apply(group_types)
    grouped_dupes = grouped_members.groupby('member_id').apply(lambda x: [list(x['type']), list(x['counts'])]).apply(pd.Series).reset_index()
    grouped_dupes.columns = ['member_id', 'exceptional_types', 'exceptional_counts']
    # grouped_deduped = grouped_dupes[[
    #     'member_id', 'exceptional_types', 'exceptional_counts']].drop_duplicates()
    exceptional_members = pd.merge(members_df, grouped_dupes, on=['member_id'], how='outer')
    exceptional_members.exceptional_types.fillna('', inplace=True)
    exceptional_members.exceptional_counts.fillna(0, inplace=True)
    exceptional_members['original_uri'] = exceptional_members.uri
    exceptional_members['uri'] = exceptional_members.member_id

    if write_to_csv:
        exceptional_events.to_csv('./data/SCoData_events_v1.2_2022-01_exceptional.csv', index=False)
        exceptional_books.to_csv('./data/SCoData_books_v1.2_2022-01_exceptional.csv', index=False)
        exceptional_members.to_csv('./data/SCoData_members_v1.2_2022-01_exceptional.csv', index=False)
    else:
        return (exceptional_members, exceptional_books, exceptional_events)


def get_shxco_exceptional_data():

    exceptional_events = pd.read_csv('../dataset_generator/data/SCoData_events_v1.2_2022-01_exceptional.csv')
    exceptional_books = pd.read_csv('../dataset_generator/data/SCoData_books_v1.2_2022-01_exceptional.csv')
    exceptional_members = pd.read_csv('../dataset_generator/data/SCoData_members_v1.2_2022-01_exceptional.csv')
    return (exceptional_members, exceptional_books, exceptional_events)


if __name__ == "__main__":
    # load_data()
    calculate_exceptional_categories(True)
    # get_shxco_exceptional_data()
