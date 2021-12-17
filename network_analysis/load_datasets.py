
import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("..")
from dataset_generator.exceptional_metadata import get_shxco_exceptional_data

def format_events_data(events_df):
    """Format the date fields in the events dataframe"""
    events_df['start_datetime'] = pd.to_datetime(events_df.start_date, format='%Y-%m-%d', errors='coerce')
    events_df['end_datetime'] = pd.to_datetime(events_df.end_date, format='%Y-%m-%d', errors='coerce')
    events_df['subscription_purchase_datetime'] = pd.to_datetime(events_df.subscription_purchase_date, format='%Y-%m-%d', errors='coerce')
    events_df['index'] = events_df.index
    events_df = events_df.reset_index(drop=True)
    events_df['year'] = events_df.start_datetime.dt.year
    events_df['month'] = events_df.start_datetime.dt.month
    return events_df

def format_subscription_events(events_df):
    """Format and subset subscription events"""
    subscription_events = events_df.copy()
    subscription_events = subscription_events[subscription_events.subscription_purchase_datetime.isna() ==False]
    subset_subscription_events = subscription_events[['subscription_purchase_datetime','start_datetime', 'event_type', 'member_id', 'end_datetime', 'index']]
    # Using Rebecca's approach to calculating subscriptions
    subscription_events = subscription_events[subscription_events.event_type.isin(['Subscription', 'Renewal', 'Supplement'])]
    subset_subscription_events = subscription_events[['subscription_purchase_datetime','start_datetime', 'event_type', 'member_id', 'end_datetime', 'index']]
    return subset_subscription_events

def format_borrow_events(events_df, get_subscription):
    """Format and subset borrow events"""
    borrow_events = events_df[(events_df.event_type == 'Borrow') & (events_df.item_uri.notna())]
    if get_subscription:
        borrow_events = events_df[(events_df.event_type == 'Borrow') & (events_df.start_date.str.len() > 9) & (events_df.end_date.str.len() > 9)]
        borrow_events.year = borrow_events.year.astype(int)
        borrow_events.month = borrow_events.month.astype(int)
    return borrow_events

def clean_subscriptions(row):
    """Check if any subscription active during borrow"""
    if row.subscription_active.any():
        row.subscription_active = True
    else:
        row.subscription_active = False
    return row

def check_for_active_subscriptions(borrow_events, subset_subscription_events):
    """Merge borrow and subscription events to check if subscriptions where active when books where checked out"""
    merged_borrows_subs = pd.merge(borrow_events, subset_subscription_events, on=['member_id'], how='left')
    merged_borrows_subs['subscription_active'] = np.where(((merged_borrows_subs.subscription_purchase_datetime_y <= merged_borrows_subs.start_datetime_x )|(merged_borrows_subs.end_datetime_x <= merged_borrows_subs.end_datetime_y )), True, False)
    # Clean out the merged dataframe to only include rows for borrow events
    merged_borrows_subs = merged_borrows_subs[['start_date', 'member_id', 'subscription_active', 'item_uri', 'index_x']]
    merged_borrows_subs = merged_borrows_subs[merged_borrows_subs.item_uri.isna() == False]
    merged_borrows_subs = merged_borrows_subs[merged_borrows_subs.duplicated() == False]
    # Merge updated borrow events into original borrow events
    merged_borrows_subs = merged_borrows_subs.rename(columns={'index_x':'index'})
    updated_borrow_events = pd.merge(borrow_events, merged_borrows_subs, on=['index','start_date', 'member_id', 'item_uri'], how='inner')
    updated_borrow_events = updated_borrow_events.groupby(['member_id', 'item_uri', 'start_date']).apply(clean_subscriptions)
    return updated_borrow_events

def get_updated_shxco_data(get_subscription):
    """Get shxco data and update accordingly"""
    members_df, books_df, events_df = get_shxco_exceptional_data()
    events_df = format_events_data(events_df)
    subset_subscription_events = format_subscription_events(events_df)
    borrow_events = format_borrow_events(events_df, get_subscription)
    if get_subscription:
        updated_borrow_events = check_for_active_subscriptions(
            borrow_events, subset_subscription_events)
        updated_borrow_events = updated_borrow_events.drop_duplicates(
            subset='index')
    else:
        updated_borrow_events = borrow_events
    
    
    return members_df, books_df, updated_borrow_events, events_df





