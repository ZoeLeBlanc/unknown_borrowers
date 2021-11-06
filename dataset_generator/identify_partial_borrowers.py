# identify members with partial borrowing history
import os.path
import sys

import numpy as np
import pandas as pd

sys.path.append("..")
from dataset_generator.dataset import get_shxco_data, DATA_DIR

PARTIAL_BORROWERS_CSV = os.path.join(DATA_DIR, "partial_borrowers.csv")

def get_partial_borrowers():
    # convenience method to load partial borrowers csv and return as dataframe
    return pd.read_csv(PARTIAL_BORROWERS_CSV)


def identify_partial_borrowers():
    members_df, books_df, events_df = get_shxco_data()
    date_events = events_df.copy()
    date_events["start_datetime"] = pd.to_datetime(
        date_events.start_date, format="%Y-%m-%d", errors="ignore"
    )
    date_events["end_datetime"] = pd.to_datetime(
        date_events.end_date, format="%Y-%m-%d", errors="ignore"
    )

    # filter to subscription events with known start and end date
    subscription_events = date_events[
        date_events.event_type.isin(["Subscription", "Renewal", "Supplement"])
        & date_events.start_datetime.notna()
        & date_events.end_datetime.notna()
    ]

    # get all book events (anything with an item uri, ignore event type)
    # [strictly speaking should we restrict to borrows?]
    book_events = date_events[date_events.item_uri.notna()]

    partial_borrowers = []

    # look over subscriptions for each member with book events
    for member_id in book_events.member_id.unique():
        # filter to all subscription and book events for this member
        member_subs = subscription_events[subscription_events.member_id == member_id]
        member_book_events = book_events[book_events.member_id == member_id]

        # check each subscription for any overlapping book events
        for sub in member_subs.itertuples():
            # NOTE: ignoring unknown end dates
            # look for book events that overlap with the subscription dates
            sub_book_events = member_book_events[
                (sub.start_datetime <= member_book_events.end_datetime)
                & (sub.end_datetime >= member_book_events.start_datetime)
            ]

            # if there are no book events within this subscription,
            # add it to the list of partial borrower dates
            if sub_book_events.empty:
                partial_borrowers.append(
                    {
                        "member_id": member_id,
                        "subscription_start": sub.start_date,
                        "subscription_end": sub.end_date,
                        "known_borrows": len(member_book_events.index),
                    }
                )

    # load into dataframe and save as csv for use elsewhere
    pd.DataFrame(data=partial_borrowers).to_csv(PARTIAL_BORROWERS_CSV, index=False)


if __name__ == "__main__":
    identify_partial_borrowers()