# identify members with partial borrowing history
import os.path
from statistics import mean
import sys

import numpy as np
import pandas as pd

sys.path.append("..")
from dataset_generator.dataset import get_shxco_data, DATA_DIR

PARTIAL_BORROWERS_CSV = os.path.join(DATA_DIR, "partial_borrowers.csv")
PARTIAL_BORROWERS_COLLAPSED_CSV = os.path.join(
    DATA_DIR, "partial_borrowers_collapsed.csv"
)


def get_partial_borrowers():
    # convenience method to load partial borrowers csv and return as dataframe
    return pd.read_csv(PARTIAL_BORROWERS_CSV)


def identify_partial_borrowers():
    members_df, books_df, events_df = get_shxco_data()
    date_events = events_df.copy()
    date_events["start_datetime"] = pd.to_datetime(
        date_events.start_date, format="%Y-%m-%d", errors="coerce"
    )
    date_events["end_datetime"] = pd.to_datetime(
        date_events.end_date, format="%Y-%m-%d", errors="coerce"
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
        # sort subscription events by start date so we can collapse sequential periods
        member_subs = subscription_events[
            subscription_events.member_id == member_id
        ].sort_values("start_date")
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
                # print('\n%s' % member_id)
                # print(f"{sub.start_date} - {sub.end_date} ({sub.event_type}, {sub.subscription_volumes} volumes)")
                partial_borrowers.append(
                    {
                        "member_id": member_id,
                        "subscription_start": sub.start_date,
                        "subscription_end": sub.end_date,
                        "subscription_type": sub.event_type,
                        "subscription_volumes": sub.subscription_volumes,
                        "known_borrows": len(member_book_events.index),
                    }
                )

    # load into dataframe and save as csv for use elsewhere
    partial_borrows = pd.DataFrame(data=partial_borrowers)
    partial_borrows.to_csv(PARTIAL_BORROWERS_CSV, index=False)

    # now collapse sequential-ish subscription periods for each member
    collapse_partial_borrowers(partial_borrows)


def collapse_partial_borrowers(partial_borrows):
    # collapse sequential periods into single time periods
    collapsed_ranges = []
    # convert to datetime for comparison
    partial_borrows["start_date"] = pd.to_datetime(
        partial_borrows.subscription_start, format="%Y-%m-%d", errors="coerce"
    )
    partial_borrows["end_date"] = pd.to_datetime(
        partial_borrows.subscription_end, format="%Y-%m-%d", errors="coerce"
    )

    # allowed gap between sequential periods
    # ?? what amount of time is reasonable here?
    allowed_gap = pd.Timedelta(days=90)

    for member_id in partial_borrows.member_id.unique():
        range_start = None
        range_end = None

        # get the previously identified subscriptions with no borrowing
        member_subs = partial_borrows[partial_borrows.member_id == member_id]
        for sub in member_subs.itertuples():

            # if we have an existing range, check if this one overlaps,
            # or start date follows within the allowed gap
            if range_start and (
                sub.start_date <= range_end or sub.start_date - range_end <= allowed_gap
            ):
                # extend the current range to the later of the two end dates
                # (overlapping supplement could end before subscription)
                range_end = max(range_end, sub.end_date)
                # store the size of the gap between the ranges, if any
                internal_gaps.append(max(0, (sub.start_date - range_end).days))

            # if no overlap, save the current range (if any) and start a new one
            else:
                # if there is an existing range, save it
                if range_start:
                    collapsed_ranges.append(
                        {
                            "member_id": member_id,
                            # TODO: format dates for export
                            "subscription_start": range_start,
                            "subscription_end": range_end,
                            # sequence of subscription events: [subscription, renew, supplement]
                            "subscription_events": ";".join(events),
                            "subscription_volumes": mean(volumes),
                            "subscription_days": (range_end - range_start).days,
                            "internal_gaps": ";".join(
                                [str(gap) for gap in internal_gaps]
                            ),
                            # preserve members' total known borrows
                            "known_borrows": sub.known_borrows,
                        }
                    )

                # start a new range based on the current subscription
                range_start = sub.start_date
                range_end = sub.end_date
                events = []
                volumes = []
                internal_gaps = []

            # keep track of subscription events and volumes
            events.append(sub.subscription_type)
            volumes.append(sub.subscription_volumes)

        # save the last range
        if range_start and range_end:
            collapsed_ranges.append(
                {
                    "member_id": member_id,
                    "subscription_start": range_start,
                    "subscription_end": range_end,
                    "subscription_events": ";".join(events),
                    "subscription_volumes": mean(volumes),
                    "subscription_days": (range_end - range_start).days,
                    "internal_gaps": ";".join([str(gap) for gap in internal_gaps]),
                    "known_borrows": sub.known_borrows,
                }
            )

    # convert to dataframe and save as csv for use elsewhere
    pd.DataFrame(data=collapsed_ranges).to_csv(
        PARTIAL_BORROWERS_COLLAPSED_CSV, index=False
    )


if __name__ == "__main__":
    identify_partial_borrowers()
