import os.path

import numpy as np
import pandas as pd

from dataset import get_shxco_data, get_data, get_model


# identify members with partial borrowing history
# - adapt from colab notebook
# - get dates for the subscriptions without documented borrowing activity
#    (chunk the time periods? collapse chunks?)

# write a method to get a list of books that
# were known to be circulating during a particular time period (year? months?)


def identify_partial_borrowers(date_events):
    # if the file already exists, just load it and return
    partial_borrowers_csv = "data/partial_borrowers.csv"
    if os.path.isfile(partial_borrowers_csv):
        return pd.read_csv(partial_borrowers_csv)

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
                (member_book_events.end_datetime >= sub.start_datetime)
                | (member_book_events.start_datetime >= sub.end_datetime)
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

    df = pd.DataFrame(data=partial_borrowers)
    # save this as as csv so we don't have to recalculate every time
    df.to_csv(partial_borrowers_csv, index=False)

    return df

    #   for each subscription period
    #        get list of books that were circulating
    #        # run model.predict for that user over those items

    # use the scores to get the most likely items
    # scores = model.predict(user_id, np.arange(n_items))
    # — how accurate/specific are scores? do they tell us the confidence of the likelihood?

    # can we use Kevin's estimates to determine how many borrowing events they
    # would likely have had during that time
    # (and/or use that particular member's activity if different?)

    # output the top books we predict they most likely borrowed (include scores/confidence? are they meaningful)
    # NOTE: may be able to use predict rank method here, depending on # of interactions
    # (but predict may be easier anyway)


def partial_borrowing():
    members_df, books_df, events_df = get_shxco_data()
    date_events = events_df.copy()
    date_events["start_datetime"] = pd.to_datetime(
        date_events.start_date, format="%Y-%m-%d", errors="ignore"
    )
    date_events["end_datetime"] = pd.to_datetime(
        date_events.end_date, format="%Y-%m-%d", errors="ignore"
    )

    print("identifying partial borrowers...")
    partial_borrowers = identify_partial_borrowers(date_events)
    print("found %d partial borrowers" % partial_borrowers.shape[0])

    # for each partial borrower subscription period
    #    get list of books that were circulating
    # run model.predict for that user over those items
    book_events = events_df[events_df.item_uri.notna()].copy()
    book_events["year"] = events_df.start_date.str.extract(r"^(\d{4})")

    dataset = get_data()
    model = get_model()

    for bookless_sub in partial_borrowers.itertuples():
        user_id = dataset["member_dataset_id"][bookless_sub.member_id]
        print(
            "\n%s : %s — %s (%d known borrow%s)"
            % (
                bookless_sub.member_id,
                bookless_sub.subscription_start,
                bookless_sub.subscription_end,
                bookless_sub.known_borrows,
                "" if bookless_sub.known_borrows == 1 else "s",
            )
        )
        # are any of these multi year? just use start year for now
        # identify books with events that year
        circulating_books = book_events[
            book_events.year == bookless_sub.subscription_start[:4]
        ]
        # get a *unique* list of item uris
        item_uris = circulating_books.item_uri.unique()
        # get a list of dataset item ids
        item_ids = [dataset["item_dataset_id"][item_uri] for item_uri in item_uris]

        # predict the interaction between this user and these items
        scores = model.predict(user_id, item_ids)

        # get the highest ranked items & their scores
        score_indices = np.argsort(-scores)[:5]
        top_items = [item_uris[i] for i in score_indices]
        top_scores = [scores[i] for i in score_indices]
        # output item id & predicted score
        for i, x in enumerate(top_items[:3]):
            print("\t%s\t%s" % (x.ljust(35), top_scores[i]))


if __name__ == "__main__":
    partial_borrowing()
