import os.path

import numpy as np
import pandas as pd

import sys

sys.path.append("..")
from dataset_generator.dataset import get_shxco_data, get_data, get_model
from dataset_generator.identify_partial_borrowers import get_partial_borrowers


def partial_borrowing():
    members_df, books_df, events_df = get_shxco_data()

    print("Loading partial borrower information...")
    partial_borrowers = get_partial_borrowers()
    print("  %d partial borrowers" % partial_borrowers.shape[0])

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
        # identify books with events that year.
        # since we have filtered out some events (orgs, shared accounts),
        # find circulating books from the full books dataset
        circulating_books = books_df[
            books_df.circulation_years.str.contains(bookless_sub.subscription_start[:4], na=False)
        ]
        # get a *unique* list of item uris
        item_uris = circulating_books.id.unique()
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
