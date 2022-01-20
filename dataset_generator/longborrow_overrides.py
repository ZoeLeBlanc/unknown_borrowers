import argparse

import pandas as pd


def long_borrow_overrides(corrected_events):
    # load v1.2 of events dataset
    events_df = pd.read_csv("source_data/SCoData_events_v1.2_2022-01.csv")
    # load the newer version with the corrections we want
    corrected_events_df = pd.read_csv(corrected_events)
    # limit just to borrows
    corrected_borrows = corrected_events_df[corrected_events_df.event_type == "Borrow"]

    # identify the long borrows that were flagged for correction
    yearplus_borrows = events_df[events_df.borrow_duration_days > 365]
    print("Found %d borrows longer than a year" % yearplus_borrows.shape[0])

    # make a list to correct override rows
    overrides = []

    # iterate through the long borrows and check for corrections
    for borrow in yearplus_borrows.itertuples():
        # check if the borrow has been corrected
        # because we don't have event ids, we need to find based on matching
        # (already filtered to borrows only)
        # filter by member/item first
        member_item_events = corrected_borrows[
            (corrected_borrows.member_uris == borrow.member_uris)
            & (corrected_borrows.item_uri == borrow.item_uri)
        ]
        # first check for start date match
        corrected_borrow = member_item_events[
            member_item_events.start_date == borrow.start_date
        ]
        match_on = "start_date"
        if not corrected_borrow.shape[0]:
            # if no match by start date, try by end date
            corrected_borrow = member_item_events[
                member_item_events.end_date == borrow.end_date
            ]
            match_on = "end_date"

        # get first row (should be one and only  one)
        correction = corrected_borrow.iloc[0]
        # if the borrow duration changed, add to the list of overrides
        if borrow.borrow_duration_days != correction.borrow_duration_days:
            corrected_borrow["match_date"] = match_on
            overrides.append(corrected_borrow)

    # combine the overrides into a new dataframe
    override_df = pd.concat(overrides)
    print("Saving %d long borrow overrides" % override_df.shape[0])
    # limit to just the columns needed for identifying & overriding
    override_df = override_df[
        [
            "event_type",
            "member_uris",
            "item_uri",
            "start_date",
            "end_date",
            "borrow_duration_days",
            "match_date",
        ]
    ]
    pd.DataFrame(override_df).to_csv("data/long_borrow_overrides.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "events",
        metavar="CSVFILE",
        help="path to more recent events export with updated borrows",
    )
    args = parser.parse_args()
    long_borrow_overrides(args.events)
