# unknown borrowers data

Data in this folder generated as part of unknown borrowers research.

- exceptional members
- exceptional books
- exceptional events
- partial borrowers
- genre/subject data
- long borrow overrides: corrected information borrow events that are incorrectly entered with durations longer than a year in the 1.2 events dataset



## incorporating long borrow corrections

The borrow corrections are meant to be used with the 1.2 version of the dataset. They can be incorporated like this:

```python
events_df = pd.read_csv("SCoData_events_v1.2_2022-01.csv")
borrow_overrides = pd.read_csv("long_borrow_overrides.csv")

events_df = pd.read_csv("SCoData_events_v1.2_2022-01.csv")
borrow_overrides = pd.read_csv("long_borrow_overrides.csv")


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
```