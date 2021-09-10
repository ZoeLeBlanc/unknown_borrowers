import numpy as np
import pandas as pd
from lightfm.data import Dataset
from lightfm import LightFM


csv_urls = {
    # online version
    # 'members': 'https://dataspace.princeton.edu/bitstream/88435/dsp01b5644v608/2/SCoData_members_v1.1_2021-01.csv',
    # 'books': 'https://dataspace.princeton.edu/bitstream/88435/dsp016d570067j/2/SCoData_books_v1.1_2021-01.csv',
    # 'events': 'https://dataspace.princeton.edu/bitstream/88435/dsp012n49t475g/2/SCoData_events_v1.1_2021-01.csv'
    # local downloaded copy
    "members": "data/SCoData_members_v1.1_2021-01.csv",
    "books": "data/SCoData_books_v1.1_2021-01.csv",
    "events": "data/SCoData_events_v1.1_2021-01.csv",
}


def get_item_features(item):
    # get features for an individual item
    return [
        "pubyear_%s" % (item.year if pd.notna(item.year) else "unknown"),
        "author_%s" % (item.author if pd.notna(item.author) else "unknown"),
        "is_multivol"
        if pd.notna(item.volumes_issues) and item.volumes_issues
        else "non_multivol",
    ]


def get_user_features(member):
    # get features for an individual item
    features = [
        "gender_%s" % (member.gender.lower() if pd.notna(member.gender) else "unknown"),
        # "organization" if member.is_organization else "person",     # probably not meaningful
    ]
    if pd.notna(member.arrondissements):
        features.extend(
            ["arrondissement_%s" % i for i in member.arrondissements.split(";") if i]
        )
    return features


def get_shxco_data():
    # load S&co datasets and return as pandas dataframes
    # returns members, books, events

    members_df = pd.read_csv(csv_urls["members"])
    books_df = pd.read_csv(csv_urls["books"])
    events_df = pd.read_csv(csv_urls["events"])

    # datasets use URIs for identifiers; generate short-form versions
    # across all datasets for easier display/use

    # - generate short id from book uri
    books_df["id"] = books_df.uri.apply(lambda x: x.split("/")[-2])
    # - generate short form member id
    members_df["member_id"] = members_df.uri.apply(lambda x: x.split("/")[-2])

    # split multiple members for shared accounts in events
    events_df[
        ["first_member_uri", "second_member_uri"]
    ] = events_df.member_uris.str.split(";", expand=True)
    # working with the first member for now...
    # generate short ids equivalent to those in member and book dfn
    events_df["member_id"] = events_df.first_member_uri.apply(
        lambda x: x.split("/")[-2]
    )
    events_df["item_uri"] = events_df.item_uri.apply(
        lambda x: x.split("/")[-2] if pd.notna(x) else None
    )

    return (members_df, books_df, events_df)



def get_data():
    # load members, books, events as csv
    members_df, books_df, events_df = get_shxco_data()

    # get list of user ids and book ids for dataset

    # get all member-book interactions from events
    # shorten URIs for readability in output
    interactions_df = events_df[events_df.item_uri.notna()].copy()

    # reduce to minimum user/item interaction fields and drop dupes
    unique_interactions_df = interactions_df[
        ["member_id", "item_uri"]
    ].drop_duplicates()

    # generate unique list of member ids with book interactions
    book_members = interactions_df.member_id.unique()
    # include all members, so we can make recommendations for members without documented interactions
    all_members = members_df.member_id.unique()

    dataset = Dataset()
    # pass list of user ids and list of book ids
    # provide list of unique categorical features

    print("fitting dataset...")
    # list of features to be used when defining the dataset
    item_feature_list = (
        [
            "pubyear_%s" % (year if pd.notna(year) else "unknown")
            for year in books_df.year.unique()
        ]
        + [
            # TODO: split out multi-author
            "author_%s" % (author if pd.notna(author) else "unknown")
            for author in books_df.author.unique()
        ]
        + ["is_multivol", "non_multivol"]
    )
    # to add: subject/genre/ from oclc db export and/or wikidata reconcile work
    # anything added here must be implemented in get_item_features

    user_feature_list = (
        [
            "gender_%s" % (gender.lower() if pd.notna(gender) else "unknown")
            for gender in members_df.gender.unique()
        ]
        + ["person", "organization"]
        + [
            # arrondissements are 1-20
            "arrondissement_%d" % i
            for i in range(1, 21)
        ]
    )

    # consider adding:
    # - birth year  / birth decade
    # - nationality (could be multiple)
    # anything added here must be implemented in get_user_features

    dataset.fit(
        all_members,
        books_df.id,
        item_features=item_feature_list,
        user_features=user_feature_list,
    )

    print("building interactions...")
    (interactions, weights) = dataset.build_interactions(
        # ((x.member_id, x.item_uri) for x in interactions_df.itertuples())
        ((x.member_id, x.item_uri) for x in unique_interactions_df.itertuples())
    )
    print("building item features...")

    item_features = dataset.build_item_features(
        (
            # for each book in our dataset, return tuple of
            # item id and list of features
            (item.id, get_item_features(item))
            for item in books_df.itertuples()
        )
    )

    user_features = dataset.build_user_features(
        (
            # for each member in our dataset, return tuple of
            # member id and list of features
            (member.member_id, get_user_features(member))
            for member in members_df.itertuples()
        )
    )

    # create reverse lookup to get dataset numeric id from member id
    # TODO: can we just add these into the dataframes?
    member_dataset_id = {member_id: i for i, member_id in enumerate(all_members)}
    item_dataset_id = {item_id: i for i, item_id in enumerate(books_df.id)}

    return {
        "dataset": dataset,  # lightfm dataset object
        "interactions": interactions,
        "user_features": user_features,
        "item_features": item_features,
        # feature labels? (movie lens example)
        "users": all_members,
        # dataframes currently needed for generating sample recommendations
        "items": books_df,
        "interactions_df": unique_interactions_df,
        "member_dataset_id": member_dataset_id,
        "item_dataset_id": item_dataset_id,
    }


def get_model():
    dataset = get_data()
    # model = LightFM(learning_rate=0.05, loss="warp", no_components=64, item_alpha=0.001)
    model = LightFM(loss="warp")

    print("fitting model...")
    model.fit(
        dataset["interactions"],
        item_features=dataset["item_features"],
        user_features=dataset["user_features"],
        epochs=50,
    )
    return model
