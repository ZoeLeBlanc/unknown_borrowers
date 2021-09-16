import itertools

import numpy as np
import pandas as pd
from lightfm.data import Dataset
from lightfm import LightFM
from sklearn.preprocessing import MinMaxScaler


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
    # return {
    features = {
        "multivol": 1 if pd.notna(item.volumes_issues) and item.volumes_issues else 0,
    }

    if pd.notna(item.pubyear_normalized):
        features["pubyear"] = item.pubyear_normalized
    else:
        features["pubyear_unknown"] = 1

    if pd.notna(item.author):
        features.update({'author %s' % a: 1 for a in item.author.split(';')})
    else:
        features['author unknown'] = 1

    return features


def get_user_features(member):
    # get features for an individual item
    features = {
        "gender_%s"
        % (member.gender.lower() if pd.notna(member.gender) else "unknown"): 1,
    }
    if pd.notna(member.arrondissements):
        features.update(
            {"arrondissement_%s" % i: 1 for i in member.arrondissements.split(";") if i}
        )
    if pd.notna(member.birth_year):
        features["birth_year"] = member.birthyear_normalized
    else:
        features["birth_year_unknown"] = 1

    if pd.notna(member.nationalities):
        features.update({'nationality %s' % c: 1 for c in member.nationalities.split(';')})
    else:
        features['nationality unknown'] = 1


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

    # remove events for organizations and shared accounts,
    # since they are unlikely to be helpful or predictive in our model
    # - remove shared accounts (any event with a second member uri)
    events_df = events_df[events_df.second_member_uri.isna()]
    # - remove organizations
    org_uris = list(members_df[members_df.is_organization].uri)
    events_df = events_df[events_df.first_member_uri.isin(org_uris)]
    # TODO: should these also be removed from the members and books
    # used as users and items in the model?

    # working with the first member for now...
    # generate short ids equivalent to those in member and book dataframes
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

    # normalize book publication year & member birth year
    # copied from https://www.geeksforgeeks.org/normalize-a-column-in-pandas/
    books_df["pubyear_normalized"] = MinMaxScaler().fit_transform(
        np.array(books_df.year).reshape(-1, 1)
    )
    members_df["birthyear_normalized"] = MinMaxScaler().fit_transform(
        np.array(members_df.birth_year).reshape(-1, 1)
    )

    dataset = Dataset()
    # pass list of user ids and list of book ids
    # provide list of unique categorical features

    print("fitting dataset...")
    # list of features to be used when defining the dataset

    # generate a list of unique authors
    # split multiple authors, use itertools to flatten the list of lists,
    # use set to re-uniquify, then convert back to a list
    authors = list(
        set(
            itertools.chain.from_iterable(
                [a.split(";") if pd.notna(a) else [] for a in books_df.author.unique()]
            )
        )
    )

    item_feature_list = (
        ["pubyear", "pubyear_unknown", "multivol", "author unknown"]
        + [
            "author %s" % author for author in authors
        ]
    )
    # to add: subject/genre/ from oclc db export and/or wikidata reconcile work
    # anything added here must be implemented in get_item_features

    # generate a list of unique nationalities; same process as for authors
    countries = list(
        set(
            itertools.chain.from_iterable(
                [c.split(";") if pd.notna(c) else [] for c in members_df.nationalities.unique()]
            )
        )
    )

    user_feature_list = (
        [
            "gender_%s" % (gender.lower() if pd.notna(gender) else "unknown")
            for gender in members_df.gender.unique()
        ]
        + ["birth_year", "birth_year_unknown", "nationality unknown"]
        + [
            # Paris arrondissements are 1-20
            "arrondissement_%d" % i
            for i in range(1, 21)
        ] + [
            "nationality %s" % country for country in countries
        ]
    )
    # any features added here must be implemented in get_user_features

    dataset.fit(
        all_members,
        books_df.id,
        item_features=item_feature_list,
        user_features=user_feature_list,
    )

    print("building interactions...")
    (interactions, weights) = dataset.build_interactions(
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
