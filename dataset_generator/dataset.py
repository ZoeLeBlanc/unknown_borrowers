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
    "members": "../dataset_generator/data/SCoData_members_v1.1_2021-01.csv",
    "books": "../dataset_generator/data/SCoData_books_v1.1_2021-01.csv",
    "events": "../dataset_generator/data/SCoData_events_v1.1_2021-01.csv",
}


def get_item_features(item, books_genres, books_subjects, wikidata_books_genres):
    # get features for an individual item
    # return {
    features = {
        "multivol": 1 if pd.notna(item.volumes_issues) and item.volumes_issues else 0,
    }
    if pd.notna(item.pubyear_normalized):
        features["pubyear"] = item.pubyear_normalized
    else:
        features["pubyear unknown"] = 1

    # split multiple authors and set feature indicator for each
    if pd.notna(item.author):
        features.update({"author %s" % a: 1 for a in item.author.split(";")})
    else:
        features["author unknown"] = 1

    genres = books_genres[books_genres.item_id == item.id]
    if genres.shape[0]:
        features.update({"genre %s" % g: 1 for g in genres.genre})
    subjects = books_subjects[books_subjects.item_id == item.id]
    if subjects.shape[0]:
        features.update({"subject %s" % s: 1 for s in subjects.subject})
    wikidata_genres = wikidata_books_genres[wikidata_books_genres.item_id == item.id]
    if wikidata_genres.shape[0]:
        features.update({"genre %s" % g: 1 for g in genres.genre})

    return features


def get_user_features(member):
    # get features for an individual item
    features = {
        "gender %s"
        % (member.gender.lower() if pd.notna(member.gender) else "unknown"): 1,
    }
    if pd.notna(member.arrondissements):
        features.update(
            {"arrondissement %s" % i: 1 for i in member.arrondissements.split(";") if i}
        )
    if pd.notna(member.birth_year):
        features["birth year"] = member.birthyear_normalized
    else:
        features["birth year unknown"] = 1

    # known viaf or wikipedia indicates some degree of "fame"
    # is this a useful feature to include?
    if pd.notna(member.viaf_url) or pd.notna(member.wikipedia_url):
        features['famousish'] = 1

    # split multiple nationalities and set feature indicator for each
    if pd.notna(member.nationalities):
        features.update(
            {"nationality %s" % c: 1 for c in member.nationalities.split(";")}
        )
    else:
        features["nationality unknown"] = 1

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
    events_df = events_df[~events_df.first_member_uri.isin(org_uris)]
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

    # restrict to borrow events only
    interactions_df = interactions_df[interactions_df.event_type == 'Borrow'].copy()

    # reduce to minimum user/item interaction fields and drop dupes
    unique_interactions_df = interactions_df[
        ["member_id", "item_uri"]
    ].drop_duplicates()

    # generate unique list of member ids with book interactions
    book_members = interactions_df.member_id.unique()
    # include all members, so we can make recommendations for members without documented interactions
    all_members = members_df.member_id.unique()

    # what if we limit to members with at least N interactions?
    # filter by count thanks to https://stackoverflow.com/a/32922418
    min_borrows = 10
    borrowing_count = unique_interactions_df.member_id.value_counts()
    active_members = members_df[members_df.member_id.isin(borrowing_count[borrowing_count >= min_borrows].index)].member_id.unique()

    # normalize book publication year & member birth year
    # copied from https://www.geeksforgeeks.org/normalize-a-column-in-pandas/
    books_df["pubyear_normalized"] = MinMaxScaler().fit_transform(
        np.array(books_df.year).reshape(-1, 1)
    )
    members_df["birthyear_normalized"] = MinMaxScaler().fit_transform(
        np.array(members_df.birth_year).reshape(-1, 1)
    )

    books_genres = pd.read_csv("data/books_genres.csv")
    books_subjects = pd.read_csv("data/books_subjects.csv")
    wikidata_books_genres = pd.read_csv("data/books_wikidata_genres.csv")

    dataset = Dataset()
    # pass list of user ids and list of book ids
    # provide list of unique categorical features

    print("fitting dataset...")
    # list of features to be used when defining the dataset

    # generate list of item ids & item features to be passed to build_item_features
    # for each book in our dataset, return tuple of
    # item id and list of features
    item_feature_data = [
        (item.id, get_item_features(item, books_genres, books_subjects, wikidata_books_genres))
        for item in books_df.itertuples()
    ]

    # corresponding list of features for users
    user_feature_data = [
        # for each member in our dataset, return tuple of
        # member id and list of features
        (member.member_id, get_user_features(member))
        for member in members_df.itertuples()
    ]
    # generate item feature list from unique keys in the item data
    item_feature_list = set()
    for item, features in item_feature_data:
        item_feature_list |= set(features.keys())

    # same for user feature list
    user_feature_list = set()
    for user, features in user_feature_data:
        user_feature_list |= set(features.keys())

    # to add: subject/genre/ from oclc db export and/or wikidata reconcile work

    dataset.fit(
        all_members,
        books_df.id,
        item_features=item_feature_list,
        user_features=user_feature_list,
    )
    # optionally filter interactions
    model_interactions = unique_interactions_df[unique_interactions_df.member_id.isin(active_members)]

    print(f"building interactions ({model_interactions.shape[0]:,} unique interactions)...")
    (interactions, weights) = dataset.build_interactions(
        ((x.member_id, x.item_uri) for x in model_interactions.itertuples())
    )
    print("building item features...")
    item_features = dataset.build_item_features(item_feature_data)
    user_features = dataset.build_user_features(user_feature_data)

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
        epochs=1050,
    )
    return model
