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

# load members, books, events as csv
print("downloading data...")
members_df = pd.read_csv(csv_urls["members"])
books_df = pd.read_csv(csv_urls["books"])
events_df = pd.read_csv(csv_urls["events"])

# get list of user ids and book ids for dataset

# - generate short id from book uri
books_df["id"] = books_df.uri.apply(lambda x: x.split("/")[-2])

# get all member-book interactions from events
# shorten URIs for readability in output
interactions_df = events_df[events_df.item_uri.notna()].copy()
# split multiple members for shared accounts
interactions_df[
    ["first_member_uri", "second_member_uri"]
] = interactions_df.member_uris.str.split(";", expand=True)
# working with the first member for now...
interactions_df["member_id"] = interactions_df.first_member_uri.apply(
    lambda x: x.split("/")[-2]
)
interactions_df["item_uri"] = interactions_df.item_uri.apply(lambda x: x.split("/")[-2])
# shorten member id
members_df["member_id"] = members_df.uri.apply(lambda x: x.split("/")[-2])


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


def get_item_features(item):
    # get features for an individual item
    return [
        "pubyear_%s" % (item.year if pd.notna(item.year) else "unknown"),
        "author_%s" % (item.author if pd.notna(item.author) else "unknown"),
        "is_multivol"
        if pd.notna(item.volumes_issues) and item.volumes_issues
        else "non_multivol",
    ]
# to add: subject/genre/ from oclc db export and/or wikidata reconcile work

user_feature_list = [
    "gender_%s" % (gender.lower() if pd.notna(gender) else "unknown")
    for gender in members_df.gender.unique()
] + ["person", "organization"]
# consider adding:
# - arrondissement (could have multiple)
# - birth year  / birth decade
# - nationality (could be multiple)


def get_user_features(member):
    # get features for an individual item
    return [
        "gender_%s" % (member.gender.lower() if pd.notna(member.gender) else "unknown"),
        "organization" if member.is_organization else "person",
    ]


dataset.fit(
    all_members,
    books_df.id,
    item_features=item_feature_list,
    user_features=user_feature_list,
)

print("building interactions...")
(interactions, weights) = dataset.build_interactions(
    ((x.member_id, x.item_uri) for x in interactions_df.itertuples())
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

# model = LightFM(learning_rate=0.05, loss="warp", no_components=64, item_alpha=0.001)
model = LightFM(loss="warp")

print("fitting model...")
model.fit(interactions, item_features=item_features, user_features=user_features, epochs=50)

# sample recommendation code adapted from lightfm quickstart example


def sample_recommendation(model, dataset, user_ids):
    n_users, n_items = dataset.interactions_shape()

    for user_id in user_ids:
        # convert numeric dataset user id to member id slug
        member_id = all_members[user_id]

        known_positives = interactions_df[
            interactions_df.member_id == member_id
        ].item_uri

        scores = model.predict(user_id, np.arange(n_items))
        # use book dataframe to translate numeric dataset id to book id
        top_items = [books_df.loc[model_id].id for model_id in np.argsort(-scores)[:3]]


        print("\nUser %s" % member_id)
        print("  Known positives:")
        for x in known_positives[:3]:
            print("\t%s" % x)

        print("  Recommended:")
        for x in top_items[:3]:
            print("\t%s" % x)


# create reverse lookup to get dataset numeric id from member id
member_dataset_id = {member_id: i for i, member_id in enumerate(all_members) }

# generate some sample recommendations
sample_recommendation(
    model,
    dataset,
    [
        member_dataset_id["hemingway"],
        member_dataset_id["pound"],
        member_dataset_id["aldington-richard"],
    ],
)

# recommend for some members without any book interactions
sample_recommendation(
    model,
    dataset,
    [
        member_dataset_id["bryher"],
        member_dataset_id["friend-of-renoir"],
        member_dataset_id["geisel-theodore"],
        member_dataset_id["copland-aaron"],
    ],
)
