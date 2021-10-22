# reshape data into format easier to use with recommenders
import sys
import csv

import pandas as pd

sys.path.insert(0, '../dataset_generator/')
from dataset import get_shxco_data

def identify_interactions(events_df):
    # get all member-book interactions from events
    interactions_df = events_df[events_df.item_uri.notna()].copy()
    # restrict to borrow events only
    interactions_df = interactions_df[interactions_df.event_type == 'Borrow'].copy()

    # reduce to minimum user/item interaction fields and drop dupes
    unique_interactions_df = interactions_df[
        ["member_id", "item_uri"]
    ].drop_duplicates().rename(columns={"item_uri": "item_id"})
    # rename for consistency (should probably rename in dataset code)

    # save to csv for use in recommender code
    unique_interactions_df.to_csv('data/interactions.csv', index=False)


# def get_item_features(item, books_genres, books_subjects, wikidata_books_genres):
def get_item_features(item):
    # get features for an individual item
    # return {
    features = {
        "multivol": 1 if pd.notna(item.volumes_issues) and item.volumes_issues else 0,
    }
    if pd.notna(item.year):
        # recommendation code should normalize/bin as needed
        features["year"] = item.year
    else:
        features["pubyear unknown"] = 1

    # split multiple authors and set feature indicator for each
    if pd.notna(item.author):
        features.update({"author %s" % a: 1 for a in item.author.split(";")})
    else:
        features["author unknown"] = 1

    # TODO: also include author gender

    # TODO
    # genres = books_genres[books_genres.item_id == item.id]
    # if genres.shape[0]:
    #     features.update({"genre %s" % g: 1 for g in genres.genre})
    # subjects = books_subjects[books_subjects.item_id == item.id]
    # if subjects.shape[0]:
    #     features.update({"subject %s" % s: 1 for s in subjects.subject})
    # wikidata_genres = wikidata_books_genres[wikidata_books_genres.item_id == item.id]
    # if wikidata_genres.shape[0]:
    #     features.update({"genre %s" % g: 1 for g in genres.genre})

    return features


def get_user_features(member):
    # get features for an individual item
    features = {
        "member_id": member.member_id,
        "gender %s"
        % (member.gender.lower() if pd.notna(member.gender) else "unknown"): 1,
    }
    if pd.notna(member.arrondissements):
        features.update(
            {"arrondissement %s" % i: 1 for i in member.arrondissements.split(";") if i}
        )
    if pd.notna(member.birth_year):
        # features["birth year"] = member.birthyear_normalized
        # should normalize within recommender code
        features["birth year"] = member.birth_year
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


def generate_member_features(members_df):
    # TODO: include all members or borrowers only?
    member_features = pd.DataFrame(data=[
        get_user_features(member) for member in members_df.itertuples()
    ])

    member_features.to_csv("data/member_features.csv", index=False)


def generate_book_features(books_df):
    book_features = pd.DataFrame(data=[
        get_item_features(book) for book in books_df.itertuples()
    ])
    book_features.to_csv("data/book_features.csv", index=False)


if __name__ == '__main__':
    # load members, books, events as csv
    members_df, books_df, events_df = get_shxco_data()
    identify_interactions(events_df)
    generate_member_features(members_df)
    generate_book_features(books_df)

