import argparse
import os.path
import sys

import pandas as pd

sys.path.append("..")
from dataset_generator.dataset import DATA_DIR




def get_oclc_subject_genre(db_export):
    # get genre & subject from database export and split out

    mep_db = pd.read_csv(db_export)

    # fields we care about are slug, Genre_List, Subject_List
    book_genres = mep_db[mep_db.Genre_List.notna()][["slug", "Genre_List"]]

    # explode genre & books —
    # thanks to https://sureshssarda.medium.com/pandas-splitting-exploding-a-column-into-multiple-rows-b1b1d59ea12e

    # split multiple genres into separate columns
    genre_df = pd.DataFrame(
        book_genres.Genre_List.str.split(";").tolist(), index=book_genres.slug
    ).stack()
    # reset index so each genre is paired with book id
    genre_df = genre_df.reset_index([0, "slug"])
    # rename columns
    genre_df.columns = ["item_id", "genre"]
    # clean up whitespace in genre labels
    genre_df["genre"] = genre_df.genre.apply(lambda x: x.strip())
    # filter out "genres" in the OCLC data that aren't relevant for us
    # (specific to a particular copy)
    genre_df = genre_df[
        ~(
            genre_df.genre.str.contains("Provenance")
            | genre_df.genre.str.contains("Binding")
            | genre_df.genre.str.contains("Printing")
            | genre_df.genre.str.contains("Text edition")
            | genre_df.genre.str.contains(r"Book (?:covers|design|jackets)", regex=True)
        )
    ]

    # set type and source for all
    genre_df["type"] = "genre"
    genre_df["source"] = "oclc"
    # rename genre to term for combined output
    genre_df = genre_df.rename(columns={"genre": "term"})

    # do the same thing for subjects
    book_subjects = mep_db[mep_db.Subject_List.notna()][["slug", "Subject_List"]]

    # split multiples into separate columns
    subject_df = pd.DataFrame(
        book_subjects.Subject_List.str.split(";").tolist(), index=book_subjects.slug
    ).stack()
    # reset index so each genre is paired with book id
    subject_df = subject_df.reset_index([0, "slug"])
    # rename columns
    subject_df.columns = ["item_id", "subject"]
    # clean up whitespace in genre labels
    subject_df["subject"] = subject_df.subject.apply(lambda x: x.strip())
    # for some reason there seem to be some exact duplicates; drop them
    subject_df = subject_df.drop_duplicates()

    # set type and source for all
    subject_df["type"] = "subject"
    subject_df["source"] = "oclc"
    # rename subject to term for combined output
    subject_df = subject_df.rename(columns={"subject": "term"})

    return pd.concat([genre_df, subject_df])


def get_wikidata_genre():
    # wikidata reconciliation only includes genre, not subject

    book_wikdata = pd.read_csv(os.path.join(DATA_DIR, "sco_books_wikidata.csv"))
    # fields we care about are genres, sco_uri
    # generate short id from book uri
    book_wikdata["item_id"] = book_wikdata.sco_uri.apply(lambda x: x.split("/")[-2])
    book_genres = book_wikdata[book_wikdata.genres.notna()][["item_id", "genres"]]

    # explode genre & books —
    # thanks to https://sureshssarda.medium.com/pandas-splitting-exploding-a-column-into-multiple-rows-b1b1d59ea12e

    # split multiple genres into separate columns
    genre_df = pd.DataFrame(
        book_genres.genres.str.split(";").tolist(), index=book_genres.item_id
    ).stack()
    # reset index so each genre is paired with book id
    genre_df = genre_df.reset_index([0, "item_id"])
    # rename columns
    genre_df.columns = ["item_id", "genre"]
    # clean up whitespace in genre labels
    genre_df["genre"] = genre_df.genre.apply(lambda x: x.strip())
    # set type and source for all
    genre_df["type"] = "genre"
    genre_df["source"] = "wikidata"

    # for some reason there seem to be some exact duplicates; drop them
    genre_df = genre_df.drop_duplicates()

    # rename genre to term for combined output
    genre_df = genre_df.rename(columns={"genre": "term"})

    return genre_df


# consolidate subjects and genres from LoC, OCLC, and wikidata into a single file
def combine_genres(other_genres):
    # load subjects & genre generated from LoC reconciliation
    genre_df = pd.read_csv("shxco_loc_subjects_genres.csv")
    # loc genre output uses full sco uri, shorten to match the other genre data
    genre_df["item_id"] = genre_df.sco_id.apply(lambda x: x.split("/")[-2])

    combined_genre_df = pd.concat([genre_df, other_genres])
    # set field order and drop unwanted field (sco_id from loc subject/genre)
    combined_genre_df = combined_genre_df[["item_id", "term", "type", "source"]]
    combined_genre_df.to_csv(os.path.join(DATA_DIR, "SCoData_books_genre.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consolidate genres & subject information"
    )
    parser.add_argument(
        "oclc_file", metavar="FILE", help="Path to the CSV of S&co book data admin export"
    )
    args = parser.parse_args()
    oclc_info = get_oclc_subject_genre(args.oclc_file)
    wikidata_info = get_wikidata_genre()
    combine_genres(pd.concat([oclc_info, wikidata_info]))




