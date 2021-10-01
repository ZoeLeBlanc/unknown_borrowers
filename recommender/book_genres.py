import argparse

import pandas as pd


def db_book_genres(book_csv):
    mep_db = pd.read_csv(book_csv)
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

    print(
        "%d books with genre information, %d unique genres"
        % (len(genre_df.item_id.unique()), len(genre_df.genre.unique()))
    )
    # save the result
    genre_df.to_csv("data/books_genres.csv", index=False)

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
    print(
        "%d books with subject information, %d unique subjects"
        % (len(subject_df.item_id.unique()), len(subject_df.subject.unique()))
    )
    # save the result
    subject_df.to_csv("data/books_subjects.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pull genres from S&co database book export"
    )
    parser.add_argument(
        "file", metavar="FILE", help="Path to the CSV export generated form S&co admin"
    )
    args = parser.parse_args()
    db_book_genres(args.file)
