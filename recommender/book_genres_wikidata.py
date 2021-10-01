import argparse

import pandas as pd


def db_book_genres(book_csv):
    book_wikdata = pd.read_csv(book_csv)
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

    print(
        "%d books with genre information, %d unique genres"
        % (len(genre_df.item_id.unique()), len(genre_df.genre.unique()))
    )
    # save the result
    genre_df.to_csv("data/books_wikidata_genres.csv", index=False)


if __name__ == "__main__":
    db_book_genres('data/sco_books_wikidata.csv')
