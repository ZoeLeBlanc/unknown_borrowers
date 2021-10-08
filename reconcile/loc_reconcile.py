import csv
import json
from time import sleep
from json.decoder import JSONDecodeError
import os.path

from ratelimit import limits, sleep_and_retry
import requests

# script to


# some searches return zero results because S&co author doesn't match LOC author
# as a workaround, add mappings for the ones I've looked up and
# corrected manually
# to fix properly, should use author export from the S&co database with viafs,
# and then use viaf to lookup LCCN authorized name
authorized_author = {
    "Dobson, Henry Austin": "Dobson, Austin",  # confirm ?
    "Shorthouse, Joseph Henry": "shorthouse, j. h. (joseph+henry)",
    "Blackmore, Richard Doddridge": "Blackmore, R. D. (Richard Doddridge)",
    "Gaskell, Elizabeth": "Gaskell, Elizabeth Cleghorn",
    "Macaulay, Thomas Babington": "Macaulay, Thomas Babington Macaulay",
    "Holmes, Oliver Wendel": "Holmes, Oliver Wendell",  # typo in S&co db, fixed now
    "Julian, of Norwich": "Julian",
    "Smollett, Tobias": "Smollett, T. (Tobias)",
    "Radcliffe, Ann": "Radcliffe, Ann Ward",
    "Bulwer-Lytton, Edward": "Lytton, Edward Bulwer Lytton",  # same?
    "Borrow, George Henry": "Borrow, George",
    "Tennyson, Alfred": "Tennyson, Alfred Tennyson",
    "Shaw, George Bernard": "Shaw, Bernard",
    "Stockton, Frank Richard": "Stockton, Frank R.",
    "Gissing, George Robert": "Gissing, George",
    "Turgenev, Ivan": "Turgenev, Ivan Sergeevich",
    "Nabokov, Vladimir": "Nabokov, Vladimir Vladimirovich",
    "Sayers, Dorothy": " Sayers, Dorothy L. (Dorothy Leigh)",
}


# LOC API docs limits:
# Collections, format, and other endpoints:
# Burst Limit  	20 requests per 10 seconds, Block for 5 minutes
# Crawl Limit 	80 requests per 1 minute, Block for 1 hour

# setting limits with 20 and 18 both resulted in rate limiting;
# not sure if problem is LOC or ratelimit, but 15 worked

@sleep_and_retry
@limits(calls=15, period=10)
def search_loc(title, author):
    # get authorized version of this author's name if there is one
    author = authorized_author.get(author, author)
    print("searching for: %s / %s" % (title, author))
    search_opts = {
        "all": "true",  # return all, not just titles available digitally
        "q": title,  # use title as search term
        "fo": "json",
        "at": "results,pagination",
    }
    # facet on author if specified
    if author:
        search_opts["fa"] = "contributor:%s" % author.lower()  # author facet

    response = requests.get("https://www.loc.gov/books/", params=search_opts)
    # seem to be getting some transient 500 errors;
    # sleep and try again?
    if response.status_code >= 500:
        print("😱 got a %s error, trying again" % response.status_code)
        sleep(1)
        return search_loc(title, author)

    try:
        json_response = response.json()
    except JSONDecodeError:
        print("json error %s" % response.url)
        raise

    if not json_response["results"]:
        print("😥 no results")

        # if no match and title has a colon, try searching without it
        if ":" in title:
            return search_loc(title.split(":")[0], author)

        return None
    # if we get no results, could try without author?
    # but would have to check the results much more carefully

    # first result seems pretty good; just use that for now
    return json_response["results"][0]

# load books dataset from json
with open("../dataset_generator/data/SCoData_books_v1.1_2021-01.json") as jsonfile:
    book_data = json.load(jsonfile)

csv_fieldnames = [
    "sco_id",
    "sco_title",
    "title",
    "sco_author",
    "author",
    "sco_date",
    "date",
    "id",
    "subject",
]

output_filename = "shxco_loc_matches.csv"

# if csv output file exists, load it and make a list of sco ids with existing matches
# append to file & skip writing header if it alread existed
previous_matches = []

if os.path.exists(output_filename):
    with open(output_filename) as csvfile:
        csvreader = csv.DictReader(csvfile)
        previous_matches = [row["sco_id"] for row in csvreader]

if previous_matches:
    print("loaded %d previous matches from csv file" % len(previous_matches))

# open to append, in case the file already exists
with open(output_filename, "a") as csvfile:
    csvwriter = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
    # only write out the header if we don't have existing matches (i.e. new file)
    if not previous_matches:
        csvwriter.writeheader()

    for book in book_data:
        # skip if we've already got a match for this one
        if book["uri"] in previous_matches:
            continue
        if book.get("format") != "Book":
            # skip books for now, since the LoC query is book specific
            continue
        title = book["title"].split("(")[0]
        # use first author only for multiple authors
        author = book["author"][0] if "author" in book else None
        result = search_loc(title, author)
        if result:
            csvwriter.writerow(
                {
                    "sco_id": book["uri"],
                    "sco_title": book["title"],
                    "title": result["title"],
                    "sco_author": ";".join(book.get("author", [])),
                    "author": ";".join(result.get("contributor", [])),
                    "sco_date": book.get("year"),
                    "date": result.get("date"),
                    "id": result["id"],
                    "subject": ";".join(result.get("subject", [])),
                }
            )
