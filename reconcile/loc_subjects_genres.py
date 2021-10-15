# script to get more accurate subject & genre information
# for LOC matches

import csv
from time import sleep

import requests
from eulxml import xmlmap
from eulxml.xmlmap import load_xmlobject_from_string
from eulxml.xmlmap.mods import MODS, Subject
from ratelimit import limits, sleep_and_retry


class LoCSubject(Subject):
    # add subject mapping to eulxml mods subject
    genre = xmlmap.StringField("mods:genre")


class LoCMODS(MODS):
    # override with our subject class
    subjects = xmlmap.NodeListField("mods:subject", LoCSubject)


# Item and resource endpoints:
# Burst Limit:  	40 requests per 10 seconds, Block for 5 minutes
# Crawl Limit  	200 requests per 1 minute, Block for 1 hour
# ... not sure if these limits apply to MODS record

@sleep_and_retry
@limits(calls=35, period=10)
def get_loc_mods(loc_id):
    # MODS xml can be retrieved relative to main loc id
    response = requests.get(f"{ loc_id }/mods")
    if response.status_code >= 400:
        print("😱 got a %s error, trying again" % response.status_code)
        sleep(1)
        return get_loc_mods(loc_id)
    return load_xmlobject_from_string(response.content, LoCMODS)


with open("shxco_loc_matches_final.csv") as csvfile:
    csvreader = csv.DictReader(csvfile)

    csv_fieldnames = ["sco_id", "term", "type", "source"]
    with open("shxco_loc_subjects_genres.csv", "w") as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)

        for row in csvreader:
            loc_id = row["id"]
            # print id as an indicator of where we are
            print("%s - %s" % (row["sco_id"].split("/")[-2], loc_id))
            # use a method to load mods so we can apply rate limiting
            mods = get_loc_mods(loc_id)
            # get unique subjects & genres
            genre_list = set()
            subject_list = set()
            for genre in mods.genres:
                genre_list.add(genre.text)
            for subject in mods.subjects:
                if subject.topic:
                    subject_list.add(subject.topic)
                if subject.genre:
                    genre_list.add(subject.genre)

            # add row to output for each subject & genre
            for genre in genre_list:
                csvwriter.writerow(
                    {
                        "sco_id": row["sco_id"],
                        "term": genre,
                        "type": "genre",
                        "source": "loc",
                    }
                )
            for subject in subject_list:
                csvwriter.writerow(
                    {
                        "sco_id": row["sco_id"],
                        "term": subject,
                        "type": "subject",
                        "source": "loc",
                    }
                )
