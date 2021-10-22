# script to get more accurate subject & genre information
# for LOC matches

import csv
from time import sleep
import os.path

import requests
from requests.models import LocationParseError
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

class SearchRetrieveResponse(MODS):
    # extend MODS to inherit the mods namespace (cheat)

    # zs namespace if we need it
    # xmlns:zs="http://www.loc.gov/zing/srw/"
    # zs:records/zs:record/zs:recordData/mods
    mods = xmlmap.NodeField("//mods:mods", LoCMODS)


# Item and resource endpoints:
# Burst Limit:  	40 requests per 10 seconds, Block for 5 minutes
# Crawl Limit  	200 requests per 1 minute, Block for 1 hour
# ... not sure if these limits apply to MODS record

# use a request session for connection pooling
session = requests.Session()
session.headers.update({
       'User-Agent': 'loc-reconcler 0.1 / Center for Digital Humanities at Princeton',
        'From': 'rkoeser@princeton.edu'
    })

@sleep_and_retry
# @limits(calls=35, period=10)
@limits(calls=15, period=10)
def get_loc_mods(loc_id, retry=0):
    # MODS xml can be retrieved relative to main loc id
    # BUT: that seems to cause problems on their server! results in 403s
    # use SRU endpoint instead
    lccn_id = loc_id.rstrip('/').split('/')[-1]
    sru_url = f"http://lx2.loc.gov:210/LCDB?operation=searchRetrieve&version=1.1&query=bath.lccn=%22^{ lccn_id }$%22&recordSchema=mods&startRecord=1&maximumRecords=10"
    # sru_url = f"http://lx2.loc.gov:210/lcdb?version=1.1&operation=searchRetrieve&query={ lccn_id }&recordSchema=mods&maximumRecords=1"
    response = session.get(sru_url)
    if response.status_code == requests.codes.ok:
        sru_response = load_xmlobject_from_string(response.content, SearchRetrieveResponse)
        mods = sru_response.mods
        if not mods:
            print('no mods %s' % sru_url)
        return mods
    else:
        if response.status_code == 500 and retry < 3:
            print("😱 got a %s error, trying again (try %s)" % (response.status_code, retry + 1))
            sleep(3*60)
            return get_loc_mods(loc_id, retry=retry + 1)

        print(response.status_code)
        print(response.content)
        # bail out
        raise Exception


# if csv output file exists, load it and make a list of sco ids with existing info
# append to file & skip writing header if it already existed
output_filename = "shxco_loc_subjects_genres.csv"
has_genre_info = []

if os.path.exists(output_filename):
    with open(output_filename) as csvfile:
        csvreader = csv.DictReader(csvfile)
        has_genre_info = set([row["sco_id"] for row in csvreader])

if has_genre_info:
    print("loaded %d records with genre/subject info from csv file" % len(has_genre_info))


with open("shxco_loc_matches_final.csv") as csvfile:
    csvreader = csv.DictReader(csvfile)

    csv_fieldnames = ["sco_id", "term", "type", "source"]
    with open(output_filename, "a") as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
        # only write out the header if we don't have existing info (i.e. new file)
        if not has_genre_info:
            csvwriter.writeheader()

        for row in csvreader:
            # skip if we've already gotten subject/genre info for this one
            if row["sco_id"] in has_genre_info:
                continue

            loc_id = row["id"]
            # print id as an indicator of where we are
            print("%s - %s" % (row["sco_id"].split("/")[-2], loc_id))
            # use a method to load mods so we can apply rate limiting
            try:
                mods = get_loc_mods(loc_id)
            except:
                # if we start getting 500s, break so the file will get closed
                break

            if not mods:
                continue

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
