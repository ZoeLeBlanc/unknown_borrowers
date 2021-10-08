import numpy as np

import sys
  
sys.path.insert(0, '../dataset_generator/')
from dataset import get_data, get_model

# sample recommendation code adapted from lightfm quickstart example

def sample_recommendation(model, dataset, user_ids):
    n_users, n_items = dataset['dataset'].interactions_shape()

    for user_id in user_ids:
        # convert numeric dataset user id to member id slug
        member_id = dataset["users"][user_id]

        known_positives = dataset['interactions_df'][
            dataset['interactions_df'].member_id == member_id
        ].item_uri

        scores = model.predict(user_id, np.arange(n_items))
        # use book dataframe to translate numeric dataset id to book id
        top_items = [dataset["items"].loc[model_id].id for model_id in np.argsort(-scores)[:3]]

        print("\nUser %s" % member_id)
        print("  Known positives:")
        for x in known_positives[:3]:
            print("\t%s" % x)

        print("  Recommended:")
        for x in top_items[:3]:
            print("\t%s" % x)



def recommend():
    dataset = get_data()
    model = get_model()

    # generate some sample recommendations
    sample_recommendation(
        model,
        dataset,
        [
            dataset["member_dataset_id"]["hemingway"],
            dataset["member_dataset_id"]["pound"],
            dataset["member_dataset_id"]["aldington-richard"],
        ],
    )

    # recommend for some members without any book interactions
    sample_recommendation(
        model,
        dataset,
        [
            dataset["member_dataset_id"]["bryher"],
            dataset["member_dataset_id"]["friend-of-renoir"],
            dataset["member_dataset_id"]["geisel-theodore"],
            dataset["member_dataset_id"]["copland-aaron"],
        ],
    )

if __name__ == "__main__":
    recommend()