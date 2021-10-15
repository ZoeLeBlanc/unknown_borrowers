from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k

from sklearn.model_selection import train_test_split

import pandas as pd
from rich.console import Console
from rich.table import Table

import sys
  
sys.path.append("..")
from dataset_generator.dataset import get_data


def validate_model():
    dataset = get_data()

    # split into test/train set
    # NOTE: there is no guarantee that users/members with interactions
    # in the test set are included in the training set
    train, test = train_test_split(dataset["interactions"])

    # create a model and fit it
    # warp seems better for our data & goals than bpr
    # how to determine number of components? too many components == overfitting?
    model = LightFM(loss="warp") #, no_components=24) #, item_alpha=0.1, user_alpha=0.1)

    print('fitting model...')
    model.fit(
        train,
        # dataset['interactions'],  # train on everything
        item_features=dataset["item_features"],
        user_features=dataset["user_features"],
        epochs=500,
    )

    # check test/train precision
    train_precision = precision_at_k(
        model,
        train,
        user_features=dataset["user_features"],
        item_features=dataset["item_features"],
        k=10,
    ).mean()
    test_precision = precision_at_k(
        model,
        test,
        # optional to include training interactions; but causes an error (different shape)
        # train_interactions=train,
        user_features=dataset["user_features"],
        item_features=dataset["item_features"],
        k=10,
    ).mean()

    # check test/train recall
    train_recall = recall_at_k(
        model,
        train,
        user_features=dataset["user_features"],
        item_features=dataset["item_features"],
        k=10,
    ).mean()

    test_recall = recall_at_k(
        model,
        test,
        # optional to include training interactions; but causes an error
        # train_interactions=train,   # causes an error
        user_features=dataset["user_features"],
        item_features=dataset["item_features"],
        k=10,
    ).mean()

    # check test/train ROC AUC
    train_auc = auc_score(
        model,
        train,
        user_features=dataset["user_features"],
        item_features=dataset["item_features"],
    ).mean()

    test_auc = auc_score(
        model,
        test,
        # optional to include training interactions to check for overlap,
        # but causes an error
        # train_interactions=train,
        user_features=dataset["user_features"],
        item_features=dataset["item_features"],
    ).mean()

    table = Table(show_header=True)
    table.add_column("dataset", justify="right")
    table.add_column("precision")
    table.add_column("recall")
    table.add_column("AUC")
    table.add_row(
        "train", "%.2f" % train_precision, "%.2f" % train_recall, "%.2f" % train_auc
    )
    table.add_row(
        "test", "%.2f" % test_precision, "%.2f" % test_recall, "%.2f" % test_auc
    )
    Console().print(table)

    # put the data into a df so we can easily generate a markdown version of the table
    results_df = pd.DataFrame(columns=['dataset', 'precision', 'recall', 'AUC'],
        data=[('train', "%.3f" % train_precision, "%.3f" %  train_recall, "%.3f" %  train_auc),
              ('test', "%.3f" % test_precision, "%.3f" % test_recall, "%.3f" % test_auc)])
    print(results_df.to_markdown(index=False))

if __name__ == "__main__":
    validate_model()
