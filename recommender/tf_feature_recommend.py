import os
import pprint
import sys
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf

import tensorflow_recommenders as tfrs

sys.path.insert(0, "../dataset_generator/")
from dataset import get_shxco_data

# adapted from https://www.tensorflow.org/recommenders/examples/basic_retrieval

# load members, books, events as csv
members_df, books_df, events_df = get_shxco_data()

# ratings in tf tutorial = interactions in lightfm
# get all member-book interactions from events
interactions_df = events_df[events_df.item_uri.notna()].copy()
# reduce to minimum user/item interaction fields and drop dupes
unique_interactions_df = interactions_df[["member_id", "item_uri"]].drop_duplicates()

unique_user_ids = unique_interactions_df.member_id.unique()
unique_item_ids = unique_interactions_df.item_uri.unique()


# define vocabulary for titles
title_lookup = tf.keras.layers.StringLookup()
# NOTE: should probably be item id here
title_lookup.adapt(books_df.id.unique())

print(f"title Vocabulary: {title_lookup.get_vocabulary()[:3]}")

# or use feature hashing; doesn't require defining the vocabulary
# We set up a large number of bins to reduce the chance of hash collisions.
num_hashing_bins = 200_000
movie_title_hashing = tf.keras.layers.Hashing(num_bins=num_hashing_bins)

# define embedding layer
title_embedding = tf.keras.layers.Embedding(
    # Let's use the explicit vocabulary lookup.
    input_dim=title_lookup.vocabulary_size(),
    output_dim=32,
)

# then define model as id, embedding
title_model = tf.keras.Sequential([title_lookup, title_embedding])

# create a text vector feature by tokenizing titles
title_text = tf.keras.layers.TextVectorization()
title_text.adapt(books_df.title.unique())


embedding_dimension = 32

user_model = tf.keras.Sequential(
    [
        tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
        # We add an additional embedding to account for unknown tokens.
        tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension),
    ]
)

item_model = tf.keras.Sequential(
    [
        tf.keras.layers.StringLookup(vocabulary=unique_item_ids, mask_token=None),
        tf.keras.layers.Embedding(len(unique_item_ids) + 1, embedding_dimension),
    ]
)


class ShxcoUserModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.user_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(
                    vocabulary=unique_user_ids, mask_token=None
                ),
                tf.keras.layers.Embedding(len(unique_user_ids) + 1, 32),
            ]
        )
        # TODO: add additional features
        # if use_timestamps:
        #   self.timestamp_embedding = tf.keras.Sequential([
        #       tf.keras.layers.Discretization(timestamp_buckets.tolist()),
        #       tf.keras.layers.Embedding(len(timestamp_buckets) + 1, 32),
        #   ])
        #   self.normalized_timestamp = tf.keras.layers.Normalization(
        #       axis=None
        #   )

        #   self.normalized_timestamp.adapt(timestamps)

    def call(self, inputs):
        # if not self._use_timestamps:
        # return self.user_embedding(inputs)
        return self.user_embedding(inputs["member_id"])
        # return self.user_embedding(inputs["user_id"])

        # TODO: add additional features
        # return tf.concat([
        #     self.user_embedding(inputs["user_id"]),
        #     self.timestamp_embedding(inputs["timestamp"]),
        #     tf.reshape(self.normalized_timestamp(inputs["timestamp"]), (-1, 1)),
        # ], axis=1)


user_model = ShxcoUserModel()


class ShxcoItemModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        max_tokens = 10_000

        self.title_embedding = tf.keras.Sequential(
            [
                title_lookup,
                tf.keras.layers.Embedding(title_lookup.vocabulary_size(), 32),
            ]
        )
        self.title_text_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(max_tokens=max_tokens),
                tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
                # We average the embedding of individual words to get one embedding vector
                # per title.
                tf.keras.layers.GlobalAveragePooling1D(),
            ]
        )

    def call(self, titles):
        return tf.concat(
            [
                self.title_embedding(titles),
                self.title_text_embedding(titles),
            ],
            axis=1,
        )


item_model = ShxcoItemModel()
item_model.title_text_embedding.layers[0].adapt(books_df.title.unique())


items = tf.data.Dataset.from_tensor_slices(unique_item_ids)

metrics = tfrs.metrics.FactorizedTopK(
    # candidates=items.batch(128).map(item_model)
    candidates=items.batch(128).map(title_lookup)
)

task = tfrs.tasks.Retrieval(metrics=metrics)


class ShxCoModel(tfrs.Model):
    def __init__(self, user_model, item_model, task):
        super().__init__()
        self.query_model = tf.keras.Sequential([user_model, tf.keras.layers.Dense(32)])
        self.candidate_model = tf.keras.Sequential(
            [item_model, tf.keras.layers.Dense(32)]
        )
        self.item_model: tf.keras.Model = item_model
        self.user_model: tf.keras.Model = user_model
        # self.task: tf.keras.layers.Layer = task
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=items.batch(128).map(self.candidate_model),
            ),
        )

    def compute_loss(
        self, features: Dict[Text, tf.Tensor], training=False
    ) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        # user_embeddings = self.user_model(features["member_id"])
        # And pick out the movie features and pass them into the movie model,
        # getting embeddings back.
        # positive_item_embeddings = self.item_model(features["item_uri"])

        query_embeddings = self.query_model(
            {
                "member_id": features["member_id"],
                # "timestamp": features["timestamp"],
            }
        )
        item_embeddings = self.candidate_model(features["item_uri"])

        # The task computes the loss and the metrics.
        return self.task(query_embeddings, item_embeddings)

    def call(self, features):
        # user_id, movie_title = inputs
        # user_embedding = self.user_embeddings(user_id)
        # movie_embedding = self.movie_embeddings(movie_title)
        query_embeddings = self.query_model(
            {
                "member_id": features["member_id"],
                # "timestamp": features["timestamp"],
            }
        )
        item_embeddings = self.candidate_model(features["item_uri"])

        return self.task(tf.concat([query_embeddings, item_embeddings], axis=1))


model = ShxCoModel(user_model, item_model, task)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))


# convert pd interaction df into tf dataset to match the tutorial
ratings = tf.data.Dataset.from_tensor_slices(dict(unique_interactions_df))

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

total = len(unique_interactions_df)
train_size = int(total * 0.8)

train = shuffled.take(train_size)
test = shuffled.skip(train_size).take(total - train_size)

# we have 21935 unique interactions from borrowing events
# 20% of that is 4387

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

# model.fit(cached_train, epochs=1)
# model.fit(cached_train, epochs=3)
model.fit(cached_train, epochs=10)

# this causes an overflow error
# ... probably because something is wrong with the test data.
# test/train split sizes wrong?
# eval = model.evaluate(cached_test, return_dict=True)
# for key, val in eval.items():
#   print("%s:\t%s" % (key, val))

train_accuracy = model.evaluate(cached_train, return_dict=True)[
    "factorized_top_k/top_100_categorical_accuracy"
]
test_accuracy = model.evaluate(cached_test, return_dict=True)[
    "factorized_top_k/top_100_categorical_accuracy"
]

print(f"Top-100 accuracy (train): {train_accuracy:.2f}.")
print(f"Top-100 accuracy (test): {test_accuracy:.2f}.")


# Create a model that takes in query model and candidate model
index = tfrs.layers.factorized_top_k.BruteForce(model.query_model)

# recommends items out of the entire dataset.
index.index_from_dataset(
    # tf.data.Dataset.zip((items.batch(100), items.batch(100).map(model.item_model)))
    tf.data.Dataset.zip((items.batch(100), items.batch(100).map(model.candidate_model)))
)


user_id = 'hemingway'
affinity_scores, item_ids = index(
    {'member_id': tf.constant([user_id])}
)

print(f"Recommendations for user {user_id} using BruteForce: {item_ids[0, :5]} ({affinity_scores[0, :5]})")


# # Export the query model.
# with tempfile.TemporaryDirectory() as tmp:
#   path = os.path.join(tmp, "model")

#   # Save the index.
#   tf.saved_model.save(index, path)

#   # Load it back; can also be done in TensorFlow Serving.
#   loaded = tf.saved_model.load(path)

#   # Pass a user id in, get top predicted movie titles back.
#   scores, titles = loaded(["hemingway"])

#   # each recommended title is a tensor with datatype string;
#   # convert to numpy then decode from binary
#   print(f"Recommendations: {', '.join(t.numpy().decode() for t in titles[0][:3])}")
