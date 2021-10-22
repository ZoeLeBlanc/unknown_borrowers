import os
import pprint
import sys
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf

import tensorflow_recommenders as tfrs

sys.path.insert(0, '../dataset_generator/')
from dataset import get_shxco_data

# adapted from https://www.tensorflow.org/recommenders/examples/basic_retrieval

# load members, books, events as csv
members_df, books_df, events_df = get_shxco_data()

# ratings in tf tutorial = interactions in lightfm
# get all member-book interactions from events
interactions_df = events_df[events_df.item_uri.notna()].copy()
# reduce to minimum user/item interaction fields and drop dupes
unique_interactions_df = interactions_df[
    ["member_id", "item_uri"]
].drop_duplicates()

unique_user_ids = unique_interactions_df.member_id.unique()
unique_item_ids = unique_interactions_df.item_uri.unique()

embedding_dimension = 32

user_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_user_ids, mask_token=None),
  # We add an additional embedding to account for unknown tokens.
  tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
])

item_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_item_ids, mask_token=None),
  tf.keras.layers.Embedding(len(unique_item_ids) + 1, embedding_dimension)
])


items = tf.data.Dataset.from_tensor_slices(unique_item_ids)

metrics = tfrs.metrics.FactorizedTopK(
  candidates=items.batch(128).map(item_model)
)

task = tfrs.tasks.Retrieval(
  metrics=metrics
)

class ShxCoModel(tfrs.Model):

  def __init__(self, user_model, item_model, task):
    super().__init__()
    self.item_model: tf.keras.Model = item_model
    self.user_model: tf.keras.Model = user_model
    self.task: tf.keras.layers.Layer = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["member_id"])
    # And pick out the movie features and pass them into the movie model,
    # getting embeddings back.
    positive_item_embeddings = self.item_model(features["item_uri"])

    # The task computes the loss and the metrics.
    return self.task(user_embeddings, positive_item_embeddings)


model = ShxCoModel(user_model, item_model, task)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))


# convert pd interaction df into tf dataset to match the tutorial
ratings = tf.data.Dataset.from_tensor_slices(dict(unique_interactions_df))

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

# we have 21935 unique interactions from borrowing events
# 20% of that is 4387

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

model.fit(cached_train, epochs=3)

# this causes an overflow error
# ... probably because something is wrong with the test data.
# test/train split sizes wrong?
# eval = model.evaluate(cached_test, return_dict=True)
# print(eval)

# Create a model that takes in raw query features, and
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
# recommends items out of the entire dataset.
index.index_from_dataset(
  tf.data.Dataset.zip((items.batch(100), items.batch(100).map(model.item_model)))
)

# Get some recommendations.
_, titles = index(np.array(["hemingway"]))
print(f"Top 3 recommendations for hemingway: {titles[0, :3]}")


# Export the query model.
with tempfile.TemporaryDirectory() as tmp:
  path = os.path.join(tmp, "model")

  # Save the index.
  tf.saved_model.save(index, path)

  # Load it back; can also be done in TensorFlow Serving.
  loaded = tf.saved_model.load(path)

  # Pass a user id in, get top predicted movie titles back.
  scores, titles = loaded(["hemingway"])

  # each recommended title is a tensor with datatype string;
  # convert to numpy then decode from binary
  print(f"Recommendations: {', '.join(t.numpy().decode() for t in titles[0][:3])}")
