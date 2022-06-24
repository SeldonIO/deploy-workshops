import pandas as pd
import datasets
from transformers import AutoTokenizer, DefaultDataCollator, TFAutoModelForSequenceClassification
from tensorflow.keras.models import load_model
from google.cloud import storage

from pathlib import Path
Path("1").mkdir(parents=True, exist_ok=True)


class ReviewRatings(object):
    def __init__(self, model_dir_path):
        self.model = None
        self.prefix = model_dir_path
        self.client = storage.Client.create_anonymous_client()
        self.bucket = self.client.bucket('kelly-seldon')
        self.local_dir = "1/"
        self.load_model()

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.data_collator = DefaultDataCollator(return_tensors="tf")

    def load_model(self):
        blobs = self.bucket.list_blobs(self.prefix)
        for blob in blobs:
            filename = blob.name.split('/')[-1]
            blob.download_to_filename(self.local_dir + filename)
        self.model = TFAutoModelForSequenceClassification.from_pretrained("1", num_labels=9)

    def transform_input(self, text, feature_names):
        dict_text = {"review": text}
        df = pd.DataFrame(data=dict_text)

        dataset = datasets.Dataset.from_pandas(df, preserve_index=False)

        tokenized_revs = dataset.map(self.tokenize, batched=True)

        tf_inf = tokenized_revs.to_tf_dataset(
            columns=["attention_mask", "input_ids"],
            label_cols=["labels"],
            shuffle=True,
            batch_size=16,
            collate_fn=self.data_collator
        )

        return tf_inf

    def tokenize(self, ds):
        return self.tokenizer(ds["review"], padding="max_length", truncation=True)

    def predict(self, text, names=[], meta=[]):
        self.load_model()
        tf_inf = self.transform_input(text, feature_names=None)
        preds = self.model.predict(tf_inf)
