import pandas as pd
import numpy as np
import datasets
from transformers import AutoTokenizer, DefaultDataCollator, TFAutoModelForSequenceClassification
from google.cloud import storage
import logging

from pathlib import Path
Path("1").mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


class ReviewRatings(object):
    def __init__(self, model_path):
        logger.info("Connecting to GCS")
        self.client = storage.Client.create_anonymous_client()
        self.bucket = self.client.bucket('kelly-seldon')

        logger.info(f"Model name: {model_path}")
        self.model = None
        self.prefix = model_path
        self.local_dir = "1/"

        logger.info("Loading tokenizer and data collator")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.data_collator = DefaultDataCollator(return_tensors="tf")

        self.ready = False

    def load_model(self):
        logger.info("Getting model artifact from GCS")
        blobs = self.bucket.list_blobs(prefix=self.prefix)
        for blob in blobs:
            filename = blob.name.split('/')[-1]
            blob.download_to_filename(self.local_dir + filename)
        logger.info("Loading model")
        self.model = TFAutoModelForSequenceClassification.from_pretrained("1", num_labels=9)

    def preprocess_text(self, text, feature_names):
        logger.info("Preprocessing text")
        logger.info(f"Incoming text: {text}")
        dict_text = {"review": text}
        df = pd.DataFrame(data=dict_text)
        logger.info(f"Dataframe created: {df}")

        dataset = datasets.Dataset.from_pandas(df, preserve_index=False)
        logger.info(f"Dataset created: {dataset}")

        tokenized_revs = dataset.map(self.tokenize, batched=True)
        logger.info(f"Tokenized reviews: {tokenized_revs}")

        logger.info("Converting tokenized reviews to tf dataset")
        tf_inf = tokenized_revs.to_tf_dataset(
            columns=["attention_mask", "input_ids"],
            label_cols=["labels"],
            shuffle=True,
            batch_size=1,
            collate_fn=self.data_collator
        )
        logger.info(f"TF dataset created: {tf_inf}")

        return tf_inf

    def tokenize(self, ds):
        return self.tokenizer(ds["review"], padding="max_length", truncation=True)

    def process_output(self, preds):
        logger.info("Processing model predictions")
        rating_preds = []
        for i in preds["logits"]:
            rating_preds.append(np.argmax(i, axis=0))

        logger.info("Create output array for predictions")
        rating_preds = np.array(rating_preds)

        return rating_preds

    def process_whole(self, text):
        tf_inf = self.preprocess_text(text, feature_names=None)
        logger.info("Predictions ready to be made")
        preds = self.model.predict(tf_inf)
        logger.info(f"Prediction type: {type(preds)}")
        logger.info(f"Predictions: {preds}")
        preds_proc = self.process_output(preds)
        logger.info(f"Processed predictions: {preds_proc}, Processed predictions type: {type(preds_proc)}")

        return preds_proc

    def predict(self, text, names=[], meta=[]):
        try:
            if not self.ready:
                self.load_model()
                logger.info("Model successfully loaded")
                self.ready = True
                logger.info(f"{self.model.summary}")
                pred_proc = self.process_whole(text)
            else:
                pred_proc = self.process_whole(text)

            return pred_proc

        except Exception as ex:
            logging.exception(f"Failed during predict: {ex}")
