import pandas as pd
import numpy as np
import datasets
from transformers import AutoTokenizer, DefaultDataCollator, TFAutoModelForSequenceClassification
from google.cloud import storage
import logging
import string
import nltk
from nltk.stem import WordNetLemmatizer

from pathlib import Path

Path("1").mkdir(parents=True, exist_ok=True)

nltk.download("stopwords", download_dir="./nltk")
nltk.download("wordnet", download_dir="./nltk")
nltk.download("omw-1.4", download_dir="./nltk")
nltk.data.path.append("./nltk")
# Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')

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

        self.wordnet_lemmatizer = WordNetLemmatizer()

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
        logger.info(f"{self.model.summary}")

    def preprocess_text(self, text, feature_names):
        logger.info("Preprocessing text")
        logger.info(f"Incoming text: {text}")
        text_list = text[0]
        dict_text = {"review": text_list}
        df = pd.DataFrame(data=dict_text)
        logger.info(f"Dataframe created: {df}")
        logger.info("Removing punctuation")
        df['review'] = df['review'].apply(lambda x: self.remove_punctuation(x))
        logger.info("Lowercase all characters")
        df['review'] = df['review'].apply(lambda x: x.lower())
        logger.info("Removing stopwords")
        df['review'] = df['review'].apply(lambda x: self.remove_stopwords(x))
        logger.info("Carrying out lemmatization")
        df['review'] = df['review'].apply(lambda x: self.lemmatizer(x))

        len_df = len(df)
        logger.info(f"{len(df)}")

        dataset = datasets.Dataset.from_pandas(df, preserve_index=False)
        logger.info(f"Dataset created: {dataset}")

        tokenized_revs = dataset.map(self.tokenize, batched=True)
        logger.info(f"Tokenized reviews: {tokenized_revs}")

        logger.info("Converting tokenized reviews to tf dataset")
        tf_inf = tokenized_revs.to_tf_dataset(
            columns=["attention_mask", "input_ids"],
            label_cols=["labels"],
            shuffle=True,
            batch_size=len_df,
            collate_fn=self.data_collator
        )
        logger.info(f"TF dataset created: {tf_inf}")

        return tf_inf

    def remove_punctuation(self, text):
        punctuation_free = "".join([i for i in text if i not in string.punctuation])
        return punctuation_free

    def remove_stopwords(self, text):
        text = ' '.join([word for word in text.split() if word not in stopwords])
        return text

    def lemmatizer(self, text):
        lemm_text = ' '.join([self.wordnet_lemmatizer.lemmatize(word) for word in text.split()])
        return lemm_text

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
