import pandas as pd
import numpy as np
import datasets
from transformers import AutoTokenizer, DefaultDataCollator, TFAutoModelForSequenceClassification
import string
import nltk
from nltk.stem import WordNetLemmatizer

from pathlib import Path

from mlserver import MLModel
from mlserver.utils import get_model_uri
from mlserver.types import (
    InferenceRequest,
    InferenceResponse
)
from mlserver.codecs import NumpyRequestCodec
from mlserver.codecs.string import StringRequestCodec
from mlserver.logging import logger
Path("1").mkdir(parents=True, exist_ok=True)

nltk.download("stopwords", download_dir="./nltk")
nltk.download("wordnet", download_dir="./nltk")
nltk.download("omw-1.4", download_dir="./nltk")
nltk.data.path.append("./nltk")
# Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')


class ReviewRatings(MLModel):
    async def load(self) -> bool:
        model_uri = await get_model_uri(
            self.settings
        )

        logger.info("Loading model")
        self._model = TFAutoModelForSequenceClassification.from_pretrained(model_uri, num_labels=9)
        logger.info("Model successfully loaded")

        self._wordnet_lemmatizer = WordNetLemmatizer()

        logger.info("Loading tokenizer and data collator")
        self._tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self._data_collator = DefaultDataCollator(return_tensors="tf")

        self.ready = True
        return self.ready

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        logger.info("Starting prediction")
        print("Starting prediction")
        review_df = self.decode_request(payload)
        # review_str = StringRequestCodec.decode_request(payload)
        pred_proc = self.process_whole(review_df)
        response = NumpyRequestCodec.encode_response(model_name=self.name, payload=pred_proc)

        print(response)

        return response

    def preprocess_text(self, df, feature_names):
        logger.info("Preprocessing text")
        # logger.info(f"Incoming text: {text}")
        # dict_text = {"review": text}
        # breakpoint()
        # df = pd.DataFrame(data=dict_text)
        # logger.info(f"Dataframe created: {df}")
        # logger.info("Removing punctuation")
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
            collate_fn=self._data_collator
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
        lemm_text = ' '.join([self._wordnet_lemmatizer.lemmatize(word) for word in text.split()])
        return lemm_text

    def tokenize(self, ds):
        return self._tokenizer(ds["review"], padding="max_length", truncation=True)

    def process_output(self, preds):
        logger.info("Processing model predictions")
        rating_preds = []
        for i in preds["logits"]:
            rating_preds.append(np.argmax(i, axis=0))

        logger.info("Create output array for predictions")
        rating_preds = np.array(rating_preds)

        return rating_preds

    def process_whole(self, text):
        logger.info("Start processing")
        tf_inf = self.preprocess_text(text, feature_names=None)
        logger.info("Predictions ready to be made")
        preds = self._model.predict(tf_inf)
        logger.info(f"Prediction type: {type(preds)}")
        logger.info(f"Predictions: {preds}")
        preds_proc = self.process_output(preds)
        logger.info(f"Processed predictions: {preds_proc}, Processed predictions type: {type(preds_proc)}")

        return preds_proc
