import sys
sys.path.append('./utils/')
import pandas as pd
from utils.utils_pp import preprocess_df, write_label_file, df_to_json_to_zip
import json


proj_id = "mesinesp"


def get_prpr_data():

    train_95_data_file = "./original_datasets/train_95.json"
    train_5_data_file = "./original_datasets/train_5.json"
    dev_data_file = "./original_datasets/dev_full.json"

    train_95_orig = json.load(open(train_95_data_file, "r"))
    train_95_orig = pd.DataFrame(train_95_orig["articles"])
    train_5_orig = json.load(open(train_5_data_file, "r"))
    train_5_orig = pd.DataFrame(train_5_orig["articles"])
    dev_orig = json.load(open(dev_data_file, "r"))
    dev_orig = pd.DataFrame(dev_orig["articles"])

    col_x, col_y = 'abstractText', 'decsCodes'
    col_title = 'title'

    print("Processing training data...")
    df_train_95_prpr, train_label_to_code = preprocess_df(
        orig_df=train_95_orig, col_x=col_x, col_y=col_y, col_title=col_title,
        keep_en=False, min_counts=20,
        label_to_code=None,
        rm_linebreaks=True, keep_quotes=False, rm_HTML=True,
        to_lower=True, to_tokens=True, nlp_lib='nltk',
        spacy_lm="es_core_news_sm", to_lemmas=True,
        rm_punctuation=True, rm_stopwords=True, join_text=True
    )
    df_to_json_to_zip(df=df_train_95_prpr, zipname=proj_id + '-train_95.zip')

    print("Processing held-out development data...")
    df_train_5_prpr, _ = preprocess_df(
        orig_df=train_5_orig, col_x=col_x, col_y=col_y, col_title=col_title,
        keep_en=False, min_counts=0,
        label_to_code=train_label_to_code,
        rm_linebreaks=True, keep_quotes=False, rm_HTML=True,
        to_lower=True, to_tokens=True, nlp_lib='nltk',
        spacy_lm="es_core_news_sm", to_lemmas=True,
        rm_punctuation=True, rm_stopwords=True, join_text=True
    )
    df_to_json_to_zip(df=df_train_5_prpr, zipname=proj_id + '-train_5.zip')

    print("Processing official development data...")
    df_dev, _ = preprocess_df(
        orig_df=dev_orig, col_x=col_x, col_y=col_y, col_title=col_title,
        keep_en=False, min_counts=0,
        label_to_code=train_label_to_code,
        rm_linebreaks=True, keep_quotes=False, rm_HTML=True,
        to_lower=True, to_tokens=True, nlp_lib='nltk',
        spacy_lm="es_core_news_sm", to_lemmas=True,
        rm_punctuation=True, rm_stopwords=True, join_text=True
    )
    df_to_json_to_zip(df=df_dev, zipname=proj_id + '-full_dev.zip')

    write_label_file(
        label_to_code=train_label_to_code,
        fname=proj_id + '-full_labels.txt')


if __name__ == "__main__":
    get_prpr_data()
