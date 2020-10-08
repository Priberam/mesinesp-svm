import sys
sys.path.append('./utils/')

import pandas as pd
from utils.utils_pp import ds_from_json_from_zip, get_label_dict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import joblib
import time


def load_prpr_data():
    filepaths = [f'mesinesp-{s}.zip' for s in ['train_95', 'train_5', 'full_dev']]

    df_train_95_prpr = ds_from_json_from_zip([filepaths[0]])
    df_train_95_prpr = pd.DataFrame(df_train_95_prpr).applymap(lambda x: x[0]).set_index('id').rename_axis(None)

    df_train_5_prpr = ds_from_json_from_zip([filepaths[1]])
    df_train_5_prpr = pd.DataFrame(df_train_5_prpr).applymap(lambda x: x[0]).set_index('id').rename_axis(None)

    label_to_code, code_to_label = get_label_dict('mesinesp-full_labels.txt')

    return df_train_95_prpr, df_train_5_prpr, label_to_code, code_to_label


if __name__ == "__main__":

    df_train, df_dev, label_to_code, code_to_label = load_prpr_data()

    # use article title and abstract as input data
    x_train = df_train['text'] + ' ' + df_train['title']
    x_dev = df_dev['text'] + df_dev['title']

    # encode labels
    all_class = list(set([elem for row in df_train['labels'] for elem in row]))
    all_class.append("00")
    print("Total number of labels: ", len(all_class))

    one_hot = MultiLabelBinarizer(classes=all_class, sparse_output=True)
    one_hot.fit(df_train['labels'])
    joblib.dump(one_hot, "./svm_model/multilabel_binarizer.sav")

    y_train = one_hot.transform(df_train['labels'])
    y_dev = one_hot.transform(df_dev['labels'])

    # get tf-idf matrix
    vect = TfidfVectorizer(min_df=1, max_df=0.5, norm='l2')
    x_train_tfidf = vect.fit_transform(x_train)
    joblib.dump(vect, "./svm_model/tfidf_vectorizer.sav")

    x_dev_tfidf = vect.transform(x_dev)
    print("TF-IDF features obtained.")

    # train classifier
    time0 = time.time()
    clf = OneVsRestClassifier(svm.LinearSVC(class_weight='balanced', penalty='l2', C=1.0, loss='squared_hinge',
                                            verbose=1, random_state=0, dual=True), n_jobs=6)
    print("SVM train started... ")
    clf.fit(x_train_tfidf, y_train)
    time_h = (time.time()-time0)/3600
    print(f"SVM train completed in {time_h} hours.")

    joblib.dump(clf, './svm_model/svm_model.joblib', compress=True)

    # evaluate classifier
    y_pred = clf.predict(x_dev_tfidf)
    print("f1 micro (dev): ", f1_score(y_dev, y_pred, average="micro"))
    print("recall micro (dev):", recall_score(y_dev, y_pred, average="micro"))
    print("precision micro (dev):", precision_score(y_dev, y_pred, average="micro"))

