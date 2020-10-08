import joblib
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from utils.utils_pp import ds_from_json_from_zip


# load official development set
df_dev = ds_from_json_from_zip(["mesinesp-full_dev.zip"])
df_dev = pd.DataFrame(df_dev).applymap(lambda x: x[0]).set_index('id').rename_axis(None)
x_dev = df_dev['text'] + df_dev['title']

# load one-hot labels
one_hot = joblib.load("./svm_model/multilabel_binarizer.sav")
y_dev = one_hot.transform(df_dev['labels'])

# load tf-idf matrix
vect = joblib.load("./svm_model/tfidf_vectorizer.sav")
x_dev_tfidf = vect.transform(x_dev)

# load svm model
clf = joblib.load("./svm_model/svm_model.joblib")

# evaluate classifier
print("Starting prediction...")
y_pred = clf.predict(x_dev_tfidf)
up, ur, uf1, _ = precision_recall_fscore_support(y_dev, y_pred, average='micro')
print("f1 micro: ", uf1)
print("precision micro: ", up)
print("recall micro: ", ur)
