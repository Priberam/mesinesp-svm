
### Priberam at MESINESP Multi-label Classification of Medical Texts Task: SVM model

This repository contains the SVM model submitted to the BioASQ task [MESINESP](https://temu.bsc.es/mesinesp/), presented at the conference CLEF2020.

The present code assumes that the oficial training and development sets are organized inside the folder "original_datasets" with the same [json structure as originally provided](https://temu.bsc.es/mesinesp/index.php/datasets/). Three files should be contained in this folder:
 * train_95.json: set with 95% of the official training set samples, used for training.
 * train_5.json: set with the remaining 5% of the official training set samples, used as an additional development set.
 * dev_full.json: official development set.

#### To preprocess the data:
```
python3 data_preprocessor.py
```
#### To train the model:
```
python3 svm_trainer.py
```
#### To predict/evaluate on official dev set:
```
python3 svm_predictor.py
```

A trained model is provided in the folder "svm_model". This contains saved versions of the tf-idf vectorizer and multilabel binarizer. To download the saved model and place it in this folder:
```
wget -P svm_model ftp://"ftp.priberam.pt|anonymous"@ftp.priberam.pt/Mesinesp/svm_model.joblib
```

For some machines, during training the error "IOError: [Errno 28] No space left on device" might happen. Since the training process uses multiprocessing, all shared memory might be used. There are two possible [solutions](https://stackoverflow.com/questions/40115043/no-space-left-on-device-error-while-fitting-sklearn-model):
 * Set n_jobs=1, this stops multiprocessing and considerably slows down training.
 * Set the environment variable JOBLIB_TEMP_FOLDER to something different, e.g., JOBLIB_TEMP_FOLDER=/tmp