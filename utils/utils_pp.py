import re
import os
from collections import OrderedDict
import zipfile
import json
import numpy as np
import pandas as pd
import numbers
import langid
from tqdm import tqdm
from utils.textpreprocessing import TextPreprocessor


def count_rows_entities(str_list, ch: str = "|"):
    # Count rows with multiple entities
    # from collections import OrderedDict
    count_dict = {}
    # Loop through list of entries
    for row in str_list:
        # If there is no entity
        if type(row) is not str:
            count_dict[0] = count_dict.get(0, 0) + 1
        # Else: increment
        else:
            count_dict[len(row.split(ch))] = \
                count_dict.get(len(row.split(ch)), 0) + 1
    # Return OrderedDict
    count_dict = OrderedDict(
        sorted(count_dict.items(), key=lambda x: x[0])
    )
    return count_dict


def count_entities(str_list, counts=True, ch: str = "|"):
    # from collections import OrderedDict
    new_dict = {}
    # Loop through list of entries
    for entry in str_list:
        # Check if entities still need to be separated
        if not isinstance(entry, list):
            entry = str(entry).split(ch)
        # Loop through separate entities
        for entity in entry:
            # Increment counters
            new_dict[entity] = new_dict.get(entity, 0) + 1
    # Sort by counts
    ordered = sorted(new_dict.items(), reverse=True, key=lambda x: x[1])

    # Return counts or the order of the labels
    if not counts:
        ordered = [(key, f'{i+1:02}') for i, (key, val) in enumerate(ordered)]

    new_dict = OrderedDict(ordered)
    return new_dict


def keep_lang(df, col='text', language='en'):

    # Only keep texts in the appropriate language
    new_df = df[(df[col].map(langid.classify))[0] == language]
    new_df = new_df.reset_index().drop('index', axis=1)

    return new_df


def rm_low_counts(df, col_x: str, col_y: str, min_counts: int = 0):
    # import pandas as pd
    assert col_x in df.columns, 'Invalid data column name'
    assert col_y in df.columns, 'Invalid label column name'

    # New dataframe to be returned:
    new_df = df

    if min_counts > 0:
        entity_counts = count_entities(new_df[col_y])
        # List of labels to remove
        lab_to_rm = [label for (label, counts) in entity_counts.items()
                     if counts < min_counts]
        # Split label column so that each point has a list of labels
        y_lab = new_df[col_y] #.apply(lambda x: x.split("|"))

        # Seek and remove unwanted labels
        for i, row in enumerate(tqdm(y_lab)):
            # y_lab[i] =
            for label in lab_to_rm:
                if label in row:
                    y_lab[i].remove(label)
            # If row becomes unlabeled, set it for removal
            if len(row) == 0:
                y_lab[i] = None

        new_df[col_y] = y_lab
        new_df = new_df.dropna(subset=[col_x, col_y])  # Remove unlabeled rows
        new_df = new_df.reset_index().drop('index', axis=1)  # Reset indices

    return new_df


def split_data(ds, test_size=0.2, use_val_set=True, val_size=0.2, seed=42):
    # Get indices of the rows for each set
    # ds can either be a pd.dataframe we want to split
    # or an int that is the size of the array to split
    # import numpy as np
    if type(ds) is int:
        ds = np.array(range(1, ds + 1))

    np.random.seed(seed=seed)

    mask_test = np.random.rand(len(ds)) < test_size
    test_set, train_val_set = ds[mask_test], ds[~mask_test]

    if not use_val_set:
        return train_val_set, test_set
    else:
        mask_val = np.random.rand(len(train_val_set)) < val_size
        val_set, train_set = train_val_set[mask_val], train_val_set[~mask_val]

        return train_set, val_set, test_set


def write_label_file(label_to_code, fname='label_file.txt', tag='',
                     colsplit='\t'):
    # Store labels and numerical encoding into file
    with open(fname, 'w') as f:
        for key, val in label_to_code.items():
            line = val + colsplit + key + '\n'
            if tag:
                line = tag + colsplit + line
            f.write(line)


def get_label_dict(fname='label_file.txt', colsplit='\t'):
    # Generate label numerical encoding dictionary from file
    # from collections import OrderedDict
    label_to_code = OrderedDict()
    with open(fname, 'r') as f:
        for line in f:
            enc = line.replace('\n', '').split(colsplit)
            label_to_code[enc[-1]] = enc[1]

    code_to_label = OrderedDict((v, k) for k, v in label_to_code.items())
    return label_to_code, code_to_label


def df_to_json_to_zip(df, zipname):
    # Store pandas.DataFrame into json files within zip archive
    # import zipfile, os
    # Open new zip archive requires
    with zipfile.ZipFile(zipname, 'w') as myzip:
        # Iterate through pd.DataFrame rows
        for i, row in df.iterrows():
            # Get row into json format
            line = f'''{{"{i+1:04}": {row.to_json(orient='index')}}}'''
            # json filename
            json_fname = zipname.replace('.zip', '') + f'-{i+1:04}.json'
            # Create json file in directory
            with open(json_fname, 'w') as fp:
                fp.write(line)
            # Add json file to zip archive
            myzip.write(json_fname)
            # Remove json file from directory
            os.remove(json_fname)


def ds_from_json_from_zip(filepaths):
    for f in filepaths:
        assert '.zip' in f, "Not a zip file"
        # Open zipfile
        with zipfile.ZipFile(f) as z:
            ds = {'id': [], 'labels': [], 'title': [], 'text': []}
            ds_size = 0

            name_list = z.namelist()
            # Iterate through json files within zip
            for fp in name_list:
                assert '.json' in fp, "Not a json file"
                with z.open(fp) as json_fp:
                    # Read sample
                    json_string = json_fp.read()
                data = json.loads(json_string.decode("latin"))
                assert len(data.keys()) == 1, "Expected 1 sample/json file"
                data = data[list(data.keys())[0]]
                data['id'] = int(re.sub("[^0-9]", "", fp))-1  # keep integer id
                # Add to sample ds
                for key in ds.keys():
                    try:
                        ds[key].append(data[key])
                    except KeyError:
                        assert key not in ['title', 'text', 'labels']
                        ds[key].append(None)
                ds_size += 1
                assert len(set([len(ds[x]) for x in ds])) <= 1
                yield ds

                # Reset ds
                ds = {'id': [], 'labels': [], 'title': [], 'text': []}
                ds_size = 0


def pad_array(arr_to_pad, length, val=0):
    # Check if array contains only numeric values
    if all(isinstance(n, numbers.Number) for n in arr_to_pad):
        padded_arr = val*np.ones(shape=length,
                                 dtype=type(next(iter(arr_to_pad))))
    else:
        padded_arr = length*[val]

    # Put the original entries into the padded array
    padded_arr[:len(arr_to_pad)] = arr_to_pad
    return padded_arr


def preprocess_df(
        orig_df, col_x: str, col_y: str, col_title: str,
        keep_en: True, min_counts: int,
        newdf_col_names=['labels', 'title', 'text'],
        label_to_code=None,
        # TextPreProcessor parameters:
        rm_linebreaks=True, keep_quotes=False, rm_HTML=True,
        to_lower=True, to_tokens=True, nlp_lib='nltk',
        spacy_lm="en_core_web_sm", to_lemmas=True,
        rm_punctuation=True, rm_stopwords=True, join_text=False,
        # TextVectorizer parameters:
        emb_method='None'):

    def label2code(i):
        try:
            return label_to_code[i]
        except KeyError:
            return '00'

    def preprocess_wrap(pp, i):
        try:
            return pp.preprocess(i)
        except TypeError:
            return ""

    # import numpy as np
    # import pandas as pd
    # import re
    assert col_x in orig_df.columns, 'Invalid data column name'
    assert col_y in orig_df.columns, 'Invalid label column name'
    assert col_title in orig_df.columns, 'Invalid title column name'

    # Remove missing values
    df = orig_df.dropna(subset=[col_x, col_y])
    df = df.reset_index().drop('index', axis=1)  # Reset indices

    y_set = df[col_y].apply(lambda x: list(set(x)))
    df[col_y] = y_set

    # Keep only english texts
    if keep_en:
        df = keep_lang(df, col=col_x, language='en')

    # Keep labels that have more than min_counts; delete rows left unlabeled
    df = rm_low_counts(df=df, col_x=col_x, col_y=col_y,
                       min_counts=min_counts)

    # OrderedDict with <label>: <label number>
    if label_to_code is None:
        label_to_code = count_entities(df[col_y], counts=False)

    # Convert labels to respective number codes
    # y_enc = df[col_y].apply(lambda x: [label_to_code[i] for i in x])
    y_enc = df[col_y].apply(lambda x: [label2code(i) for i in x])

    # Create text processed pandas.DataFrame
    newdf = pd.DataFrame()
    newdf[newdf_col_names[0]] = y_enc

    # TextPreProcessor parameters:
    pp_params = {'rm_linebreaks': rm_linebreaks, 'keep_quotes': keep_quotes,
                 'rm_HTML': rm_HTML, 'to_lower': to_lower,
                 'to_tokens': to_tokens, 'nlp_lib': nlp_lib,
                 'spacy_lm': spacy_lm, 'to_lemmas': to_lemmas,
                 'rm_punctuation': rm_punctuation,
                 'rm_stopwords': rm_stopwords, 'join_text': join_text}

    pp1 = TextPreprocessor(**pp_params)
    pp2 = TextPreprocessor(**pp_params)

    # print(df[col_title].shape)
    # for i, row in enumerate(df[col_title]):
    #     print()
    #     print(row, i)
    #     print(pp1.preprocess(row))

    newdf[newdf_col_names[1]] = [preprocess_wrap(pp1, i) for i in tqdm(df[col_title])]
    newdf[newdf_col_names[2]] = [pp2.preprocess(i) for i in tqdm(df[col_x])]

    return newdf, label_to_code


def batch_generator(file_paths, batch_size=1, preprocessing=None,
                    condition=lambda x: True, seed=42):
    # Returns generator to load data
    # processed=False is the original non-preprocessed dataset, True otherwise
    # preprocessing is the function to apply at the end of data extraction
    # condition is the boolean function to select samples (applied to labels)
    # TODO: Accepts more than one file but does not shuffle them all at once

    # Cycle through each provided zip file
    for i, fname_zip in enumerate(file_paths):
        assert '.zip' in fname_zip, "Not a zip file"

        # Open zipfie containing 1 file per sample
        with zipfile.ZipFile(fname_zip) as f_zip:
            # Initialize batch
            b_size = 0
            batch = {
                'id': [],
                'text': [],
                'labels': []
            }

            # Shuffle samples
            name_list = f_zip.namelist()
            np.random.seed(seed)
            np.random.shuffle(name_list)

            for fname_json in name_list:
                assert '.json' in fname_json, "Not a json file"

                with f_zip.open(fname_json) as f_json:
                    # Read sample
                    json_string = f_json.read()

                data = json.loads(json_string.decode("latin"))
                assert len(data.keys()) == 1, "Only 1 sample per json file"

                data = data[list(data.keys())[0]]
                data['id'] = fname_json
                if not condition(data):
                    continue

                # Add sample to batch
                for key in batch.keys():
                    try:
                        batch[key].append(data[key])
                    except KeyError:
                        assert key not in ['text', 'labels']
                        batch[key].append(None)
                b_size += 1

                # Return batch
                if b_size == batch_size:
                    # Assert batch has proper dimensions in every key
                    assert len(set([len(batch[x]) for x in batch])) <= 1
                    yield batch if preprocessing is None else preprocessing(batch)

                    # Reset batch
                    b_size = 0
                    batch = {
                        'id': [],
                        'text': [],
                        'labels': []
                    }

            # Process last batch (if it does not have batch_size)
            if b_size != 0:
                # Assert batch has proper dimensions in every key
                assert len(set([len(batch[x]) for x in batch])) <= 1
                yield batch if preprocessing is None else preprocessing(batch)

