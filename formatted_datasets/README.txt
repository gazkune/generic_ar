Here you can find the datasets used for the experiments, formatted according to the approach required by this project. More concretely, each action and activity is encoded using Word2Vec English embeddings (from Google News). There are two scripts to perform the transformation between the original datasets and the formatted ones:

word-embedding-creator.py: this script builds three dictionaries, stored as JSON files (each dataset folder can contain various dictionaries, depending on the OP global varaible defined in the script). A dictionary to relate an action with its embedding (eg: word_sum_actions.json), an activity with its embedding (eg: word_sum_activities.json) and a day period (morning, evening... -> word_sum_temporal.json) with its embedding (temporal representations will be used to process the timestamps of the actions).

data-framer.py: this script generates the formatted dataset. X: where word embedding sequences are stored, and y: where activity embedding corresponding to a sequence is stored. The script is prepared to select between 'avg' and 'sum' for action/activity representation, to include (or not) daytime period as part of an action sequence and to remove None type activities. 
The stored files are the following:
   DATASET + '/' + OUTPUT_ROOT_NAME + '_x.npy': X
   DATASET + '/' + OUTPUT_ROOT_NAME + '_y_index.npy': y (one hot encoding for activity labels)
   DATASET + '/' + OUTPUT_ROOT_NAME + '_y_embedding.npy': y (activity labels are encoded by word embeddings)

train-val-test-set-creator.py: script to generate the train, validation and test sets following different strategies.
