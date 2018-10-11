Here you can find the datasets used for the experiments, formatted according to the approach required by this project. More concretely, each action and activity is encoded using Word2Vec English embeddings (from Google News). There are two scripts to perform the transformation between the original datasets and the formatted ones:

word-embedding-creator.py: this script builds three dictionaries, stored as JSON files (each dataset folder can contain various dictionaries, depending on the OP global varaible defined in the script). A dictionary to relate an action with its embedding, an activity with its embedding and a day period (morning, evening...) with its embedding (temporal representations will be used to process the timestamps of the actions).

data-framer.py:
