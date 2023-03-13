# Install cord19q project




# Install scispacy model

import os

import shutil



from cord19q.etl.execute import Execute as Etl

from cord19q.index import Index



# Copy study design models locally

os.mkdir("cord19q")

shutil.copy("../input/cord19-study-design/attribute", "cord19q")

shutil.copy("../input/cord19-study-design/design", "cord19q")



# Build SQLite database for metadata.csv and json full text files

Etl.run("../input/trec-covid-information-retrieval/CORD-19/CORD-19", "cord19q", "../input/cord-19-article-entry-dates/entry-dates.csv", False)



# Copy vectors locally for predictable performance

shutil.copy("../input/cord19-fasttext-vectors/cord19-300d.magnitude", "/tmp")



# Build the embeddings index

Index.run("cord19q", "/tmp/cord19-300d.magnitude")
