import json
import pickle

from bertopic import BERTopic
from umap import UMAP



def main():

    print("0: load the config")
    with open("config.json", "r") as f:
        config = json.load(f)

    print("1. load the poor comments")
    with open("comments.json", "r", encoding="utf-8") as f:
        imported_comments: dict = json.load(f)

    print("2. combine the comment lists, in the order of their keys, as imported")
    comments = []
    for seed in imported_comments.keys():
        comments += imported_comments[
            seed
        ]  # same order as poor_comments.json, thus same order as, indices.json

    print("3. set up the topic model")
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=config["random_seeds"]["topic"],
    )
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2), stop_words="english", min_df=0.05
    )  # reduce size of resulting tf-idf table
    topic_model: BERTopic = BERTopic(
        embedding_model="all-MiniLM-L6-v2",  # default model
        calculate_probabilities=False,  # saves computation
        vectorizer_model=vectorizer_model,
        min_topic_size=170,
        # low_memory=True,
    )

    print("3. fit topic model")
    topics, _ = topic_model.fit_transform(comments)

    print("4. save")
    topic_model.save(
        "topic_model.safetensors",
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=False,  # avoids issues with fitting on GPU as per https://stackoverflow.com/questions/74860769/loading-a-gpu-trained-bertopic-model-on-cpu
    )


if __name__ == "__main__":
    main()
