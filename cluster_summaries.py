#%%
from flair.data import Sentence
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
from flair.models import SequenceTagger
import seaborn as sns


#%%
hunflair = SequenceTagger.load("hunflair-disease")

#%%
embedder = SentenceTransformer('stsb-roberta-base')

with open("data/50_50/val.target") as f:
    corpus = [l.strip() for l in f.readlines()]
corpus_embeddings = embedder.encode(corpus)

#%%
tagged_corpus = [Sentence(l) for l in corpus]
hunflair.predict(tagged_corpus)
#%%
blinded_corpus = []
sentence: Sentence
for sentence in tagged_corpus:
    text = sentence.to_original_text()
    for span in sentence.get_spans():
        text = text.replace(span.text, "disease")
    blinded_corpus.append(text)

blinded_corpus_embeddings = embedder.encode(corpus)


#%%
dists = euclidean_distances(corpus_embeddings)
sns.distplot(dists)

#%%
dists = euclidean_distances(blinded_corpus_embeddings)
sns.distplot(dists)

#%%
clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=20) #, affinity='cosine', linkage='average', distance_threshold=0.4)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = {}
for sentence_id, cluster_id in enumerate(cluster_assignment):
    if cluster_id not in clustered_sentences:
        clustered_sentences[cluster_id] = []

    clustered_sentences[cluster_id].append(corpus[sentence_id])

with open("clusters.txt", "w") as f:
    for i, cluster in clustered_sentences.items():
        f.write("Cluster " + str(i+1))
        f.write("\n")
        f.write("\n".join(cluster))
        f.write("\n\n")

#%%
clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=20) #, affinity='cosine', linkage='average', distance_threshold=0.4)
clustering_model.fit(blinded_corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = {}
for sentence_id, cluster_id in enumerate(cluster_assignment):
    if cluster_id not in clustered_sentences:
        clustered_sentences[cluster_id] = []

    clustered_sentences[cluster_id].append(blinded_corpus[sentence_id])

with open("clusters_blinded.txt", "w") as f:
    for i, cluster in clustered_sentences.items():
        f.write("Cluster " + str(i+1))
        f.write("\n")
        f.write("\n".join(cluster))
        f.write("\n\n")
# %%
