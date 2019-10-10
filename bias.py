import numpy as np
from functools import reduce
from sklearn.decomposition import PCA
import sklearn
from scipy import spatial, special
import matplotlib.pyplot as plt
import word_list


seed_pairs = word_list.return_seed_pairs()
class2words = word_list.return_class2words()
adjectives = word_list.return_adjectives()
professions = word_list.return_professions()


def fetch_seed_pairs_vocab():
	return {x for x, _ in seed_pairs} | {x for _, x in seed_pairs}


def fetch_class_words(classes):
	return set.union(*[class2words[c] for c in classes])


def fetch_adjectives():
	return adjectives


def fetch_professions():
	return professions


def compute_bias_direction(embeddings):
	g_sum = None
	diff_embeddings = [embeddings[x] - embeddings[y] for x,y in seed_pairs]
	X = np.array(diff_embeddings)
	pca = PCA(n_components=1)
	pca.fit(X)
	return pca.components_[0], pca.explained_variance_ratio_[0]


def compute_garg_euc_bias(attribute_embeddings, neutral_embeddings, A1, A2):
	a1, a2 = sum([attribute_embeddings[w] for w in A1]) / len(A1), sum([attribute_embeddings[w] for w in A2]) / len(A2)
	return sum([abs(np.linalg.norm(neutral_embeddings[w] - a1) - np.linalg.norm(neutral_embeddings[w] - a2)) for w in neutral_embeddings]) / len(neutral_embeddings)


def compute_garg_cos_bias(attribute_embeddings, neutral_embeddings, A1, A2):
	a1, a2 = sum([attribute_embeddings[w] for w in A1]) / len(A1), sum([attribute_embeddings[w] for w in A2]) / len(A2)
	return sum([abs((1 - spatial.distance.cosine(neutral_embeddings[w], a1)) - (1 - spatial.distance.cosine(neutral_embeddings[w], a2))) for w in neutral_embeddings]) / len(neutral_embeddings)


def compute_manzini_bias(attribute_embeddings, neutral_embeddings, A_list):
	a_list = [(sum([attribute_embeddings[w] for w in Ai]) / len(Ai)) for Ai in A_list]
	neutral_bias = []
	for w, e in neutral_embeddings.items():
		neutral_bias.append(abs(sum([(1 - spatial.distance.cosine(e, ai)) for ai in a_list]) / len(a_list)))
	return sum(neutral_bias) / len(neutral_bias)
### CHANGE THIS TO USE NORMALIZER, JUST DefAULTS TO SOFTMAX RIGHT NOW
def compute_normalized_averageSimilarity_bias(attribute_embeddings, neutral_embeddings, A_list, normalizer, dist, p):
    assert sum(p) == 1 and all(i>=0 for i in p) and all(i<=1 for i in p)
    mean_vectors= [(sum([attribute_embeddings[w] for w in Ai]) / len(Ai)) for Ai in A_list]
    average_similarities = []
    for mv in mean_vectors:
        average_similarities.append(sum(1 - spatial.distance.cosine(mv, neutral) for neutral in list(neutral_embeddings.values()))/len(neutral_embeddings))
    observed_distribution = special.softmax(average_similarities)
    return dist(observed_distribution, p)
    
def compute_normalized_averageBias_bias(attribute_embeddings, neutral_embeddings, A_list, normalizer, dist, p):
    assert sum(p) == 1 and all(i>=0 for i in p) and all(i<=1 for i in p)
    mean_vectors= [(sum([attribute_embeddings[w] for w in Ai]) / len(Ai)) for Ai in A_list]
    neutral_vector_sims = []
    for w, e in neutral_embeddings.items():
        neutral_vector_sims.append([1-spatial.distance.cosine(e, ai) for ai in mean_vectors])
    norm_observed_distributions = special.softmax(neutral_vector_sims, axis = 1)
    KL_scores = [dist(observed_distribution, p) for observed_distribution in norm_observed_distributions]
    return np.mean(KL_scores)

def compute_mean_difference(embeddings):
	diff_embeddings = [embeddings[x] - embeddings[y] for x,y in seed_pairs]
	return sum(diff_embeddings) / len(diff_embeddings)


def compute_mean(embeddings, words):
	return sum([embeddings[x] for x in words]) / len(words)


def compute_bias(embeddings, g, pool):
	return sum([pool(1 - spatial.distance.cosine(embeddings[w],g)) for w in embeddings]) / len(embeddings)


def compute_all_biases(attribute_embeddings, neutral_embeddings, attribute_vocab, neutral_vocab, gender_pair, classes, A1=None, A2=None, A_list=None):
	bias_bolukbasi, bias_garg_euc, bias_garg_cos, bias_manzini = None, None, None, None
	if gender_pair: # Set P exists
		g, pca_variance_explained = compute_bias_direction(attribute_embeddings)
		bias_bolukbasi = compute_bias(neutral_embeddings, g, abs)
		A1, A2 = [x for (x,y) in seed_pairs], [y for (x,y) in seed_pairs]
		A_list = [A1, A2]

	if classes == 2: # Binary bias
		bias_garg_euc = compute_garg_euc_bias(attribute_embeddings, neutral_embeddings, A1, A2)
		bias_garg_cos = compute_garg_cos_bias(attribute_embeddings, neutral_embeddings, A1, A2)
	
	bias_manzini = compute_manzini_bias(attribute_embeddings, neutral_embeddings, A_list)
	return bias_bolukbasi, bias_garg_euc, bias_garg_cos, bias_manzini
	