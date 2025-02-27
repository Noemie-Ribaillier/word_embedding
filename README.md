# Word embedding

## Word embedding
Word embeddings capture relationships between words.
Word embeddings are very computationally expensive to train, so in general we load a pre-trained set of embeddings.
In this project we use the GloVe embedding, which was trained on aggregated global word-word co-occurrence statistics from a corpus. Vectors from words occuring often together are pretty similar. the word embeddings from GloVe indicate strong similarity in their context, even though they are opposite words, because the word embeddings are learned based on statistical co-occurence patterns.

## Embedding vectors vs one-hot vectors 
One-hot vectors don't do a good job of capturing the level of similarity between words because every one-hot vector has the same Euclidean distance from any other one-hot vector.
Embedding vectors (for example GloVe vectors) provide much more useful information about the meaning of individual words.

## Cosine similarity 
It measures the similarity between embedding vectors for 2 words. It's equal to the cos of the angle between these 2 vectors. We compute it as the dot product of the 2 embedding vectors divided by the product of L2 norm of each word embedding. Cosine similarity is a relatively simple and intuitive, yet powerful, method you can use to capture nuanced relationships between words. 
The cosine similarity indicates:
* if both word vectors are very similar, the cosine similarity will be close to 1
* if both word vectors are very dissimilar, the cosine similarity will be close to 0
* if both word vectors are contrary (similar but opposite), the cosine similarity will be close to -1

## Word analogy
Example of a word analogy problem: man is to woman, as king is to queen. To do that, we use word embeddings.
A word analogy problem tries to find the word d such that the associated word vectors e_a, e_b, e_c and e_d are related in the following manner: e_b - e_a approximately equal to e_d - e_c.
To measure the similarity between e_b and e_a, we use cosine similarity.

## Debiasing word vectors
We notice that the word embeddings can be biased, in this case we examine the gender biases. 
1st we build a gender embedding by sbstracting the man embedding to the woman embedding.
For example, if we build a gender embedding and compare it to some (neutral vectors such as receptionist or warrior) we find that receptionist is rather linked to female and warrior linked to male.
So we want to unbiased the word embeddings so that all neutral words are not linked to gender.
We end up with modified (unbiased) embeddings to reduce the gender bias.

## Neutralize bias for non-gender specific words
For non-gender specific words, we want to neutralize the gender bias. Indeed, we don;t want a receptionnist to sound more like female or a warrior to sound more like male. 
In this case we work with a 50 dimensional word embedding, the 50 dimensional space can be split into:
* the bias-direction g (for gender here)
* the remaining 49 dimensions, which is called g_perp here. This 49 dimensional space is orthognal to g.
The neutralization step takes a vector such as $e_receptionist and zeros out the component in the direction of g, giving us e_receptionist_debiased. We get the unbiased component after 2 steps:
1. We compute the bias component
2. We remove the bias component to the original emebdding: e - ebias_component

## Equalization algorithm for gender-specific words
Some words (such as actess vs actor or policeman vs policewoman) should be gender specific. But we need to check that they are equally distributed from the gender embedding. The key idea behind equalization is to make sure that a particular pair of words are equidistant from the 49-dimensional g_perp.
