# Word embeddings

## Word embeddings
Word embeddings capture the relationship between words. Words that are pretty similar will have similar embeddings.
Word embeddings are very computationally expensive to train, so in general we load a pre-trained set of embeddings.
In this project, we use the GloVe 50 embedding (an array of 50 values is associated to each word). It was trained on aggregated global word-word co-occurrence statistics from a corpus. Vectors from words occuring often together are pretty similar. The word embeddings from GloVe indicate strong similarity in their context (even though they are opposite words). 


## Embedding vectors vs one-hot vectors 
One-hot vectors are simple and useful for basic tasks, but they lack semantic meaning and are computationally inefficient for large vocabularies (because high dimensionality and sparse).
Embedding vectors are much more powerful for real-world applications, as they capture the meaning and relationships between words. They are efficient and they can generalize well. They are very heavy to train, but the idea is to use a pre-trained word embedding (such as GloVe here).

Key differences between one-hot vectors and embedding vectors
| Feature | One-Hot Vectors | Embedding Vectors |
| --- | --- | --- |
| Dimensionality | High (equal to vocabulary size) | Low (typically 50, 100, or 300) |
| Representation | Sparse, binary (mostly 0's) | Dense, real-valued (continuous)|
| Semantic meaning | No semantic similarity captured | Capture semantic relationships (similar words are close) |
| Memory usage | Inefficient for large vocabularies | Efficient (low-dimensional) |
| Computation | Computationally expensive for large vocabularies | More efficient in terms of computation and storage |
| Training | Can be generated easily (static) | Requires training or pre-trained models |
| Handling of similarity | No inherent similarity between words | Similar words have similar vectors |
| Example | "cat" → [1, 0, 0] | "cat" → [0.24, -0.56, 0.89, ...] |


## Cosine similarity 
It measures the similarity between the embedding vectors for 2 words. It's equal to the cos of the angle between these 2 vectors. We compute it as the dot product of the 2 embedding vectors divided by the product of L2 norm of each word embedding. Cosine similarity is a relatively simple, intuitive and powerful method to capture nuanced relationships between words. 
The cosine similarity takes values between -1 and 1:
* a cosine similarity close to 1 indicates that both word vectors are very similar (for GLoVe embeddings,it shows that words are used in similar contexts)
* a cosine similarity close to 0 indicates that both word vectors are very dissimilar 
* a cosine similarity close to -1 indicates that both word vectors are very similar but opposite


## Word analogy
Example of a word analogy problem: man is to woman, as king is to queen. To do that, we use word embeddings.
A word analogy problem tries to find the word d such that the associated word vectors e_a, e_b, e_c and e_d are related in the following manner: e_b - e_a approximately equal to e_d - e_c.
To measure the similarity between e_b and e_a, and e_d and e_c, we use the cosine similarity.


## Debiasing word vectors
We notice that the word embeddings can be biased, in this case we examine the gender bias:
1. We substract the man embedding to the woman embedding, and do the same for the pairs girl/boy and mother/father.
2. We build the gender embedding by taking the average of the 3 previous vectors (taking the average leads to more reliable results).
3. We compare the gender vector to "not neutral words" (such as girl and boy names, and gender specific words such as actor and actress). We find that indeed the "female" words have a rather positive similarity with the gender vector. On contrary we find that the "male" words have a rather negative similarity with the gender vector.
4. We compare the gender vector to "neutral words". We find that some words (such as lipstick and receptionnist) have a rather positive similarity with the gender vector (meaning these words are closer in values to "female" words) and other words (such as warrior and guns) have a rather negative similarity with the gender vector (meaning these words are closer in values to "male" words).
Later in the project, we are going to unbias the word embeddings so that all neutral words are not linked to gender, but the gender specific words shouldn't be unbiased (they should only be equalized, meaning being equidistant from neutral gender words).


## Neutralize bias for non-gender specific words
For non-gender specific words, we want to neutralize the gender bias. Indeed, we don't want a receptionnist to sound more like female or a warrior to sound more like male. 
In this case we work with a 50 dimensional word embedding, the 50 dimensional space can be split into:
* the bias-direction g (for gender here)
* the remaining 49 dimensions, which is called g_perp here. This 49 dimensional space is orthognal to g.

We transform the word vectors such that they have a length of 1. It simplifies the mathematical process, it ensures that the focus remains on adjusting the direction (not the length) of the vectors, and it leads to more standardized and comparable results.

The neutralization step takes a vector (such as e_receptionist) and zeros out the component in the direction of g (giving us e_receptionist_debiased). It removes the gender bias of "receptionnist" by projecting the word vector on the space orthogonal to the gender bias axis. It ensures that gender neutral words are zero in the gender subspace. 
We get the unbiased component after 2 steps:
1. We compute the bias component
2. We remove the bias component from the original embedding: e - ebias_component


## Equalization algorithm for gender-specific words
Some words (such as actess vs actor or policeman vs policewoman) should be gender specific. But we need to check that they are equally distributed from the gender embedding. The key idea behind equalization is to make sure that a particular pair of words are equidistant from the 49-dimensional g_perp.