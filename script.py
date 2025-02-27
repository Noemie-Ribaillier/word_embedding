
##########################################################################################################
#####                                                                                                #####
#####                                   OPERATIONS ON WORD VECTORS                                   #####
#####                                     Created on 2025-02-26                                      #####
#####                                     Updated on 2025-02-27                                      #####
#####                                                                                                #####
##########################################################################################################

##########################################################################################################
#####                                            PACKAGES                                            #####
##########################################################################################################

# Clear the whole environment
globals().clear()

# Load the libraries
import numpy as np
from tqdm import tqdm

# Set up the right directory
import os
os.chdir('C:/Users/Admin/Documents/Python Projects/_operations')


##########################################################################################################
#####                                     LOAD THE WORD VECTORS                                      #####
##########################################################################################################

# Create the function to read the Glove word embedding
def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding="utf8") as f:
        words = set()
        word_to_vec_map = {}
        
        # Read per line and extract 1st the word then the embedding vector
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
            
    return words, word_to_vec_map

# We load the pre-trained word embeddings (here GloVe embeddings which is 50-dimensional GloVe vectors to represent words)  
words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

# words: set of words in the vocabulary (400k words)
type(words) ; len(words) ; list(words)[0]

# word_to_vec_map: dictionary mapping words to their GloVe vector representation 
# (dictionnary with key being the words and values being a numpy array of 50 values representing the words in a continuous vector space, where semantically similar words are close to each other)
type(word_to_vec_map) ; len(word_to_vec_map) ; word_to_vec_map[list(words)[0]]

# For example, with a quick look at the word embedding of man vs woman vs apple, we see that the numbers seem to be positive/negative the same way between woman and man while different with apple
word_to_vec_map['man']
word_to_vec_map['woman']
word_to_vec_map['apple']


##########################################################################################################
#####                                       COSINE SIMILARITY                                        #####
##########################################################################################################

# Implement cosine_similarity function to evaluate the similarity between 2 word vectors
def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similarity between 2 vectors (words embedding)
        
    Arguments:
        u -- a word vector of shape (n,)          
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v
    """
    
    # Special case
    if np.all(u == v):
        return 1

    # Compute the dot product between u and v
    dot = np.dot(u,v)
    
    # Compute the L2 norm of u
    norm_u = np.linalg.norm(u)
    
    # Compute the L2 norm of v
    norm_v = np.linalg.norm(v)
    
    # Avoid division by 0
    if np.isclose(norm_u * norm_v, 0, atol=1e-32):
        return 0
    
    # Compute the cosine similarity
    cosine_similarity = dot/(norm_u*norm_v)
    
    return cosine_similarity


# Check with some examples, take some words and their word embedding
father = word_to_vec_map["father"]
mother = word_to_vec_map["mother"]
ball = word_to_vec_map["ball"]
crocodile = word_to_vec_map["crocodile"]
france = word_to_vec_map["france"]
italy = word_to_vec_map["italy"]
paris = word_to_vec_map["paris"]
rome = word_to_vec_map["rome"]

# Get the cosine similarity between 2 words
print("cosine_similarity(father, mother) = ", cosine_similarity(father, mother))
print("cosine_similarity(ball, crocodile) = ",cosine_similarity(ball, crocodile))
print("cosine_similarity(france - paris, rome - italy) = ",cosine_similarity(france - paris, rome - italy))
# We notice that cosine similarity between father and mother is higher than the cosine similarity between ball and crocodile
# This is because father and mother are rather more used in the same context, than ball and crocodile
# Cosine similarity between France and Paris vs Rome vs Italy is negative because they are similar but opposite

##########################################################################################################
#####                                       WORD ANALOGY TASK                                        #####
##########################################################################################################

# Create the function to perform word analogies (finding a word d such that the associated word vectors 
# e_a, e_b, e_c, e_d are related in the following manner e_b - e_a approx equal to e_d - e_c)
def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task as explained above: a is to b as c is to ____. 
    
    Arguments:
    word_a -- a word, string
    word_b -- a word, string
    word_c -- a word, string
    word_to_vec_map -- dictionary that maps words to their corresponding vectors. 
    
    Returns:
    best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
    """
    
    # Convert words to lowercase
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    
    # Get the word embeddings e_a, e_b and e_c
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]
    
    # Get all the possible words, define a big max_cosine_sim (that we will aim at reducing as much as possible)
    words = word_to_vec_map.keys()
    max_cosine_sim = -100
    best_word = None
    
    # Loop over the whole words vector set
    for w in words:   
        # To avoid best_word being one the input words, skip the input word_c
        if w == word_c:
            continue
        
        # Compute cosine similarity between the vector (e_b - e_a) and the vector ((w's vector representation) - e_c)
        cosine_sim = cosine_similarity((e_b - e_a),(word_to_vec_map[w] - e_c))
        
        # If the cosine_sim is more than the max_cosine_sim seen so far, we set the new max_cosine_sim to the current cosine_sim and the best_word to the current word
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w
        
    return best_word


# Get the analogy for the following tasks
triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), 
                 ('small', 'smaller', 'large'), ('carrot', 'orange', 'cucumber')] 
for triad in triads_to_try:
    print ('{} -> {} :: {} -> {}'.format(*triad, complete_analogy(*triad, word_to_vec_map)))
# * character is used to unpack the tuple
# The word analogy doesn't get that if carrot is orange, cucumber should be green


##########################################################################################################
#####                                     DEBIASING WORD VECTORS                                     #####
##########################################################################################################

# Examine the gender biases that can be reflected in a word embedding, and explore algorithms for reducing the bias.

# Get the gender word embedding (gender vector)
g = word_to_vec_map['woman'] - word_to_vec_map['man']
# g1 = word_to_vec_map['mother'] - word_to_vec_map['father']
# g2 = word_to_vec_map['girl'] - word_to_vec_map['boy']
# g3 = word_to_vec_map['woman'] - word_to_vec_map['man']
# g = np.mean(np.array([g1,g2,g3]),axis=0)
print(g)

# Compute now the cosine similarity between the g vector (gender) and other words

# Get the smilarity between each name (girls and boys) and the gender vector
name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin']
for w in name_list:
    print (w, cosine_similarity(word_to_vec_map[w], g))
# We see that girl names have a positive cosine similarity while boy names have negative cosine similarity
 
# Try with other words (that should be gender neutral but that are victim of stereotypes in our society)
word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior','doctor', 'tree', 'receptionist', 
             'technology',  'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
for w in word_list:
    print (w, cosine_similarity(word_to_vec_map[w], g))
# We see that some words (supposed to be neutral) have a positive cosine similarity with gender (for example receptionnist)
# so are closer in value to female first names while other words (for example warrior) have a negative cosine similarity
# (meaning that warrior is closer in value to male first names)
# This shows the stereotypes from our society. Goal of this is to solve the gender and other stereotypes so that our models is more neutral

# Try with other words (that sare not gender neutral)
word_list = ['actor', 'actress', 'grandmother', 'grandfather', 'policeman', 'policewoman', 'waiter', 'waitress']
for w in word_list:
    print (w, cosine_similarity(word_to_vec_map[w], g))
# These pairs should remain gender-specific


##########################################################################################################
#####                         NEUTRALIZE BIAS FOR NON-GENDER SPECIFIC WORDS                          #####
##########################################################################################################

# In this project we are using a 50 dimensional word-embedding, the 50 dimensional space can be split into 2 parts:
# * the bias direction (g here)
# * the remaining 49 dimensions (orthogonal to g) [called g_perp]

# The neutralization step takes a vector such as e_receptionist and zeros out the component in the direction
#  of g, giving us e_receptionist_debiased. 

# The theory assumes all word vectors to have L2 norm as 1
word_to_vec_map_unit_vectors = {
    # it divides each component of the word vector by its L2 norm (the resulting vector will point in the same direction as the original vector but will have a length of 1)
    word: embedding / np.linalg.norm(embedding)
    for word, embedding in word_to_vec_map.items()
}
g_unit = word_to_vec_map_unit_vectors['woman'] - word_to_vec_map_unit_vectors['man']

# Create neutralize function to remove the bias of words such as "receptionist" or "scientist" 
def neutralize(word, g, word_to_vec_map):
    """
    Removes the bias of "word" by projecting it on the space orthogonal to the bias axis. 
    This function ensures that gender neutral words are zero in the gender subspace.
    
    Arguments:
        word -- string indicating the word to debias
        g -- numpy-array of shape (50,), corresponding to the bias axis (such as gender)
        word_to_vec_map -- dictionary mapping words to their corresponding vectors.
    
    Returns:
        e_debiased -- neutralized word vector representation of the input "word"
    """
    
    # Select word vector representation of "word"
    e = word_to_vec_map[word]
    
    # Compute e_biascomponent using the formula given above
    e_biascomponent = (np.dot(e,g)/(np.linalg.norm(g))**2)*g
 
    # Neutralize e by subtracting e_biascomponent from it (e_debiased should be equal to its orthogonal projection)
    e_debiased = e-e_biascomponent
    
    return e_debiased


# Neutralize the word embedding of the word receptionist
word = "receptionist"
print("cosine similarity between " + word + " and g, before neutralizing: ", cosine_similarity(word_to_vec_map[word], g))
e_debiased = neutralize(word, g_unit, word_to_vec_map_unit_vectors)
print("cosine similarity between " + word + " and g_unit, after neutralizing: ", cosine_similarity(e_debiased, g_unit))


##########################################################################################################
#####                        EQUALIZATION ALGORITHM FOR GENDER-SPECIFIC WORDS                        #####
##########################################################################################################

# Debiasing can also be applied to word pairs such as "actress" and "actor".
# Equalization is applied to pairs of words that you might want to have differ only through the gender property.
# To make sure that a particular pair of words are equidistant from the 49-dimensional g_perp. 
# The equalization step also ensures that the two equalized steps are now the same distance from any other work that has been neutralized (for example e_receptionist_debiased). 

# Create the equalize function
def equalize(pair, bias_axis, word_to_vec_map):
    """
    Debias gender specific words by following the equalize method
    
    Arguments:
    pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor") 
    bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
    word_to_vec_map -- dictionary mapping words to their corresponding vectors
    
    Returns
    e_1 -- word vector corresponding to the first word
    e_2 -- word vector corresponding to the second word
    """
    
    # Step 1: Select word vector representation of "word"
    w1, w2 = pair
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]
    
    # Step 2: Compute the mean of e_w1 and e_w2
    mu = (e_w1 + e_w2)/2

    # Step 3: Compute the projections of mu over the bias axis and the orthogonal axis
    mu_B = (np.dot(mu,bias_axis)/(np.linalg.norm(bias_axis)**2))*bias_axis
    mu_orth = mu - mu_B
    
    # Step 4: Use equations (7) and (8) to compute e_w1B and e_w2B
    e_w1B = (np.dot(e_w1,bias_axis)/(np.linalg.norm(bias_axis)**2))*bias_axis
    e_w2B = (np.dot(e_w2,bias_axis)/(np.linalg.norm(bias_axis)**2))*bias_axis

    # Step 5: Adjust the bias part of e_w1B and e_w2B using the formulas (9) and (10) given above
    corrected_e_w1B = np.sqrt(1-np.linalg.norm(mu_orth)**2)*(e_w1B-mu_B)/(np.linalg.norm(e_w1B-mu_B))
    corrected_e_w2B = np.sqrt(1-np.linalg.norm(mu_orth)**2)*(e_w2B-mu_B)/(np.linalg.norm(e_w2B-mu_B))

    # Step 6: Debias by equalizing e1 and e2 to the sum of their corrected projections
    e1 = corrected_e_w1B + mu_orth
    e2 = corrected_e_w2B + mu_orth
                                                                    
    return e1, e2


# Equalize the word embedding of man and woman
print("cosine_similarity between man and gender = ", cosine_similarity(word_to_vec_map["man"], g))
print("cosine_similarity between woman and gender = ", cosine_similarity(word_to_vec_map["woman"], g))
e1, e2 = equalize(("man", "woman"), g_unit, word_to_vec_map_unit_vectors)
print("cosine_similarity between man equalized e1 and gender = ", cosine_similarity(e1, g_unit))
print("cosine_similarity between woman equalized e2 and gender = ", cosine_similarity(e2, g_unit))

# Equalize the word embedding of actor and actress
print("cosine_similarity between actor and gender = ", cosine_similarity(word_to_vec_map["actor"], g))
print("cosine_similarity between actress and gender = ", cosine_similarity(word_to_vec_map["actress"], g))
e1, e2 = equalize(("actor", "actress"), g_unit, word_to_vec_map_unit_vectors)
print("cosine_similarity between actor equalized e1 and gender = ", cosine_similarity(e1, g_unit))
print("cosine_similarity between actress equalized e2 and gender = ", cosine_similarity(e2, g_unit))
