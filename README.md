# BiasQuant
Methods quantifying bias in word embeddings  

**Word Embedding Data**  
  Word embeddings can be found here:  
  GloVe: https://nlp.stanford.edu/projects/glove/  
  Word2Vec: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit  
  
  
  Using the words in seed_pairs as the attribute words and the professions as neutral words, the bias calculated using softmax normalization, p being the uniform distribution, and div being stats.entropy which calculates KL divergence, the two scores I got were: 
  
  
with calculating bias by: for each attribute word, calculate average similarity to all neutral words, normalize distribution, etc. this was 0.00010202112675212344

With calculating bias by: for each neutral word, calculate similarity to the attribute words. This gives us a distriubtion for each neutral word. Normalize each of these and calculat the KL divergence for each. Average them. This gives us 0.001055579608182993
