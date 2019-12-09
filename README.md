# nlp_text_summarization_implementation

Inculde 3 module: 
word frequency model, text rank graph model, Kmean clustering using BERT sentence embedding

Please chekc jupyter notebook test_report.v2.ipynb for test report

1. Opinosis Dataset with golden human written summary for test
2. ROUGH 1-gram F1-score for test
3. Word Frequency model implementation
4. Text rank graph model (Gensim package)
5. BERT sentence embedding and Kmeans clustering algorithm

Lingyu Li 
12/9/2019 update

## 1. Opinosis Dataset
    Contain 51 paragraphs of user reviews on a given topic, obtained from Tripdvisor, Edmunds and Amazon
    Each paragraph contains 100 sentence in average
    Data file also contains gold standard summaries of each paragraph for test and validation
    https://kavita-ganesan.com/opinosis-opinion-dataset/#.Xe6Ya-hKhPY
    https://github.com/kavgan/opinosis-summarization


## 2. Rouge Score for text summarization

#### Metric: Rouge score (Recall-Oriented Understudy for Gisting Evaluation)
 - Rouge N: measure N-gram overlap between model output summary and reference summary
 - Rouge L: measures longest matching sequence of words using LCS(longest Common Subsequence)

#### Rouge score is composed of: 
- Precision = # of overlapping words / total words in the reference summary
- Recall = # of overlapping words / total workds in the model generated summary
- F1-score

#### Interpretation:
- ROUGE-n recall=40% : 40% of the n-grams in the reference summary are also present in the generated summary.
- ROUGE-n precision=40% : 40% of the n-grams in the generated summary are also present in the reference summary.
- ROUGE-n F1-score=40% is like any F1-score.


## 3. Word Frequency Algorithm
#### Steps
    a. Bag of words based algorithm
    b. compute word frequency
    c. score each sentence according to word frequency (can be weighted)
    d. generate threshold of sentence selection (average score, etc.)
    e. Selected sentence (score > threshold) as summary

## 4. TextRank Algorithms
    #### Graph based algorithm Basic steps

    a. Cleaning Text (remove punctuation, Stopwords, stemming)
    b. Vector representation of sentences: This part can be customized by using different pre-trained vectorization models or train your own model
    c. Use cosine similarity find similarity of sentences
    d. Apply PageRank algorithm: use networkx(networkx.score) to rank sentences
    e. Extract top N sentences as summary
    Skip implementation, there are >3 existing packages using graph

## 5. Kmean clustering of sentence embedding using Bert

### 5. 1 Impelemtation leveraging Bert Pretrained pytorch Model
    https://pypi.org/project/pytorch-pretrained-bert/

Implementation
    Step 1: Tokenize paragraph into sentences
    Step 2: Format each sentence as Bert input format, and Use Bert tokenizer to tokenize each sentence into words
    Step 3: Call Bert pretrained model, conduct word embedding, obtain embeded word vector for each sentence.(The Bert output is a 12-layer latent vector) 
    Step 4: Decide how to use the 12-layer latent vector: 
    1) Use only the last layer; 
    2) Average all or last 4 layers, and more...
    Step 5: Apply pooling strategy to obtain sentence embedding from word embedding, eg. mean, max of all word vector
    Step 6: Obtain sentence vector for each sentence in the paragraph, apply Kmeans, Gaussian Mixture, etc to cluster similar sentence
    Step 7: Return the closest sentence to each centroid (euclidean distance) as the summary, ordered by appearance


### 6. Test and Compare using Opinosis Dataset

Mean ROUGE 1-gram F1-score:

Word Frequency model: 0.100
Text Rank model: 0.091
Kmeans clustering BERT: 0.116

It seems the clustering summary using BERT embedding is slightly better than word frequency and text rank model summary!


Future work could be considered to
1) Try out different BERT layers to produce the latent vectors (word embedding)
2) Try different pooling strategy from word vector to sentence vectors
3) Some other clustering method
Use supervise learning to fine tune BERT model for summarization purpose could be another topic to develop
    https://arxiv.org/pdf/1903.10318.pdf
