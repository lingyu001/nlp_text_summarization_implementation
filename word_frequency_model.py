
# coding: utf-8

# In[ ]:


# import nltk
# nltk.download('Stopwords')
# nltk.download('punkt')


# In[100]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer


# In[83]:


def create_frequency_table(text_string) -> dict:
    
    """
    input:  a paragraph as text_string
    process: tokenize text into words, stem words, remove stopwords
    output: a bag of word dictionary {word: frequency}
    
    Note: customized weight of word could be applied
    """

    stopWords = set(stopwords.words("english"))
    
    words = word_tokenize(text_string)
    ps = PorterStemmer()

    freqTable = dict()
    for word in words:
        
        word = ps.stem(word)
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1 
        else:
            freqTable[word] = 1

    return freqTable


# In[86]:


def score_sentences(sentences, freqTable) -> dict:
    
    """
    input:  list of sentences and word frequency table
    process: compute score for each sentence = total word value / word count
    output: a sentence soore dictionary {sentence: score}
    
    """   
    sentenceValue = dict()

    for sentence in sentences:
        word_count_in_sentence = (len(word_tokenize(sentence)))
        for wordValue in freqTable:
            if wordValue in sentence.lower():
                if sentence[:10] in sentenceValue:
                    sentenceValue[sentence[:10]] += freqTable[wordValue]
                else:
                    sentenceValue[sentence[:10]] = freqTable[wordValue]

        sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] // word_count_in_sentence

    return sentenceValue


# In[106]:


def find_average_score(sentenceValue) -> int:
    """
    input:  sentence score dictionary
    process: compute average sentence score = total sentence score / sentence number
    output: avreage sentence score as threshold
    
    Note: the computation ov average score can be customized / weighted
    """
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original text
    average = int(sumValues / len(sentenceValue))

    return average


# In[111]:


def generate_summary(sentences, sentenceValue, threshold) -> str:
    
    """
    input:  list of sentences, sentence value dictionary
    
    output: sentence whose score > threshold as the summary
    
    """
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] > (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary


# In[114]:


def summarize_text_wf(text) -> str:
     
    """
    input:  a paragraph of text
    
    output: summary of text according to word frequency algorithm
    
    """
    freq_table = create_frequency_table(text)
    sent = sent_tokenize(text)
    sent_value = score_sentences(sent,freq_table)
    threshold = find_average_score(sent_value)
    
    return generate_summary(sent,sent_value,threshold)

