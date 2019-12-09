
# coding: utf-8

# In[ ]:


from gensim.summarization.summarizer import summarize
#import pytextrank
#import spacy


# In[ ]:


def gensim_summarize(text, ratio = 0.2,word_count=None):
    """
    Input: a paragraph as text string, proportion of sentences as summary, or limit words as summary
    Output a summary as text string
    """
    
    return summarize(text, ratio, word_count)


# In[ ]:


def pytextrank_rank(text):
    """
    Input: a paragraph as text string
    Process: spacy pipeline including pytextrank
    Output: a spacy nlp doc
    """
    tr = pytextrank.TextRank()
    nlp = spacy.load("en")
    nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)
    doc = nlp(text)
    return doc


# In[ ]:


def pytextrank_get_rank(doc):
    """
    return the rank table of processed text
    
    """
    rank = {}
    for p in doc._.phrases:
        rank[p] = [p.rank,p.count]
    return rank


# In[ ]:


def pytextrank_get_summary(doc, n=2):
    """
    return the full sentence of the rank n key words
    
    """
    summary = ""
    for p in doc._.phrases[0:2]:
        for s in doc.sents:
            if p.text in s.text:
                summary += ''.join(s.text)
    return summary

