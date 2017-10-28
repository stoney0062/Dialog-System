#! /usr/bin/python
# -*- coding:utf-8 -*-

from __future__ import division
import os
import gensim
import operator
import jieba
import collections
import sys
import jieba.analyse as analyse
import pickle
import glob
from gensim.models import Word2Vec
import numpy as np
import cPickle
import re
from gensim import corpora, models, similarities


reload(sys)
sys.setdefaultencoding('utf-8')

DOC_KEYWORDS = 'doc_keywords.data'
DOC_SENTS = 'doc_sents.data'
FILES = 'files.data'

clazz_map_files = None
clazz_map_docs = None
clazz_map_docskeyword = None

def readdoc(path):
    with open(path) as f:
        text = f.readlines()
    text = [x.strip() for x in text if x != '\n']
    raw_text = ''
    for para in text:
        raw_text = raw_text + para
    text = [raw_text]
#    text = sentence_seg(text)
    pro_text = process_text(text)
    return text, pro_text

def sentence_seg(text): # text = [para1, para2, \cdots]
    senten_seg = []  # senten_seg = [sen1, sen2, \cdots ]
    for item in text: # item is a string
        item = item.strip()
        senten_seg = senten_seg + cut_sentence(item)
    return senten_seg

def map_to_file(data_map,path_to_file,dim):
    with open(path_to_file,'a+') as f:
        for key in data_map.keys():
            f.write(key.strip())
            f.write('\t')
            val = data_map[key]
            if dim == 1:
                f.write(' '.join(val))
            elif dim == 2:
                f.write(','.join([' '.join(sent) for sent in val]))
            f.write('\n')


def cut_sentence(words):
    start = 0
    i = 0
    sents = []
    punt_list = '!?:;~！？：；。'.decode('utf8')
    try:
        words = words.decode('utf8')
    except:
        print words
    for word in words:
        if word in punt_list:
            sents.append(words[start:i+1])
            start = i + 1
            i = i + 1
        else:
            i = i + 1
    if start < len(words):
        sents.append(words[start:]) # to deal the end of words dont have punt
    return sents

def process_text(text):
    '''
    Here is preprocess of text , before preprocess text = [sen1, sen2, sen3, \cdots], sen is raw string.
    the preprocess : 1. word segment 2. del stop words, low frequency words, and punts.
    after preprocess text = [sen1, sen2, sen3, \cdots] sen = [word1, word2, \cdots]
    '''
    with open('tran_stopword') as f:
        stopword = f.readlines()
    stopword = [x.strip() for x in stopword if x.strip() != '']
    pro_text = []
    jieba.suggest_freq('日亚',True)
    for sentence in text:
        seg_sen = jieba.lcut(sentence)
        seg_sen = [x for x in seg_sen if x not in stopword and x != ' ']
        pro_text.append(seg_sen)
    Words = []
    for item in pro_text:
        Words = Words + item
    cout = collections.Counter(Words).most_common(300)
    dictionary = dict()
    for word,_ in cout:
        dictionary[word] = len(dictionary)
    for i in range(len(pro_text)):
        pro_text[i] = [x for x in pro_text[i] if x in dictionary.keys()]
    return pro_text

def combine_text(pro_text):
    '''
    Here is try to combine text for keywords extraction.
    pro_text = [sen1, sen2, \cdots], sen = [word1, word2, word3, \cdots]
    return combine text = 'all text of pro_text'
    '''
    combine_text = ''
    for sen in pro_text:
        for word in sen:
            combine_text = combine_text + ' ' + word
    return combine_text


def keywords_extract(text, weight = True):
    '''
    Here is process of extract keywords from text, if withWeight is Ture, return a list = [(keyword, weight), (keyword, weight), \cdots], if withWeight is False, return a list = [keyword, \cdots]
    text is a string. text=[sen1, sen2, sen3, \cdots]
    extract_tags is using tfidf method
    textrank is using textrank method
    '''
    comb_text = combine_text(text)
    return analyse.extract_tags(comb_text, withWeight = weight)


def compute_sim(sen, query, weight = True):
    '''
    Here is to compute similarity between keywords from sentence(doc) and name entity from query. 
    If weight is False, sen = [keywords, \cdots],  return the number of element which is in sen and query.
    If weight is True, sen = [(keyword, weight), \cdots] return the score = element * weight in sen.
    '''
    if weight:
        score = 0
        for item in sen:
            if item[0] in query:
                score = score + item[1]
        return score
    else:
        score = 0
        for item in sen:
            if item in query:
                score = score + 1
        return score

def find_sim_doc(clazz_map_docskeywords, query, weight = True):
    '''
    Here is try to find the doc that is most relative to the query.
    the method is compute the similarity by keywords.
    return the score of doc.
    docs = [doc1, doc2, doc3, \cdots], if weight is True, doc1 = [(keyword, weight), \cdots], if weight is False, doc1 = [keywords, \cdots]
    query = [keywords, \cdots]
    '''
    docname_map_score = {}
    print "find_sim_doc: clazz_map_docskeywords is " + str(clazz_map_docskeywords)
    for clazz in clazz_map_docskeywords.keys():
        clazz_val = clazz_map_docskeywords[clazz]
        for doc_name in clazz_val.keys():
            keywords_list = clazz_val[doc_name]
            calc_val = compute_sim(keywords_list, query, weight)
            print "calc_val is " + str(calc_val)
            docname_map_score[doc_name] = calc_val
#    doc_score = []
#    for doc in docs:
#        doc_score.append(compute_sim(doc, query, weight))
    return docname_map_score

def find_sim_sen(doc, query):
    '''
    Here is try to find the sentence in doc that is most relative to the query.
    the method is compute the similarity by keywords.
    return the score of sen.
    sen = [word1, word2, \cdots], query = [keywords, \cdots]
    '''
    sen_score = []
    for sen in doc:
        sen_score.append(compute_sim(sen, query, weight = False))
    return sen_score

"""
def answer_selection(sen, query, w2vmodel):
    
   # Here is try to select answer of query in sentence. Suppose the answer is a word!
   # sen = [word1, word2, \cdots], query = [keywords, \cdots]
   # w2vmodel is a word2vec model trained by gensim.
   
    word_score = []
    for word in sen:
        score = 0
        for keyword in query:
            if word in w2vmodel.wv and keyword in w2vmodel.wv:
                score = score + w2vmodel.wv[word].dot(w2vmodel.wv[keyword])/(np.linalg.norm(w2vmodel.wv[word]) * np.linalg.norm(w2vmodel.wv[keyword]))
        word_score.append(score)
    return [sen[i] for i in range(len(sen)) if word_score[i] == max(word_score)]


"""

#if __name__ == "__main__":
#    text , pro_text = readdoc('test.txt')
#    for item in pro_text:
#        for word in item:
#            print word
#        print '###'*5
#    model = gensim.models.Word2Vec.load('zh.bin')
#    model.save('w2v')
##    w2v = gensim.models.Word2Vec()
#    w2v = gensim.models.Word2Vec.load('w2v')
#    w2v.build_vocab(pro_text, update = True)
#    w2v.train(pro_text, total_examples=w2v.corpus_count, epochs = w2v.iter)
#    print w2v.wv[u'日亚']


def calc_docs(query_sent):
    path = 'NEW_V1/NEW_V1'
    # path = sys.argv[2]
    docs = []
    clazzs = ['1_产品使用说明文档',
              '2_产品制作工艺流程',
              '3_产品原材料需求文档',
              '4_产品外观外形设计文档',
              '5_产品相关专利文档',
              '6_产品工艺设计']
    # w2vmodel = Word2Vec.load('wiki.zh.text.simplified.vec')
    
    '''
    clazz_map_files = {}
    if os.path.exists(FILES):
        print("restore from " + FILES + "!")
        clazz_map_files = pickle.load(open(FILES,"r"))
    else:
        # files = os.listdir(path)
        for clazz in clazzs:
            clazz_map_files[clazz] = glob.glob(path + "/" + clazz +  "/*.txt")
        print clazz_map_files 
        # files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if os.path.splitext(f)[1] == '.txt' and not os.path.basename(f)[0] == '.']
        pickle.dump(clazz_map_files, open(FILES, "w"))
        print("dump success!")
    '''

    '''
    clazz_map_docs = {}
    if os.path.exists(DOC_SENTS):
        print("restore from " + DOC_SENTS + "!")
        clazz_map_docs = pickle.load(open(DOC_SENTS, "r"))
    else:
        for key in clazz_map_files: 
            files = clazz_map_files[key]
            docname_map_docsents = {}
            for docpath in files:
                if docpath.startswith('.'):
                    continue
                doc_path = docpath
                text, pro_text = readdoc(doc_path)
                docname_map_docsents[doc_path] = pro_text
            clazz_map_docs[key] = docname_map_docsents
        pickle.dump(clazz_map_docs, open(DOC_SENTS,"w"))
        print("dump success!")
    '''
    more_sen = []
    sum_docname_map_docsents = {}
    for key in clazz_map_docs.keys():
        sum_docname_map_docsents.update(clazz_map_docs[key])
    #for docname in sum_docname_map_docsents.keys():
    #    for sen in sum_docname_map_docsents[docname]:
    #        more_sen.append(sen)
    # w2vmodel.build_vocab(more_sen, update=True)
    # w2vmodel.train(more_sen, total_examples=w2vmodel.corpus_count, epochs = w2vmodel.iter)

    '''
    clazz_map_docskeyword = {}
    if os.path.exists(DOC_KEYWORDS):
        print("restore from " + DOC_KEYWORDS + "!")
        clazz_map_docskeyword = pickle.load(open(DOC_KEYWORDS,"r"))
    else:
        print "clazz_map_docskeyword is " + str(clazz_map_docskeyword)

        for key in clazz_map_docs.keys():
            # 每次处理一个类别
            docname_map_keywords = {}
            docname_map_docsents = clazz_map_docs[key]
            for docname in docname_map_docsents.keys():
                doc_key = keywords_extract(docname_map_docsents[docname], weight = True)
                # docs_keyword.append(doc_key)
                docname_map_keywords[docname] = doc_key
            # clazz_map_docskeyword[key] = docname_map_keywords

            keywords_vocab = {}
            for docname in docname_map_keywords.keys():
                key_lst = docname_map_keywords[docname]
                for m_tuple in key_lst:
                    if m_tuple[0] not in keywords_vocab:
                        keywords_vocab[m_tuple[0]] = 1
                    else:
                        keywords_vocab[m_tuple[0]] += 1
            modified_docname_map_keywords = {} 
            for docname in docname_map_keywords.keys():
                key_lst = docname_map_keywords[docname]
                modified_key_lst = []
                for m_tuple in key_lst:
                    modified_key_lst.append((m_tuple[0], m_tuple[1]/(keywords_vocab[m_tuple[0]])**2))
                modified_docname_map_keywords[docname] =modified_key_lst
                # modified_docs_keyword.append(modified_key_lst)
            clazz_map_docskeyword[key] = modified_docname_map_keywords

        pickle.dump(clazz_map_docskeyword, open(DOC_KEYWORDS,"w"))
        print("dump success!")
    '''
    # 总的docs_keyword
    docs_keyword = {}

    for key in clazz_map_docskeyword.keys():
        docs_keyword.update(clazz_map_docskeyword[key])

    # query = u'优降糖片使用说明书中规定，该药品在食用每日不能超过多少片'
    query = query_sent
    # query = sys.argv[1]
    query = analyse.extract_tags(query)
    # print docs_keyword
    docname_map_score = find_sim_doc(clazz_map_docskeyword, query, weight = True)
    docname_with_max_value = max(docname_map_score.iteritems(), key=operator.itemgetter(1))[0]
    print docname_with_max_value
    clazz_of_docname_with_max_value = docname_with_max_value.split('/')[-2]

    final_docsents = sum_docname_map_docsents[docname_with_max_value]
    sen_score = find_sim_sen(final_docsents, query)
    j = sen_score.index(max(sen_score))
    # print j
    # ans = answer_selection(final_docsents[j], query, w2vmodel)
    # print ''.join([item.encode("UTF-8") for item in final_docsents[j]])
    # print ans[0]
    # print ''.join([m_tuple[0].encode('UTF-8') + '-->' + str(m_tuple[1]) for m_tuple in docs_keyword[docname_with_max_value]])

    return '/'.join(docname_with_max_value.split('/')[-2:])

def findplace(query, path_to_file, TOP_K):
    # path = "new_train/2_产品制作工艺流程/pdf-2-47-conv.txt"
    path = "new_train/" + path_to_file
    with open(path) as f:
        doc = f.read()
        doc = re.split('。',doc)
    docs = [list(jieba.cut(d)) for d in doc]
    # print docs
    dictionary = corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=15)
    index = similarities.MatrixSimilarity(lsi[corpus])
    query_bow = dictionary.doc2bow(jieba.lcut(query))
    query_lsi = lsi[query_bow]
    sims = index[query_lsi]
    sort_sims = sorted(enumerate(sims), key=lambda item:-item[1])
    # print sort_sims
    TOP_K_answer = ""
    for item in sort_sims[:TOP_K]:
        sen = docs[item[0]]
        m_ret_answer = ""
        for w in sen:
            m_ret_answer += w
        TOP_K_answer = TOP_K_answer + m_ret_answer + "。"
    return TOP_K_answer 

def load_model():
    m_clazz_map_files = pickle.load(open(FILES,"r"))
    m_clazz_map_docs = pickle.load(open(DOC_SENTS, "r"))
    m_clazz_map_docskeyword = pickle.load(open(DOC_KEYWORDS,"r"))
    return m_clazz_map_files, m_clazz_map_docs, m_clazz_map_docskeyword 

if __name__ == "__main__":
    # Loading model ...
    clazz_map_files, clazz_map_docs, clazz_map_docskeyword = load_model()

    # Loading model complete 
    query_sentence = sys.argv[1]
    TOP_K = 1
    ret_path = calc_docs(query_sentence)
    ret_result = findplace(query_sentence, ret_path, TOP_K)

    print ret_result
