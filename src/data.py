
import bcolz
import pickle
import re
import os
import random

import pandas as pd
import numpy as np

from nltk import word_tokenize
from nltk import sent_tokenize
from nltk import bigrams
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

from sklearn.preprocessing import MinMaxScaler

MAX_WORDS = 1000
MAX_PARAGRAPHS = 52



def load_folds(fn = "data/OrganizationFolds.txt", id2idx = {}):
    
    return [[id2idx.get(x, x) for x in v.strip().split('\n')] for f, v in re.findall("^Fold ([1-5]):\n([A-Z0-9\n]+)$", open(fn).read(), flags=re.DOTALL|re.MULTILINE)]

    
def load_pretrained_embeddings(dir_vectors = "/home/mim/"):
    embedding_matrix = bcolz.open(os.path.join(dir_vectors, "En_vectors.dat"))[:]
    word2idx = pickle.load(open(os.path.join(dir_vectors, "En_vectors_idx.pkl"), 'rb'))
    words = pickle.load(open(os.path.join(dir_vectors, "En_vectors_words.pkl"), 'rb'))

    return {w: embedding_matrix[word2idx[w]] for w in words}


def load_discourse_indicators(fn_di = "/home/mim/DI_wo_and.txt"):
    file = open(fn_di)
    data = file.read()
    data = data.splitlines()

    lowered_list = [i.lower() for i in data]
    di_list = [i.split() for i in lowered_list]
    
    return sorted(di_list, key=len, reverse=True)


def load_annotated_essay(fn_essays):
    df_ic = pd.ExcelFile(fn_essays)
    df = df_ic.parse('Sheet1')

    return df["Essay Number"], np.array(df["essay"]), np.array(df["Organization"]), np.array(df["Prompt"])


def load_annotated_essay_with_normalized_score(fn_essays, score_source = "./data/OrganizationScores.txt"):
    """
    Getting normalized score and making a dataframe
    """
    
    df_ic = pd.ExcelFile(fn_essays)
    df = df_ic.parse('Sheet1')
    
    # Add scores.
    df_x = pd.read_csv(score_source, delimiter="\t", header=None)
    df_x.columns = "essay_id score".split()
    
    # Special treatment for organization scores (as it contains mutiple scores)
    if "OrganizationScores" in score_source:
        df_x.score = [float(x.split(",")[0]) for x in df_x.score.values]

    def get_score(x):
        q  = df_x[df_x.essay_id == x["Essay Number"]].score.values
        return q[0] if len(q) > 0 else None
        
    df['score'] = df.apply(get_score, axis=1)
    df = df[pd.notna(df.score)]
    
    sc = MinMaxScaler()
    sc.fit(df.score.values.reshape((-1, 1)))
    df['n_score'] = sc.transform(df.score.values.reshape((-1, 1)))
    
    return df["Essay Number"], np.array(df.essay), np.array(df.score), np.array(df.n_score), np.array(df.Prompt), sc


def load_essay(fn_essays):
    df_all = pd.read_csv(fn_essays)
    essay_all = df_all.essay
    essay_all = essay_all.tolist()
    
    tokenizer = RegexpTokenizer(r'\w+|\n')

    refined_essay = []

    for e in essay_all:
        if len(tokenizer.tokenize(e)) <= MAX_WORDS:
            refined_essay.append(e)
        
    return refined_essay


def get_fold(folds, i):
    pattern = [
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 0],
        [2, 3, 4, 0, 1],
        [3, 4, 0, 1, 2],
        [4, 0, 1, 2, 3],
    ]
        
    tr, ts = folds[pattern[i][0]] + folds[pattern[i][1]] + folds[pattern[i][2]] + folds[pattern[i][3]], folds[pattern[i][4]]
    
    random.seed(33)
    random.shuffle(tr)
    trsz = int(len(folds[pattern[i][0]])*3.5)
    
    return (tr[:trsz], tr[trsz:], ts)

def essay_with_minimum_words(essay_all):
    
    tokenizer = RegexpTokenizer(r'\w+')
    refined_essay = []
    
    for e in essay_all:
        if len(tokenizer.tokenize(e)) <= MAX_WORDS:
            refined_essay.append(e)
        
    return refined_essay

def preprocess_essay_encoder(essay_list, args, di_list=None, boseos=False):
    ret = []
    
    for e in essay_list:
        e = re.sub(r'\t', '', e)
        e = re.sub(r'\n', '', e)
        e = e.lower()
        if args.mp_punct:
            e = ' '.join(word_tokenize(e))

        if di_list != None:
            e = find_replace_di(e, di_list)
        
        if boseos:
            e = ["BOS {} EOS".format(x) for x in sent_tokenize(e)]
            
        ret += [e]

    return np.array(ret)

def preprocess_essay_encoder_wPara(essay_list, args, di_list=None, boseos=False):
    ret = []
    
    for e in essay_list:
        e = re.sub(r'\t', '', e)
        e = e.lower()
        e = re.sub(r'\n\n', '\n', e)
        e = re.sub(r'\n', ' MMM ', e)
        if args.mp_punct:
            e = ' '.join(word_tokenize(e))

        if di_list != None:
            e = find_replace_di(e, di_list)
        
        if boseos:
            e = ["BOS {} EOS".format(x) for x in sent_tokenize(e)]
            
        ret += [e]

    return np.array(ret)
    
    
def preprocess_essay(essay_list, args, di_list=None, boseos=False):
    ret = []
    tokenizer = RegexpTokenizer(r'\w+')
    
    n=' '+'MMM'+' ' 
    
    for e in essay_list:
        e = re.sub(r'\t', '', e)
        e = e.lower()
            
        e = re.sub(r'\n', ' MMM ', e)
        
        if args.mp_punct:
            e = ' '.join(word_tokenize(e))
            
        if di_list != None:
            e = find_replace_di(e, di_list)
        
        e = re.split('MMM',e)[1:-1]
       
        
        essay =[]

        for i in e:
    
            nn = (["BOS {} EOS ".format(x) for x in(sent_tokenize(i))])
            essay.append(nn)
        
        all_essay = []
        
        for i in essay:
            i.append('EOP')
            for n, k in enumerate(i):
                if not args.mp_punct:
                    k = ' '.join(tokenizer.tokenize(k))
                all_essay.append(k)

        e = all_essay
        
        if boseos:
            e = e
            
        ret += [e]

    return np.array(ret)

    
def shuffled_essay(essay_list):
    shuffle_essay = []
    segmented_essay = []

    for i in essay_list:
        s = ["BOS {} EOS".format(x) for x in sent_tokenize(i)]
        segmented_essay+= [" ".join(s)]

        np.random.shuffle(s)
        nn = " ".join(s)

        shuffle_essay+=[nn]

    return segmented_essay, shuffle_essay

def paragraph_shuffled_essay(essay_list):
    
    shuffle_essay = []
    segmented_essay = []
    
    tokenizer = RegexpTokenizer(r'\w+')

    for e in essay_list:
        
        emp = re.split('MMM',e)[1:-1]
        
        essay =[]

        for i in emp:
    
            nn = (["BOS {} EOS ".format(x) for x in(sent_tokenize(i))])
            essay.append(nn)
        
        all_essay = []
        
        for i in essay:
            i.append('EOP')
            for n, k in enumerate(i):
                all_essay.append(k)

        e = all_essay
            
        segmented_essay += [e]
        
        np.random.shuffle(emp)
        
        essay =[]

        for i in emp:
    
            nn = (["BOS {} EOS ".format(x) for x in(sent_tokenize(i))])
            essay.append(nn)
        
        all_essay = []
        
        for i in essay:
            i.append('EOP')
            for n, k in enumerate(i):
                all_essay.append(k)

        e = all_essay
        
        shuffle_essay += [e]
        
        
    return segmented_essay, shuffle_essay


def di_shuffled_essay(essay_list, di_list):
    shuffle_essay = []
    segmented_essay = []
    
    for i in essay_list:
        # Replace discourse indicators with the predefined list.
        s = ["BOS {} EOS".format(x) for x in sent_tokenize(i)]
        s = " ".join(s)
        i_di = find_replace_di(s, di_list)

        # Shuffle the discourse indicators in the essay.
        i_di_shuf = di_change(i_di)

        segmented_essay.append(s)
        shuffle_essay.append(i_di_shuf)

    return segmented_essay, shuffle_essay


def find_replace_di(essay, di_list):
    
    # Query will look like: "because|in order to"
    regex_di = "|".join([" ".join(di) for di in di_list])
    
    def _rep(m):
        return r" DI_{} ".format(m.group(2).replace(" ", "_"))
        
    essay = re.sub(
        "(^| )(" + regex_di + ") ",
        _rep,
        essay)

    return essay

def find_replace_di_m(essay, di_list):
    for di in di_list:
        if len(di) == 9:
            w0 = di[0]
            w1 = di[1]
            w2 = di[2]
            w3 = di[3]
            w4 = di[4]
            w5 = di[5]
            w6 = di[6]
            w7 = di[7]
            w8 = di[8]
            
            da = 'DI'+'_'+w0+'_'+w1+'_'+w2+'_'+w3+'_'+w4+'_'+w5+'_'+w6+'_'+w7+'_'+w8
            
            if bool(re.search(r'\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b'% 
                             (w0, w1, w2, w3, w4, w5, w6, w7, w8), essay)) == True:

                essay = re.sub(r'\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b'% 
                                 (w0, w1, w2, w3, w4, w5, w6, w7, w8), da, essay)
            
        elif len(di) == 8:
            w0 = di[0]
            w1 = di[1]
            w2 = di[2]
            w3 = di[3]
            w4 = di[4]
            w5 = di[5]
            w6 = di[6]
            w7 = di[7]
            
            da = 'DI'+'_'+w0+'_'+w1+'_'+w2+'_'+w3+'_'+w4+'_'+w5+'_'+w6+'_'+w7
            
            if bool(re.search(r'\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b'% 
                             (w0, w1, w2, w3, w4, w5, w6, w7), essay)) == True:

                essay = re.sub(r'\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b'% 
                                 (w0, w1, w2, w3, w4, w5, w6, w7), da, essay) 
            
        elif len(di) == 7:
            w0 = di[0]
            w1 = di[1]
            w2 = di[2]
            w3 = di[3]
            w4 = di[4]
            w5 = di[5]
            w6 = di[6]
            
            da = 'DI'+'_'+w0+'_'+w1+'_'+w2+'_'+w3+'_'+w4+'_'+w5+'_'+w6
            
            if bool(re.search(r'\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b'% 
                             (w0, w1, w2, w3, w4, w5, w6), essay)) == True:

                essay = re.sub(r'\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b'% 
                                 (w0, w1, w2, w3, w4, w5, w6), da, essay)
            
        elif len(di) == 6:
            w0 = di[0]
            w1 = di[1]
            w2 = di[2]
            w3 = di[3]
            w4 = di[4]
            w5 = di[5]
            
            da = 'DI'+'_'+w0+'_'+w1+'_'+w2+'_'+w3+'_'+w4+'_'+w5
            
            if bool(re.search(r'\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b'% 
                             (w0, w1, w2, w3, w4, w5), essay)) == True:

                essay = re.sub(r'\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b'% 
                                 (w0, w1, w2, w3, w4, w5), da, essay)
            
        elif len(di) == 5:
            w0 = di[0]
            w1 = di[1]
            w2 = di[2]
            w3 = di[3]
            w4 = di[4]
            
            da = 'DI'+'_'+w0+'_'+w1+'_'+w2+'_'+w3+'_'+w4
            
            if bool(re.search(r'\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b'% 
                             (w0, w1, w2, w3, w4), essay)) == True:

                essay = re.sub(r'\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b'% 
                                 (w0, w1, w2, w3, w4), da, essay)
            
        elif len(di) == 4:
            w0 = di[0]
            w1 = di[1]
            w2 = di[2]
            w3 = di[3]
            
            da = 'DI'+'_'+w0+'_'+w1+'_'+w2+'_'+w3
            
            if bool(re.search(r'\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b'% 
                             (w0, w1, w2, w3), essay)) == True:

                essay = re.sub(r'\b%s\b\s\b%s\b\s\b%s\b\s\b%s\b'% 
                                 (w0, w1, w2, w3), da, essay)
            
        elif len(di) == 3:
            w0 = di[0]
            w1 = di[1]
            w2 = di[2]
            
            da = 'DI'+'_'+w0+'_'+w1+'_'+w2
            
            if bool(re.search(r'\b%s\b\s\b%s\b\s\b%s\b'% 
                             (w0, w1, w2), essay)) == True:

                essay = re.sub(r'\b%s\b\s\b%s\b\s\b%s\b'% 
                                 (w0, w1, w2), da, essay)
            
        elif len(di) == 2:
            w0 = di[0]
            w1 = di[1]
            
            da = 'DI'+'_'+w0+'_'+w1
            
            if bool(re.search(r'\b%s\b\s\b%s\b'% 
                             (w0, w1), essay)) == True:

                essay = re.sub(r'\b%s\b\s\b%s\b'% 
                                 (w0, w1), da, essay)
            
        elif len(di) == 1:
            w0 = di[0]
            
            da = 'DI'+'_'+w0
            
            if bool(re.search(r'\b%s\b'% 
                             (w0), essay)) == True:

                essay = re.sub(r'\b%s\b'% 
                                 (w0), da, essay)
                
    return essay


def di_change(essay):
    
    tks = [tk for tk in essay.split(" ")]
    dis = [tk for tk in tks if tk.startswith("DI_")]
    
    random.shuffle(dis)
    
    new_tks = [dis.pop()[3:].replace("_", " ") if tk.startswith("DI_") else tk for tk in tks]

    return " ".join(new_tks)


def create_training_data_for_shuffled_essays(refined_essay):
    essay_orig, essay_shf = shuffled_essay(refined_essay)
    total_essay = essay_orig + essay_shf
    
    # The original essays work as positive examples (label: 1), while the shuffled essays work as negative examples (label: 0)
    scores = [1] * len(refined_essay) + [0] * len(refined_essay)
    
    return np.array(total_essay), scores

def create_training_data_for_paragraph_shuffled_essays(refined_essay):
    essay_orig, essay_shf = paragraph_shuffled_essay(refined_essay)
    total_essay = essay_orig + essay_shf
    
    # The original essays work as positive examples (label: 1), while the shuffled essays work as negative examples (label: 0)
    scores = [1] * len(refined_essay) + [0] * len(refined_essay)
    
    return np.array(total_essay), scores


def create_training_data_for_di_shuffled_essays(refined_essay, di_list):
    essay_orig, essay_shf = di_shuffled_essay(refined_essay, di_list)
    total_essay = essay_orig + essay_shf
    
    # The original essays work as positive examples (label: 1), while the shuffled essays work as negative examples (label: 0)
    scores = [1] * len(refined_essay) + [0] * len(refined_essay)
    
    return np.array(total_essay), scores 

def get_padded_essay(input_essay, max_seq_len):
        
    essay_padded = []
    for essay in input_essay:
        
        new_ess = []
        
        for i in range(max_seq_len):
            try:
                new_ess.append(essay[i])
                
            except:
                new_ess.append("__PAD__")
                          
        essay_padded += [new_ess]
        
    return essay_padded


def get_persing_sequence(essay, prompt):
    essay=essay.lower()
    essay_sent=sent_tokenize(essay)
    
    # The first and last elements are empty.
    essay_para=re.split('\n',essay)[1:-1]

    transition_words=['also', 'again', 'as well as', 'besides', 'coupled with', 'furthermore', 'in addition', 'likewise', 
                  'moreover', 'similarly', 'accordingly', 'as a result', 'consequently', 'for this reason', 
                  'for this purpose', 'hence', 'otherwise', 'so then', 'subsequently', 'therefore', 'thus', 
                  'thereupon', 'wherefore', 'contrast', 'by the same token', 'conversely', 'instead', 'likewise', 
                  'on one hand', 'on the other hand', 'on the contrary', 'rather', 'similarly', 'yet', 'but', 
                  'however', 'still', 'nevertheless', 'in contrast','here', 'there', 'over there', 'beyond', 'nearly',
                  'opposite', 'under', 'above','to the left', 'to the right', 'in the distance', 'by the way', 
                  'incidentally', 'above all', 'chiefly', 'with attention to', 'especially', 'particularly', 
                  'singularly', 'aside from', 'barring', 'beside', 'except', 'excepting', 'excluding', 'exclusive of',
                  'other than', 'outside of', 'save', 'chiefly', 'especially', 'for instance', 'in particular', 
                  'markedlFy', 'namely', 'particularly', 'including', 'specifically', 'such as', 'as a rule',
                  'as usual', 'for the most part', 'generally', 'generally speaking', 'ordinarily', 'usually', 
                  'or example', 'for instance', 'for one thing', 'as an illustration', 'illustrated with', 
                  'as an example', 'in this case', 'comparatively', 'coupled with', 'correspondingly',
                  'identically', 'likewise', 'similar', 'moreover', 'together with', 'in essence', 
                  'in other words', 'namely', 'that is', 'that is to say', 'in short', 'in brief', 
                  'to put it differently', 'at first', 'first of all', 'to begin with', 'in the first place',
                  'at the same time','for now', 'for the time being', 'the next step', 'in time', 'in turn', 
                  'later on','meanwhile', 'next', 'then', 'soon', 'the meantime', 'later', 'while', 'earlier',
                  'simultaneously', 'afterward', 'in conclusion', 'with this in mind', 'after all', 
                  'all in all', 'all things considered', 'briefly', 'by and large', 'in any case', 'in any event', 
                  'in brief', 'in conclusion', 'on the whole', 'in short', 'in summary', 'in the final analysis', 
                  'in the long run', 'on balance', 'to sum up', 'to summarize', 'finally']

    No_of_label=[]
    score=0

    a1=re.compile(r'\b[Tt]hey\b')
    a2=re.compile(r'\b[Th]hem\b')
    a3=re.compile(r'\b[Mm]y\b')
    a4=re.compile(r'\b[Hh]e\b')
    a5=re.compile(r'\b[Ss]he\b')

    a6=re.compile(r'\b[Aa]gree\b')
    a7=re.compile(r'\b[Dd]isagree\b')
    a8=re.compile(r'\b[Th]ink\b')
    a9=re.compile(r'\b[Oo]pinion\b')

    a10=re.compile(r'\b[Ff]irstly\b')
    a11=re.compile(r'\b[Ss]econdly\b')
    a12=re.compile(r'\b[Tt]hirdly\b')
    a13=re.compile(r'\b[Aa]nother\b')
    a14=re.compile(r'\b[Aa]spect\b')

    a15=re.compile(r'\b[Ss]upport\b')
    a16=re.compile(r'\b[Ii]nstance\b')

    a17=re.compile(r'\b[Cc]onclusion\b')
    a18=re.compile(r'\b[Cc]onclude\b')
    a19=re.compile(r'\b[Tt]herefore\b')
    a20=re.compile(r'\b[[Ss]um\b]')

    a21=re.compile(r'\b[Hh]owever\b')
    a22=re.compile(r'\b[Bb]ut\b')
    a23=re.compile(r'\b[Aa]rgue\b')

    a24=re.compile(r'\b[Ss]olve\b')
    a25=re.compile(r'\b[Ss]olved\b')
    a26=re.compile(r'\b[Ss]olution\b')

    a27=re.compile(r'\b[Ss]hould\b')
    a28=re.compile(r'\b[Ll]et\b')
    a29=re.compile(r'\b[Mm]ust\b')
    a30=re.compile(r'\b[Oo]ught\b')
    

    
    stop_words = set(stopwords.words('english'))

    prompi=RegexpTokenizer(r'\w+').tokenize(prompt.lower())
    promp= [w for w in prompi if not w in stop_words]


    paragraph=0

    sequence=""

    for j in essay_para:

        essay_sent=sent_tokenize(j)
        paragraph=paragraph+1
        s=0
        No_of_label=[]

        intro=0
        body=0
        rebut=0
        conclude=0

        for i in essay_sent:

            Elaboration = 0
            Prompt = 0
            Transition = 0
            Thesis = 0
            MainIdea = 0
            Support = 0
            Conclusion = 0
            Rebuttal = 0
            Solution = 0
            Suggestion = 0


            s=s+1

            #Elaboration

            b1=a1.findall(i)
            b2=a2.findall(i)
            b3=a3.findall(i)
            b4=a4.findall(i)
            b5=a5.findall(i)

            if len(b1)!=0:
                Elaboration=Elaboration+1
            if len(b2)!=0:
                Elaboration=Elaboration+1
            if len(b3)!=0:
                Elaboration=Elaboration+1
            if len(b4)!=0:
                Elaboration=Elaboration+1
            if len(b5)!=0:
                Elaboration=Elaboration+1 



            if s==1:
                Prompt=Prompt+1
                Thesis=Thesis+1

            if s==len(essay_sent):
                Conclusion=Conclusion+1   

             #Prompt


            content_wordsi=RegexpTokenizer(r'\w+').tokenize(i.lower())  
            content_words=[w for w in content_wordsi if w not in stop_words]

            match_words=[]

            for j in promp:
                if j in content_words:
                    match_words.append(j)

            if len(content_words)!=0:
                Prompt=Prompt+(5/2)*(len(match_words)/len(content_words)) 
            else:
                Prompt=Prompt+(5/2)*0

            #Transition

            word_tokens=word_tokenize(i)
            if '?' in word_tokens:
                Transition=Transition+1

            n=4
            bi_grami=list(bigrams(RegexpTokenizer(r'\w+').tokenize(i.lower())))
            bi_gram=[' '.join(str(w)for w in l) for l in bi_grami]
            tri_grami=list(bigrams(RegexpTokenizer(r'\w+').tokenize(i.lower())))
            tri_gram=[' '.join(str(w)for w in l) for l in tri_grami]
            four_grami=list(bigrams(RegexpTokenizer(r'\w+').tokenize(i.lower())))
            four_gram=[' '.join(str(w)for w in l) for l in four_grami]

            n_gram=content_wordsi+bi_gram+tri_gram+four_gram

            for k in transition_words:
                if k in n_gram:
                    Transition=Transition+1


            #Thesis

            b6=a6.findall(i)
            b7=a7.findall(i)
            b8=a8.findall(i)
            b9=a9.findall(i)
            
            if len(b6)!=0:
                Thesis=Thesis+1
            if len(b7)!=0:
                Thesis=Thesis+1      
            if len(b8)!=0:
                Thesis=Thesis+1
            if len(b9)!=0:
                Thesis=Thesis+1 
              


            #MainIdea

            b10=a10.findall(i)
            b11=a11.findall(i)
            b12=a12.findall(i)
            b13=a13.findall(i)
            b14=a14.findall(i)
           

            if len(b10)!=0:
                MainIdea=MainIdea+1
            if len(b11)!=0:
                MainIdea=MainIdea+1
            if len(b12)!=0:
                MainIdea=MainIdea+1
            if len(b13)!=0:
                MainIdea=MainIdea+1   
            if len(b14)!=0:
                MainIdea=MainIdea+1
              


            #Support

            b15=a15.findall(i)
            b16=a16.findall(i)

            if len(b15)!=0:
                Support=Support+1
            if len(b16)!=0:
                Support=Support+1

            #Conclusion

            b17=a17.findall(i)
            b18=a18.findall(i)
            b19=a19.findall(i)
            b20=a20.findall(i)

            if len(b17)!=0:
                Conclusion=Conclusion+1
            if len(b18)!=0:
                Conclusion=Conclusion+1
            if len(b19)!=0:
                Conclusion=Conclusion+1
            if len(b20)!=0:
                Conclusion=Conclusion+1
            

            #Rebuttal

            b21=a21.findall(i)
            b22=a22.findall(i)
            b23=a23.findall(i)

            if len(b21)!=0:
                Rebuttal=Rebuttal+1
            if len(b22)!=0:
                Rebuttal=Rebuttal+1
            if len(b23)!=0:
                Rebuttal=Rebuttal+1    

            #Solution

            b24=a24.findall(i)
            b25=a25.findall(i)
            b26=a26.findall(i)

            if len(b24)!=0:
                Solution=Solution+1
            if len(b25)!=0:
                Solution=Solution+1  
            if len(b26)!=0:
                Solution=Solution+1    


            #Suggestion

            b27=a27.findall(i)
            b28=a28.findall(i)
            b29=a29.findall(i)
            b30=a30.findall(i)

            if len(b27)!=0:
                Suggestion=Suggestion+1    
            if len(b28)!=0:
                Suggestion=Suggestion+1
            if len(b29)!=0:
                Suggestion=Suggestion+1    
            if len(b30)!=0:
                Suggestion=Suggestion+1  


            dictn={}
            dictn['Elaboration']=Elaboration.real
            dictn['Transition']=Transition.real
            dictn['Thesis']=Thesis.real
            dictn['MainIdea']=MainIdea.real
            dictn['Support']=Support.real
            dictn['Conclusion']=Conclusion.real
            dictn['Rebuttal']=Rebuttal.real
            dictn['Solution']=Solution.real
            dictn['Suggestion']=Suggestion.real
            dictn['Prompt']=Prompt.real


            s_label=sorted(dictn, key=dictn.get, reverse=True)[:1]

            s_label_s=s_label[0]


            if s_label_s=='Thesis' or s_label_s=='Prompt':
                intro=intro+1
                conclude=conclude+1
            elif s_label_s=='MainIdea' and s<=3:
                body=body+1
                conclude=conclude+1
                rebut=rebut+1
            elif s_label_s=='MainIdea' and s>len(essay_sent)-3:
                body=body+1
                intro=intro+1   
            elif s_label_s=='Elaboration':
                intro=intro+1
                body=body+1
            elif s_label_s=='Support':
                body=body+1
            elif s_label_s=='Suggestion' or s_label_s=='Conclusion':
                body=body+1
                conclude=conclude+1
            elif s_label_s=='Rebuttal' or s_label_s=='solution':
                body=body+1
                rebut=rebut+1  


        if paragraph==1:
            intro=intro+1
        elif paragraph==len(essay_para):
            conclude=conclude+1
        else:
            body=body+1
            rebut=rebut+1

        dict_para={}
        dict_para['I']=intro.real
        dict_para['B']=body.real
        dict_para['C']=conclude.real
        dict_para['R']=rebut.real

        para_label=sorted(dict_para, key=dict_para.get, reverse=True)[:1]

        para_label_s=para_label[0]

        sequence=sequence+para_label_s

    return(sequence)           