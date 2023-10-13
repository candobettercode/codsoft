import streamlit as st

# NLP
import string, re, nltk
from string import punctuation
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords

from num2words import num2words

from spellchecker import SpellChecker
from nltk.stem.porter import PorterStemmer
import spacy
from spacy.lang.en.examples import sentences 
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

cv = CountVectorizer(max_features=10000, ngram_range=(1,2))
nb = MultinomialNB()

st.set_page_config(page_title="Movie Genre Detection", page_icon=":clapper board:", layout="centered")
# giving a title
st.title(":camera: Movie Genre Classification")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)

st.text("Tecknowcode @Tecknowcode")
st.text("Siddhesh M.(stream)")

def genre_prediction(sample_script):
    
    sample_script = re.sub(pattern='[^a-zA-Z]',repl=' ', string=sample_script)
    sample_script = sample_script.lower()
    sample_script_words = sample_script.split()
    sample_script_words = [word for word in sample_script_words if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    final_script = [ps.stem(word) for word in sample_script_words]
    final_script = ' '.join(final_script)
    temp = cv.transform([final_script]).toarray()
    return nb.predict(temp)[0]

def main():
        
    message = st.text_area("Enter your text","Type here",height=400)

    submit = st.button("Find Genre")

    value = genre_prediction(message)

    '''
    if submit:
        sumary, inputdata, summary_len, data_len = summarizer(message)  
        st.text("Text is being Summarized ... ")
        st.markdown("## Summarized output")
        st.warning('Number of characters in Input data: {}'.format(data_len))
        st.error('Number of characters in Summarized data: {}'.format(summary_len))
        st.success(sumary)
      
     '''
       
if __name__ == '__main__':
    main()
