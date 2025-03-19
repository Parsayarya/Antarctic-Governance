import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import triu
from sklearn.decomposition import LatentDirichletAllocation
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# df = pd.read_csv('../Data/Inputs/ATCM46/ATCM46_WithNames_Categories-WITH-TEXT.csv')
df = pd.read_csv('Data/cleaned_dataset_with_Categories_And_BPs.csv')
print(df.info())
df = df.dropna(subset=['text'])
df['Year'] = df['Year'].apply(lambda x: int(x))
df = df[df['Year']>=1991]

SW_list = ['al', 'et', 'mt', 'ip', 'ofthe', 'ix', 'xxiii', 'use','used','february', 'january', 'december',
           'march', 'november' , 'october','atcm', 'including','iii','set','view','party','annex','year',
           'government','kingdom','accordance','submitted', 'chile','located','small','contact','available'
           ,'report','document','meeting','secretariat','key','new','zealand','land','united','level','shall',
           'accordance','person','antarctica','purpose','material','description','ensure','issue','based','question',
           'need','initial','effort','understanding','work','using','carried','noted','source', 'uk','year','co','also','de',
           'ii','le','iv', 'would', 'could', 'may', 'wether', 'might', 'however', 'paragraph','reported','within','']
def preprocess_text(text: str) -> str:
    """
    Preprocess a given text by removing punctuation, special characters, digits,
    and then lemmatizing all the words.

    Parameters:
    text (str): The input text to be preprocessed.

    Returns:
    str: The preprocessed text.
    """
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    cleaned_text = cleaned_text.lower()
    tokens = nltk.word_tokenize(cleaned_text)
    extended_stopwords = stopwords.words('english') + SW_list
    tokens = [word for word in tokens if word not in extended_stopwords and len(word) > 1]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    lemmatized_text = ' '.join(lemmatized_tokens)

    return lemmatized_text



def LDAV1(df, text_column, n_topics=30, n_top_words=20):
    """
    Perform LDA topic modeling on a given DataFrame and extract topic scores and top words for each topic.

    :param df: Input pandas DataFrame containing text data.
    :param text_column: The name of the column in the DataFrame containing text data.
    :param n_topics: Number of topics to model (default 50).
    :param n_top_words: Number of top words to extract for each topic (default 20).
    :return: Tuple of DataFrames (topic_scores_df, topics_df) where:
             - topic_scores_df contains original data with added topic score columns,
             - topics_df lists top words for each topic.
    """
    # Vectorize text data
    extended_stopwords = stopwords.words('english') + SW_list
    count_vectorizer = CountVectorizer(max_df=0.85, min_df=0.15, stop_words=extended_stopwords)
    dtm = count_vectorizer.fit_transform(df[text_column])
    lda = LatentDirichletAllocation(n_components=n_topics, learning_method='online', learning_offset=50., random_state=42, verbose=1,n_jobs=-1).fit(dtm)

    # Get feature names and topic words
    tf_feature_names = count_vectorizer.get_feature_names_out()
    topics_df = pd.DataFrame(index=range(n_top_words))
    for topic_idx, topic in enumerate(lda.components_):
        topics_df[f"Topic {topic_idx}"] = [tf_feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]

    # Create DataFrame with topic scores
    topic_scores = lda.transform(dtm)
    for i in range(n_topics):
        df[f"Topic {i} Score"] = topic_scores[:, i]

    # Saving topics and their top words
    # topics_df.to_csv('topics_and_words_since91.csv')

    # Remove original text column and save DataFrame with topic scores
    # df.drop(columns=[text_column], inplace=True)
    # df.to_csv('document_topic_loadings_since91.csv')

    return df, topics_df


df['text'] = df['text'].apply(lambda x: preprocess_text(x))
df2, topics_df = LDAV1(df, 'text', n_topics=20, n_top_words=20)
df2.to_csv('Since91_20_Topic_Loading_Score.csv',index=False)
topics_df.to_csv('Since91_20_Topic_Words.csv', index=False)



