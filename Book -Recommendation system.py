import pandas as pd
import numpy as np
Book = pd.read_csv("D:\\ExcelR Data\\Assignments\\Recommendation System\\Book2.csv",encoding='ISO-8859-1')
Book .shape #(10000, 3)
Book.columns
#Out[104]: Index(['UserID', 'Book_Title', 'Book_Rating'], dtype='object')
Book.Book_Rating

rating_count = pd.DataFrame(Book, columns=['UserID','Book_Rating'])
# Sorting and dropping the duplicates
rating_count.sort_values('Book_Rating', ascending=False).drop_duplicates().head(10)
#Next I would like to create a dataframe for my  best ten  books
most_rated_books = pd.DataFrame([3943, 278750, 278772, 2453, 278807,278818 ,278831,278832,2442,278843], index=np.arange(10), columns=['UserID'])

#Here am merging the above dataset with my original data set
BookDtL = pd.merge(most_rated_books, Book, on='UserID')
BookDtL

#To Get mean values
rating= pd.DataFrame(Book.groupby('UserID')['Book_Rating'].mean()) #mean values
rating.head(10) #mean values

from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus

#Tfidf Vectorizer is to remove all stop words (Eg:in,all,are,to,....etc)
tfidf = TfidfVectorizer(stop_words="english")    #taking stop words from tfid vectorizer 

# replacing the NaN values in overview column with empty string
Book["Book_Title"].isnull().sum() 
Book["Book_Title"] = Book["Book_Title"].fillna(" ")

# Preparing the Tfidf matrix by fitting and transforming

tfidf_matrix = tfidf.fit_transform(Book.Book_Title)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape #(10000, 11394)

# For now we will be using cosine similarity matrix
# A numeric quantity to represent the similarity between 2 
# Cosine similarity - metric is independent of 
# magnitude and easy to calculate 

from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix)

# creating a mapping of book name to index number 
Book_index = pd.Series(Book.index,index=Book['Book_Title']).drop_duplicates()

Book_index["PLEADING GUILTY"]

def get_Book_recommendations(Book_Title,topN):
    
   
    #topN = 10
    # Getting the Book index using its title 
    Book_id = Book_index[Book_Title]
    
    # Getting the pair wise similarity score for all the Book's

    cosine_scores = list(enumerate(cosine_sim_matrix[Book_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores,key=lambda x:x[1],reverse = True)
    
    # Get the scores of top 10 most similar Book's
    cosine_scores_10 = cosine_scores[0:topN+1]
    
    # Getting the Book index 
    Book_idx  =  [i[0] for i in cosine_scores_10]
    Book_scores =  [i[1] for i in cosine_scores_10]
    
    # Similar movies and scores
    Book_similar_show = pd.DataFrame(columns=["Book_Title","Rating"])
    Book_similar_show["Book_Title"] = Book.loc[Book_idx,"Book_Title"]
    Book_similar_show["Rating"] = Book_scores
    Book_similar_show.reset_index(inplace=True)  
    Book_similar_show.drop(["index"],axis=1,inplace=True)
    print (Book_similar_show)
    

    
# Enter your Book and number of Book's to be recommended 
get_Book_recommendations("PLEADING GUILTY",topN=10)
