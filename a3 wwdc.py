#!/usr/bin/env python
# coding: utf-8

# # FILMATIC

# In[2]:


import pandas as pd
import numpy as np


# https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows

# ### Clean the data 
# 
# 

# In[3]:


df=pd.read_csv('imdb.csv')


# In[4]:


df.head()


# #### Splitting genres and removing the [genre] column
# 

# In[5]:


df['genres']=df['Genre'].str.split(',')


# In[6]:


df


# In[7]:


removing_genre='Genre'


# In[8]:


df=df.drop(columns=[removing_genre])


# #### Collate 'star' columns into one column

# In[9]:


remove_columns=['Star1','Star2','Star3','Star4']


# In[10]:


df['Stars']=df.apply(lambda row: f"{row['Star1']} - {row['Star2']}", axis=1)


# In[11]:


df.drop(columns=remove_columns, inplace=True)


# In[12]:


df


# #### Add user id column

# In[13]:


user_ids=range(1,1001)


# In[14]:


df['user_id']=user_ids


# In[15]:


df.insert(0,'user_id', df.pop('user_id'))


# In[16]:


df.head()


# In[17]:


df['user_id']=np.random.randint(1,1001,size=len(df))


# In[18]:


df


# In[19]:


df.to_csv('imdb2.csv',index=False)


# #### Change Runtime column variables from object to integers

# In[20]:


df['Runtime']=df['Runtime'].str.replace(r'\W', "")
df['Runtime']


# In[21]:


df['Runtime']=df['Runtime'].str.replace("min", "")
df['Runtime']=df['Runtime'].str.replace(",","")


# In[22]:


df['Runtime']=df['Runtime'].astype('object').astype('int64')


# In[23]:


df


# In[24]:


df.isnull().sum()


# In[25]:


df.dtypes


# In[26]:


df.describe()


# #### Check the average/mean of the ratings and count of the ratings 

# In[27]:


no_votes= df[df['No_of_Votes'].notnull()]['No_of_Votes'].astype('int')


# In[28]:


imdb_average =df[df['IMDB_Rating'].notnull()]['IMDB_Rating'].astype('int')


# In[29]:


voteaverages =imdb_average.mean()


# In[30]:


voteaverages


# In[31]:


no_of_votes=no_votes.mean()


# In[32]:


no_of_votes


# #### Genre Statistics, explode the genres into their own rows 

# In[33]:


genre_stats=df['genres'].describe()


# In[34]:


genre_stats


# In[35]:


df_genres=df.explode('genres')
df_genres


# In[36]:


show_genre_movies='Drama'


# In[37]:


drama_movies=df_genres[df_genres['genres'].str.contains(show_genre_movies, case=False, na=False)]


# In[38]:


drama_movies


# In[39]:


counts = df_genres['genres'].value_counts(ascending=True)


# In[40]:


genrecounts = df_genres['genres'].str.contains('Drama')
drama_count = genrecounts.sum()

print(f'In this dataset there are {drama_count} movies in the drama genre.')


# In[41]:


genrecounts = df_genres['genres'].str.contains('Comedy')
Comedy_count = genrecounts.sum()

print(f'In this dataset there are {Comedy_count} movies in the comedy genre.')


# In[42]:


genrecounts = df_genres['genres'].str.contains('Action')
action_count = genrecounts.sum()

print(f'In this dataset there are {action_count} movies in the action genre.')


# In[43]:


genrecounts = df_genres['genres'].str.contains('Romance')
romance_count = genrecounts.sum()

print(f'In this dataset there are {romance_count} movies in the romance genre.')


# In[44]:


genrecounts = df_genres['genres'].str.contains('Thriller')
thriller_count = genrecounts.sum()

print(f'In this dataset there are {thriller_count} movies in the thriller genre.')


# ### Data Visualisation

# In[45]:


axs = df.plot.area(figsize=(14,5), subplots = True)


# In[46]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[47]:


imdb_column='IMDB_Rating'
rating_column='No_of_Votes'


# In[48]:


plt.plot(df[imdb_column],label='IMDB_Rating', linestyle='-')

plt.xlabel('Number of movies')
plt.ylabel('Ratings')
plt.title('Changes of ratings within the dataset')


# In[49]:


fig,ax=plt.subplots(figsize=(30,10))
df.groupby('Released_Year').count()['Series_Title'].plot(kind='bar')
plt.title("Amount of movies released throughout the decades", fontsize=20)



# In[50]:


plt.hist(df['Runtime'],bins=90)
plt.show()
plt.hist(df['IMDB_Rating'],bins=30)
plt.show()



        


# ### Create recommendation systems

# In[51]:


import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer




# In[52]:


df['Series_Title']=df['Series_Title'].str.lower()


# In[53]:


df['Overview']=df['Overview'].str.lower()


# In[54]:


df['Released_Year']=df['Released_Year'].str.lower()


# In[55]:


df['Stars']=df['Stars'].str.lower()


# In[56]:


df['Director']=df['Director'].str.lower()


# In[57]:


df_copy2=df.copy()
df_copy2


# In[58]:


df2=df_copy2.drop(['genres','Overview', 'Stars', 'Director'],axis=1)


# In[59]:


df2


# ###  Simple content-based recommender system

# In[60]:


df2['datastrings']=df2[df2.columns[1:]].apply(
    lambda x: ''.join(x.dropna().astype(str)),
    axis=1
)
print(df2['datastrings'].head())


# In[61]:


vectorizer=CountVectorizer()
vectorized=vectorizer.fit_transform(df2['datastrings'])


# In[62]:


vectorized


# In[63]:


similarities=cosine_similarity(vectorized)


# In[64]:


print(similarities)


# In[65]:


vectorizer= TfidfVectorizer(ngram_range=(1,2))

# The ngram range is important to the functioning of the system, as it focuses on the composition of more than one word


# In[66]:


df=pd.DataFrame(similarities,columns=df_copy2['Series_Title'],index=df_copy2['Series_Title']).reset_index()

df.head()


# In[67]:


movie_example_input='toy story'
movie_recommendation=pd.DataFrame(df.nlargest(4,movie_example_input)['Series_Title'])
recommendations=movie_recommendation[movie_recommendation['Series_Title']!=movie_example_input]
print(recommendations)


# In[ ]:





# ### Genre content-based recommendation engine

# In[68]:


df3=df_copy2.copy()


# In[69]:


df3


# In[70]:


df_genres['genres']=df_genres['genres'].str.replace('Sci-Fi','SciFi')
df_genres['genres']=df_genres['genres'].str.replace('Film-noir','Noir')


# In[71]:


tfidf_vector=TfidfVectorizer(stop_words='english')


# In[72]:


tfidf_matrix=tfidf_vector.fit_transform(df_genres['genres'])


# In[73]:


print(list(enumerate(tfidf_vector.get_feature_names_out())))


# In[74]:


print(tfidf_matrix[:5])


# In[75]:


similarities2=linear_kernel(tfidf_matrix,tfidf_matrix)
print(similarities2)


# In[76]:


from fuzzywuzzy import fuzz


# In[77]:


def matching_score(a,b):
    return fuzz.ratio(a,b)


# In[78]:


def get_title_year_from_index(index):
    return df_genres[df_genres.index==index]['Series_Title'].values[0]

def get_title_from_index(index):
    return df_genres[df_genres.index==index]['Series_Title'].values[0]


def get_index_from_title(title):
    return df_genres[df_genres==title].index.values[0]
    


# In[79]:


def search_titles(title):
    leven_scores = list(enumerate(df_genres['Series_Title'].apply(matching_score, b=title)))
    sorted_leven_scores=sorted(leven_scores,key=lambda x: x[1],reverse=True)
    closest_title=get_title_from_index(sorted_leven_scores[0][0])
    distance_score=sorted_leven_scores[0][1]
    return closest_title, distance_score


# In[80]:


def recs_based_on_genre(movie_likes, how_many):
    
    closest_title, distance_score =search_titles(movie_likes)
    
    if distance_score ==100:
        film_index=get_index_from_title(closest_title)
        film_list=list(enumerate(similarities2[int(film_index)]))
        similar_films=list(filter(lambda x:x[0] != int(film_index), sorted(film_list,key=lambda x:x[1], reverse=True)))

        print('Here are the list of films similar to '+'\033[1m'+str(closest_title)+'\033[0m'+'.\n')

        for i, s in similar_films[:how_many]:
            print(get_title_year_from_index(i))


    else:
        print('You have mispelled your title, did you mean '+'\033[1m'+str(closest_title)+'\033[0m'+'?','\n')

        
        film_index=get_index_from_title(closest_title)
        film_list=list(enumerate(similarities2[int(film_index)]))
        similar_films=list(filter(lambda x:x[0] != int(film_index), sorted(film_list,key=lambda x:x[1], reverse=True)))
        
        print('Here are the list of films similar to '+'\033[1m'+str(closest_title)+'\033[0m'+'.\n')


        for i, s in similar_films[:how_many]:
            print(get_title_year_from_index(i))



# In[120]:


recs_based_on_genre('the godfather',20)


# In[ ]:





# ### Content based recommender combining qualitative and quantitative values

# In[82]:


df3


# In[83]:


df3['combined_columns']=df3['Overview']+['IMDB_Rating']


# In[84]:


tfidf_vectorizer1=TfidfVectorizer(stop_words='english')


# In[85]:


tfidf_matrix1=tfidf_vectorizer1.fit_transform(df3['combined_columns'].values.astype('U'))


# In[86]:


cosine_sim1=linear_kernel(tfidf_matrix1,tfidf_matrix1)


# In[87]:


tfidf_matrix1.shape


# In[88]:


def recs(title, cosine_sim1=cosine_sim1):
    idx=df3.index[df3['Series_Title']==title].tolist()[0]
    sim_scores=list(enumerate(cosine_sim1[idx]))
    sim_scores=sorted(sim_scores,key=lambda x: x[1], reverse=True)
    sim_scores=sim_scores[1:11]
    movie_indices=[i[0] for i in sim_scores]
    return df3['Series_Title'].iloc[movie_indices].tolist()


# In[89]:


recs('goodfellas')


# In[ ]:





# ### Collaborative Filtering recommendation engine

# In[90]:


movies=pd.read_csv("imdb2.csv")


# In[91]:


df4=movies.loc[:,['user_id','IMDB_Rating','Series_Title','No_of_Votes']]


# In[92]:


df4


# In[93]:


users_movie_matrix=pd.pivot_table(df4,columns='Series_Title',index='user_id',values='No_of_Votes')
users_movie_matrix


# In[94]:


toy_story_rating=users_movie_matrix["Toy Story"]
toystory_corr=users_movie_matrix.corrwith(toy_story_rating)
corr_df=pd.DataFrame(toystory_corr,columns=['Correlation'])
corr_df=corr_df.join(df4['No_of_Votes'])
similar_movies=corr_df[corr_df['No_of_Votes']>500].sort_values(['Correlation'],ascending=False)
similar_movies


# ### Charts constructed on the basis of the top movies, through weighted rating(WR)

# In[95]:


# Whilst this is not a functioning recommender system, this interprets collaborative filtering as it focalises user ratings


# In[96]:


voteaverages


# In[97]:


va=voteaverages
va


# In[98]:


no_of_votes


# In[99]:


v_quantile=no_votes.quantile(0.50)


# In[100]:


v_quantile


# In[101]:


qualified_movies=movies[(movies['No_of_Votes']>=v_quantile) & (movies['No_of_Votes'].notnull() & (movies['IMDB_Rating'].notnull()))]
[['Series_Title','Released_Year','No_of_Votes','IMDB_Rating','genres']]
                                                                                


# In[119]:


qualified_movies['No_of_Votes']=qualified_movies['No_of_Votes'].astype('int')
qualified_movies['IMDB_Rating']=qualified_movies['IMDB_Rating'].astype('int')
qualified_movies.shape
pd.options.mode.chained_assignment = None  # default='warn'


# In[103]:


def averagely_weighted_rating(x):
    v=x['No_of_Votes']
    R=x['IMDB_Rating']
    return (v/(v+v_quantile) * R) + (v_quantile/(v_quantile+v) * va)


# In[104]:


qualified_movies['wr']=qualified_movies.apply(averagely_weighted_rating, axis=1)
pd.options.mode.chained_assignment = None  # default='warn'


# In[105]:


qualified_movies=qualified_movies.sort_values('wr', ascending=False).head(250)


# In[106]:


qualified_movies.head(15)


# In[107]:


qualified_movies.tail(15)


# ### Random sampling of popular films

# In[108]:


random_film=qualified_movies.sample(n=1)


# In[109]:


print(random_film[['Series_Title','genres']])


# In[110]:


specific_genre='Drama'


# In[111]:


random_genre_film=qualified_movies[qualified_movies['genres'].str.contains(specific_genre)]


# In[112]:


random_drama_film=random_genre_film.sample(n=10)
print(pd.DataFrame(random_drama_film[['Series_Title','Runtime','Released_Year','Overview']]))


# In[113]:


data_random_frame=pd.DataFrame({
    'Series Title':random_drama_film['Series_Title'].values,
    'Released Year':random_drama_film['Released_Year'].values,
    'Genre':random_drama_film['genres'].values,
    'Overview':random_drama_film['Overview'].values,
    'Runtime':random_drama_film['Runtime'].values,
    'Rating':random_drama_film['IMDB_Rating'].values,
    'Stars':random_drama_film['Stars'].values
})


# In[114]:


data_random_frame


# In[115]:


specified_director= 'Stanley Kubrick'


# In[116]:


specified_director=qualified_movies[qualified_movies['Director']==specified_director]


# In[117]:


print(specified_director[['Series_Title']])


# In[118]:


unique_directors = qualified_movies['Director'].unique()
unique_directors

