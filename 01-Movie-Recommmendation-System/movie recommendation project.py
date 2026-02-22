import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import re
from spacy import displacy
from sklearn.metrics.pairwise import linear_kernel

dataset = pd.read_csv('Dataset.csv')
movie_title = pd.read_csv('Movie_Id_Titles.csv')


print('dataset :')
print(dataset.head(2))
print('titles :')
print(movie_title.head(2))


data = pd.merge(dataset,movie_title, on = 'item_id')
print(data.head())
print(data.shape)

print(len(data['user_id'].value_counts()))
print(len(data['title'].unique()))


max = data['rating'].max()
min = data['rating'].min()
print(f'high rating = {max}, low rating = {min}')


data_rating = dict(data['rating'].value_counts())
print(data_rating)


plt.title('rating count')
sns.barplot(x = list(data_rating.keys()), y = list(data_rating.values()))
plt.show()


average_rating = dict()
count_rating = dict()
for title in data['title'].unique():
    average_rating[title] = data[data['title'] == title]['rating'].mean()
    count_rating[title] = data[data['title'] ==title]['rating'].count()
    tmp_average = np.zeros((data.shape[0]))
tmp_count  = np.zeros((data.shape[0]))
for idx in range(data.shape[0]):
  tmp_average[idx]  = average_rating[data['title'][idx]]
  tmp_count[idx] = count_rating[data['title'][idx]]
data['average_rating'] = tmp_average
print(data['average_rating'])
data['count_rating'] = tmp_count
print(data['count_rating'])
print(data.head(2))


C = data['average_rating'].mean()
M =data['count_rating'].quantile(0.90)

Q_movies = data.copy().loc[data['count_rating'] >= M]
print(Q_movies.shape)


def weighted_rating(x,M=M, C=C):
   v = x['count_rating']
   R = x['average_rating']
   
   return (v/(v+M) * R) +(M/(M+v) * C) 

Q_movies['score'] = Q_movies.apply(weighted_rating , axis = 1)
print(Q_movies['score'])
Q_movies = Q_movies.sort_values('score' , ascending = False)
print(Q_movies)
Q_movies['title'].unique()[:15]
print(Q_movies['title'])


print(len(Q_movies['title'].unique()))

tmp_d = Q_movies['title'].value_counts()
plt.title('Top 15 Movies')
sns.barplot(y=tmp_d.keys()[:15], x=tmp_d.values[:15])
plt.show()

plt.title('Count Movies')
sns.barplot(y = list(Q_movies['title'].unique()) , x = list(Q_movies['title'].value_counts()))
plt.show()

movie_rate = data.pivot_table(index = 'user_id' ,columns = 'title' , values ='rating')
movie_rate.fillna(0 , inplace = True)
print(movie_rate.head())

df = pd.DataFrame()
df['count_rating'] = pd.DataFrame(data.groupby('title')['rating'].count())
print(df.head())


def get_recommendations(title , min_rating_count = 50):

    user_rating = movie_rate[title]

    similar_movies = movie_rate.corrwith(user_rating)

    corr_movies = pd.DataFrame(similar_movies , columns = ['correlations'])

    corr_movies.dropna(inplace = True)

    corr_movies = corr_movies.join(df['count_rating'], how='left', lsuffix='_left', rsuffix='_right')

    final = corr_movies[corr_movies['count_rating'] > min_rating_count].sort_values('correlations' , ascending = False)

    return final
recommended = get_recommendations('Year of the Horse (1997)')
print(recommended.head(10))

plt.title('Best 10 Movies For Year of the Horse Movie')
sns.barplot(y = list(recommended.index)[:10] , x = list(recommended['count_rating'])[:10])
plt.show()





