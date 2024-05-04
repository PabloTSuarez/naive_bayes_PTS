from utils import db_connect
#engine = db_connect()

# your code here
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB,BernoulliNB
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from pickle import dump
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv')

df.drop('package_name',axis=1,inplace=True)
df

df['review'] = df['review'].str.strip().str.lower()

X = df['review']
y = df['polarity']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

X_train

vector_model = CountVectorizer(stop_words='english')

X_train = vector_model.fit_transform(X_train).toarray()
X_test = vector_model.transform(X_test).toarray()

X_train

model = MultinomialNB()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
y_pred


accuracy_score(y_test,y_pred)


for model_a in [GaussianNB(),BernoulliNB()]:
    model_a.fit(X_train,y_train)
    y_pred_a = model_a.predict(X_test)
    print(f'{model_a} con precision de {accuracy_score(y_test,y_pred_a)*100}%')


hiperparametros = {
    'alpha': np.linspace(0.01,20.0,400),
    'fit_prior': [True,False]
}

random_search = RandomizedSearchCV(model,hiperparametros,n_iter=100,scoring='accuracy',cv=5,random_state=42)
random_search

random_search.fit(X_train,y_train)
random_search.best_params_


best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy_score(y_test,y_pred)

dump(best_model,open('../models/naive_bayes_multinomial_alpha_1-9138095238095236_fit_prior_False_42.model','wb'))

model2 = DecisionTreeClassifier(random_state=42)
model2.fit(X_train,y_train)
y_pred2 = model2.predict(X_test)
accuracy_score(y_test,y_pred2)