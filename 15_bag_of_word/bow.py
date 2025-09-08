
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
import tensorflow_datasets as tfds

corpus=[
    'this is the first document',
    'this is the second document',
    'and this is the third document',
    'is this the first document',
]

vectorizer=CountVectorizer()
x=vectorizer.fit_transform(corpus)
word= vectorizer.get_feature_names_out()
print(word)
out=x.toarray()
print(out)
new= vectorizer.transform(['How is you']).toarray()
print(new)

dataset=tfds.load('ag_news_subset')
ds_train=dataset['train']
ds_test=dataset['test']
classes=['world','sports','business', 'sci/tech']
vectorizer_tf=tf.keras.layers.TextVectorization(max_tokens=50000)
vectorizer_tf.adapt(ds_train.take(500).map(lambda x:x['title'] + ' ' + x['description']))

def to_bow(text):
    return tf.reduce_sum(tf.one_hot( vectorizer_tf(text),50000), axis=0)

dog=to_bow('my dogs likes hot dogs')
print(dog)