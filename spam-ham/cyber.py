import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

def make_prediction(text1):
    text = [text1]
    df =pd.read_csv('spam.tsv',sep='\t')
    x = df.iloc[:, 1].values
    y = df.iloc[:, 0].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
    vect = TfidfVectorizer()
    x_train_v = vect.fit_transform(x_train)
    x_test_v = vect.transform(x_test)
    model = SVC()
    model.fit(x_train_v,y_train)
    def load_model_and_predict(text):
        text_vector = vect.transform(text)
        prediction = model.predict(text_vector)
        print(prediction[0])
        return prediction[0]

    new_text = text
    prediction = load_model_and_predict(new_text)
    return prediction
make_prediction("win free tickets")