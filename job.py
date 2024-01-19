import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from mrjob.job import MRJob
from mrjob.protocol import JSONValueProtocol
import pickle
import re
import time

class SMSSpamDetection(MRJob):

    OUTPUT_PROTOCOL = JSONValueProtocol

    def __init__(self, *args, **kwargs):
        MRJob.__init__(self, *args, **kwargs)
        self.start_time = time.time()
        data = pd.read_csv('spam.csv', encoding = "ISO-8859-1")
        with open('model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        messages = data['v2']
        preprocessed_messages = [self.preprocess_message(message) for message in messages]
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit_transform(preprocessed_messages)

    def preprocess_message(self, message):
        message = message.lower()
        message = re.sub(r'[^\w\s]', '', message)
        message = re.sub(r'\s+', ' ', message)
        return message     

    def mapper(self, _, message):
        preprocessed_message = self.preprocess_message(message)
        features = self.vectorizer.transform([preprocessed_message])
        proba = self.model.predict_proba(features).tolist()
        yield "messages", [list(proba[0]), message]

    def reducer(self, key, values):
        messages = []
        for value in values: 
            record = value
            label = "spam"
            proba = record[0]
            if proba[0] < proba[1]:
                label = "spam"
            else:
                label = "ham"
            messages.append({
                "message": record[1],
                "label": label,
            })
        output = {
            "messages": messages,
            "metaData": {
                "totalMessages": len(messages),
                "time": "{0:.3f}s".format(time.time() - self.start_time)
            }
        }
        yield key, output
        
if __name__ == '__main__':
    SMSSpamDetection.run()
