from __future__ import print_function
import math
import random

class Bayes_Classifier:

    def __init__(self):
        self.unique_word_num = 0  # the number of unique words in the entire training data (used in smoothing)
        self.neg_num = 0  # total number of negative reviews
        self.pos_num = 0  # total number of positive reviews
        self.freq_neg = {}  # a map contains pairs of words and their frequencies in the negative review
        self.freq_pos = {}  # a map contains pairs of words and their frequencies in the positive review
        self.neg_word_count = 0 # the total number of words in the negative review (repeating words count separately)
        self.pos_word_count = 0 # the total number of words in the positive review



    def train(self,filename):
        # code to be completed by students to extract features from training file, and
        # to train naive bayes classifier.
        all_words = []

        with open(filename, 'rt') as f:
            lines = f.readlines()

        for line in lines:
            line = line.replace('\n', '')
            fields = line.split('|')
            sentiment = fields[1]
            content = fields[2].split()


            if sentiment == '5':
                self.pos_num +=1
                for word in content:
                    self.pos_word_count += 1
                    word = word.lower()
                    if word in self.freq_pos:
                        self.freq_pos[word] += 1
                    else:
                        self.freq_pos[word] = 1
                    all_words.append(word)

            else:
                self.neg_num +=1
                for word in content:
                    self.neg_word_count += 1
                    word = word.lower()
                    if word in self.freq_neg:
                        self.freq_neg[word] += 1
                    else:
                        self.freq_neg[word] = 1
                    all_words.append(word)

        self.unique_word_num = len(set(all_words))



    def classify(self,filename):
        # code to be completed by student to classifier reviews in file using naive bayes
        # classifier previously trains.  member function must return a list of predicted
        # classes with '5' = positive and '1' = negative

        total_num = float (self.pos_num + self.neg_num) # total number of reviews in the training data

        result = []

        with open(filename, 'rt') as f:
            lines = f.readlines()

        for line in lines:
            line = line.replace('\n', '')
            fields = line.split('|')
            content = fields[2].split()

            pos_prob = math.log(float(self.pos_num) / total_num)
            neg_prob = math.log(float(self.neg_num) / total_num)

            #log probability: log(P(class i| data))=log(P(class i))+ SUM (log(P(data j|class i)))

            for word in content:
                word = word.lower()
                if word in self.freq_pos:
                    pos_prob += math.log((self.freq_pos[word] + 1) / float(self.pos_word_count + self.unique_word_num))
                if word not in self.freq_pos:
                    pos_prob += math.log(1 / float(self.pos_word_count + self.unique_word_num))

                if word in self.freq_neg:
                    neg_prob += math.log((self.freq_neg[word] + 1) / float(self.neg_word_count + self.unique_word_num))
                if word not in self.freq_neg:
                    neg_prob += math.log(1 / float(self.neg_word_count + self.unique_word_num))

            if pos_prob < neg_prob:
                result.append('1')
            else:
                result.append('5')

        return result


