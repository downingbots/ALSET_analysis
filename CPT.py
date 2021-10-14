# MIT License
# 
# Copyright (c) 2018 Neeraj Singh Sarwan
# 
# PredictionTree is now included below
# from PredictionTree import *

# Modified for Alset Move prediction
import pandas as pd
from tqdm import tqdm


class CPT():

    alphabet = None # A set of all unique items in the entire data file
    root = None # Root node of the Prediction Tree
    II = None #Inverted Index dictionary, where key : unique item, value : set of sequences containing this item
    LT = None # A Lookup table dictionary, where key : id of a sequence(row), value: leaf node of a Prediction Tree

    def __init__(self, k):
        self.alphabet = set()
        self.root = PredictionTree()
        self.K = k
        self.II = {}
        self.LT = {}
        self.data = []
        self.move_history = []

    def load_files(self,train_file,test_file = None):
        """
        This function reads in the wide csv file of sequences separated by commas and returns a list of list of those
        sequences. The sequences are defined as below.
        seq1 = A,B,C,D
        seq2 = B,C,E
        Returns: [[A,B,C,D],[B,C,E]]
        """
        data = [] # List of list containing the entire sequence data using which the model will be trained.
        target = [] # List of list containing the test sequences whose next n items are to be predicted
        if train_file is None:
            return train_file
        train = pd.read_csv(train_file)
        for index, row in train.iterrows():
            data.append(row.values)
        if test_file is not None:
            test = pd.read_csv(test_file)
            for index, row in test.iterrows():
                data.append(row.values)
                target.append(list(row.values))
            return data, target
        return data

    def add_next_move(self, move, done=False):
        """
        Called after a new element has been appended to move_history.
        After the calls, the following sets will registered with the CPT;

        move_history = A,B,C,D,E,F,G,H
          seq1 = A,B,C,D,E
          seq2 = B,C,D,E,F
          seq3 = C,D,E,F,G
          seq4 = D,E,F,G,H
        """
        self.move_history.append(move)
        if len(self.move_history) < self.K:
          return

        move_sequence = self.move_history[-self.K:]
        self.data.append(move_sequence)
        # print("CPT: move_seq", move_sequence)
        self.train(move_sequence)
        if done:
          # in Alset, this is a run of an app to completion.
          # The next run can use/continue training the CPT.
          self.move_history = []


    # In[3]:
    def train(self, row):
        """
        This functions populates the Prediction Tree, Inverted Index and LookUp Table for the algorithm.
        Input: The list of list training data
        Output : Boolean True
        """
        cursornode = self.root
        # for seqid,row in enumerate(data):
        seqid = len(self.data)
        # print("CPT: train row:", seqid, row)
        # for seqid,row in enumerate(data):
        if True:
            for element in row:
                # adding to the Prediction Tree
                if cursornode.hasChild(element)== False:
                    cursornode.addChild(element)
                    cursornode = cursornode.getChild(element)
                else:
                    cursornode = cursornode.getChild(element)
                # Adding to the Inverted Index
                if self.II.get(element) is None:
                    self.II[element] = set()

                self.II[element].add(seqid)
                
                self.alphabet.add(element)

            self.LT[seqid] = cursornode

            cursornode = self.root
            
        return True


    def score(self, counttable,key, length, target_size, number_of_similar_sequences, number_items_counttable):
        """
        This function is the main workhorse and calculates the score to be populated against an item. Items are predicted
        using this score.

        Output: Returns a counttable dictionary which stores the score against items. This counttable is specific for a 
        particular row or a sequence and therefore re-calculated at each prediction.

        """

        weight_level = 1/number_of_similar_sequences
        weight_distance = 1/number_items_counttable
        score = 1 + weight_level + weight_distance* 0.001
        
        if counttable.get(key) is None:
            counttable[key] = score
        else:
            counttable[key] = score * counttable.get(key)
            
        return counttable

    def predict_move(self,n=1): 
        """
        # ARD: Modified for an each-move prediction
        #
        Here target is the test dataset in the form of list of list,
        k is the number of last oelements that will be used to find similar sequences and,
        n is the number of predictions required.

        Input: training list of list, target list of list, k,n

        Output: max n predictions for each sequence
        """

        if len(self.move_history) < self.K:
          return []

        each_target = self.move_history[-self.K+1:]
        predictions = []
        
        # ARD: tqdm provides an progress bar and is not necessary.
        # for each_target in tqdm(target):
        #   each_target = each_target[-self.K:]
        if True:
            print("each target:", each_target)
            
            intersection = set(range(0,len(self.data)))
            
            for element in each_target:
                if self.II.get(element) is None:
                    continue
                intersection = intersection & self.II.get(element)
            
            similar_sequences = []
            
            for element in intersection:
                currentnode = self.LT.get(element)
                tmp = []
                while currentnode.Item is not None:
                    tmp.append(currentnode.Item)
                    currentnode = currentnode.Parent
                similar_sequences.append(tmp)
                
            for sequence in similar_sequences:
                sequence.reverse()

                
            counttable = {}

            # print("CPT: len similar sequences:", len(similar_sequences))
            # print("CPT: similar sequences:", similar_sequences)
            for  sequence in similar_sequences:
                # print("CPT: sequence:", sequence)
                try:
                    index = next(i for i,v in zip(range(len(sequence)-1, 0, -1), reversed(sequence)) if v == each_target[-1])
                    # index = None
                    # print("CPT: len seq/rev seq ", len(sequence)-1, reversed(sequence))
                    # print("CPT: zip ", set(zip(range(len(sequence)-1, 0, -1),reversed(sequence))))
                    # for i,v in zip(range(len(sequence)-1, 0, -1),reversed(sequence)):
                    #   if v == each_target[-1]:
                    #     # index = next(i)
                    #     index = i-1
                    #     print("CPT: i,v,ind", i, v, index)
                    #   else:
                    #     print("CPT: i,v", i, v)
                    # if index is None:
                    #     print("CPT: seq", sequence)
                    #     print("CPT: zip", set(zip(range(len(sequence)-1, 0, -1),reversed(sequence))))
                except Exception as e:
                    index = None
                    # print("CPT: index None:", e)
                if index is not None:
                    count = 1
                    for element in sequence[index+1:]:
#                        if element in each_target:
#                            print("CPT: in each_target ", element)
#                            continue
                            
                        counttable = self.score(counttable,element,len(each_target),len(each_target),len(similar_sequences),count)
                        count+=1
                        # print("CPT: count ", count)

            # if len(counttable) > 0:
            #   print("CPT: len intersection:", len(intersection), len(similar_sequences), len(counttable))

            pred = self.get_n_largest(counttable,n)
            if len(pred) > 0:
                predictions.append(pred)

        return predictions



    def get_n_largest(self,dictionary,n):


        """
        A small utility to obtain top n keys of a Dictionary based on their values.

        """
        largest = sorted(dictionary.items(), key = lambda t: t[1], reverse=True)[:n]
        return [key for key,_ in largest]


class PredictionTree():
    Item = None
    Parent = None
    Children = None
    
    def __init__(self,itemValue=None):
        self.Item = itemValue
        self.Children = []
        self.Parent = None
        
    def addChild(self, child):
        newchild = PredictionTree(child)
        newchild.Parent = self
        self.Children.append(newchild)
        
    def getChild(self,target):
        for chld in self.Children:
            if chld.Item == target:
                return chld
        return None
    
    def getChildren(self):
        return self.Children
        
    def hasChild(self,target):
        found = self.getChild(target)
        if found is not None:
            return True
        else:
            return False
        
    def removeChild(self,child):
        for chld in self.Children:
            if chld.Item==child:
                self.Children.remove(chld)
