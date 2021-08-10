from CPT import *

class AnalyzeMove():
  def __init__(self):
      self.model = CPT()
      self.data = None
      self.target = None

  def add_next_move(self,move):
      self.data = self.model.add_next_data_list([move])

  def predict(self):
      if self.data is None:
        return None
      print("len data:", len(self.data))
      self.model.train(self.data)
      prediction = self.model.predict(self.data,self.data,3,1)
      return prediction

