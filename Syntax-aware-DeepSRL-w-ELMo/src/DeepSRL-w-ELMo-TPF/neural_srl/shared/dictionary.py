''' Bidirectional dictionary that maps between words and ids.
'''
from constants import *

class Dictionary(object):
  def __init__(self, padding_token=None, unknown_token=None):
    self.str2idx = {}
    self.idx2str = []
    self.accept_new = True
    self.padding_token = None
    self.padding_id = None
    self.unknown_token = None
    self.unknown_id = None
    if padding_token != None:  # add the padding info into the dictionary
        self.set_padding_token(padding_token)
    if unknown_token != None:
      self.set_unknown_token(unknown_token)

  def set_padding_token(self, padding_token):
      self.padding_token = padding_token
      self.padding_id = self.add(self.padding_token)

  def set_unknown_token(self, unknown_token):
    self.unknown_token = unknown_token
    self.unknown_id = self.add(unknown_token)

  def add(self, new_str): 
    if not new_str in self.str2idx:
      if self.accept_new:
        self.str2idx[new_str] = len(self.idx2str)
        self.idx2str.append(new_str)
      else:
        if self.unknown_id is None:
          raise LookupError('Trying to add new token to a freezed dictionary with no pre-defined unknown token: ' + new_str)
        return self.unknown_id
      
    return self.str2idx[new_str] 
  
  def add_all(self, str_list):
    return [self.add(s) for s in str_list] 
  
  def get_index(self, input_str):
    if input_str in self.str2idx:
      return self.str2idx[input_str]
    return None
  
  def size(self):
    return len(self.idx2str)

  def save(self, filename):
    with open(filename, 'w') as f:
      for s in self.idx2str:
        f.write(s + '\n')
      f.close()

  def load(self, filename):
    with open(filename, 'r') as f:
      for line in f:
        line = line.strip()
        if line != '':
          if line == UNKNOWN_TOKEN:  # process padding and unknown add by @kiro
            self.set_unknown_token(line)
          elif line == PADDING_TOKEN:
            self.set_padding_token(line)
          else:
            self.add(line)
      f.close()
  

