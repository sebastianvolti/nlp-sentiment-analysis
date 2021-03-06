import pandas as pd
import re
from constants import TWEET, ES_ODIO, COL_NAMES
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def toLowerCase(text):
  return text.lower()

def removeUrl(text):
  return re.sub(r'http\S+', 'URL',text) 

def removePunctuation(text):
  return re.sub('[\W_]+', ' ',text)

def removeRepetitions(text):
  def repl(matchobj):
    c=matchobj.group(0)
    return c[0]
  return re.sub(r'(\w)\1{2,}',repl ,text)         

def removeUsers(text):
  return re.sub("\\@.*?\\ ", "USER ", text)

def removeHashtags(text):
  return re.sub(r"#", "", text)

def removeLaughter(text):
  return re.sub(r"((j|J)aja[\w]*)|((j|J)ajs[\w]*)|((j|J)eje[\w]*)|(JAJA[\w]*)", "jaja", text)

def removeLaughter2(text):
  def repl(matchobj):
    c=matchobj.group(0)
    laugth=re.sub(r"((j|J)aja[\w]*)|((j|J)ajs[\w]*)|((j|J)eje[\w]*)","",c)
    return laugth
  return re.sub(r"([\w]*(j|J)aja[\w]*)|([\w]*(j|J)ajs[\w]*)|([\w]*(j|J)eje[\w]*)",repl, text)+" "+"jaja"

def removen(text):
  return re.sub(r'\\n', "",text)

def RemoveStopWords(dataset):
  stop = stopwords.words('spanish')
  important_words = dataset['Tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
  return important_words

class Data():
  def __init__(self, path, test_names = None):
    test = pd.read_csv(path + "/test.csv", names=COL_NAMES, sep="\t")
    train = pd.read_csv(path + "/train.csv", names=COL_NAMES, sep="\t")
    val = pd.read_csv(path + "/val.csv", names=COL_NAMES, sep="\t")

    self.test_files = None
    if (test_names):
      test_files = []
      for name in test_names:
        test_files.append((name.replace('.csv', ''), pd.read_csv(path + "/" + name, names=COL_NAMES, sep="\t").iloc[:,0]))

      self.test_files = test_files

    for i , row in test.iterrows():
      test.at[i,'Tweet'] = self.preprocess(row[TWEET])

    for i , row in train.iterrows():
      train.at[i,'Tweet'] = self.preprocess(row[TWEET])
    
    for i , row in val.iterrows():
      val.at[i,'Tweet'] = self.preprocess(row[TWEET])

    test.Tweet=RemoveStopWords(test)
    train.Tweet=RemoveStopWords(train)
    val.Tweet=RemoveStopWords(val)

    self.test = test
    self.train = train
    self.val = val

  def preprocess(self, text):
    text = removen(text)
    text = removeUrl(text)
    text = removeHashtags(text)
    text = toLowerCase(text)
    text = removeUsers(text)
    text = removeRepetitions(text)
    text = removePunctuation(text)
    text = removeLaughter(text)
    return text

def main():
  pass

if __name__ == "__main__":
  main()