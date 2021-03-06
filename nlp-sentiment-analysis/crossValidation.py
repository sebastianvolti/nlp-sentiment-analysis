from sklearn.model_selection import KFold
from constants import ES_ODIO, TWEET, MODEL_TYPES
import data
import model

PATH="./resources"
d = data.Data(PATH)
X = d.val[TWEET]
Y = d.val[ES_ODIO]

folds = KFold(n_splits=5)
epochs_simple=[10,20,50,100,200,300,400,500]
epochs=[10,20,40,60,80,100]
neurons=[1,4,16,32,64,128]
dropout=[0.1,0.3,0.5,0.7]
batchs=[16,32,64,128,256]
types = [MODEL_TYPES["SIMPLE"], MODEL_TYPES["LSTM1"], MODEL_TYPES["LSTM2"], MODEL_TYPES["CONVOLUTIONAL"], MODEL_TYPES["BIDIRECTIONAL"]]
results=[]
results_accuracy=[]
results_recall=[]
results_precision=[]
cont=0
cont_accuracy=0
cont_recall=0
cont_precision=0
res=open('res.txt','w')

for model_type in types:
  if model_type==MODEL_TYPES["SIMPLE"]:
    res.write('Modelo Simple: \n')
    res.write('\n')
    res.write('Ajuste epocas: \n')
    res.write('\n')
    for e in epochs_simple:
      l=len(list(folds.split(d.val)))
      for train_index, test_index in folds.split(d.val):
        X_train = d.val.iloc[train_index]
        X_test = d.val.iloc[test_index]
        m = model.Model(model_type=model_type, train_dataset=X_train,  neurons=128, dropout=0.5,  val_dataset=X_test, path=PATH)
        m.train(e,0)
        accuracy,f1_score,recall,precision = m.eval()
        cont=cont+f1_score
        cont_accuracy=cont_accuracy+accuracy
        cont_recall=cont_recall+recall
        cont_precision=cont_precision+precision
      
      cont=cont/l
      results.append((e,cont))

      cont_accuracy=cont_accuracy/l
      results_accuracy.append((e,cont_accuracy))

      cont_recall=cont_recall/l 
      results_recall.append((e,cont_recall))

      cont_precision=cont_precision/l 
      results_precision.append((e,cont_precision))

      cont=0
      cont_accuracy=0
      cont_precision=0
      cont_recall=0
    for r in results:
      res.write('Promedio de F1-score para cada valor del hiperparametro epocas: %s \n' % (r,)) 
    res.write('\n')
    for ra in results_accuracy:
      res.write('Promedio de accuracy para cada valor del hiperparametro epocas: %s \n' % (ra,))
    res.write('\n')
    for rp in results_precision:
      res.write('Promedio de precision para cada valor del hiperparametro epocas: %s \n' % (rp,))
    res.write('\n')
    for rc in results_recall:
      res.write('Promedio de recall para cada valor del hiperparametro epocas: %s \n' % (rc,))
    results=[]  
    results_accuracy=[]
    results_precision=[]
    results_recall=[]
  else:  #MODELOS NO SIMPLES
    res.write('\n')
    res.write('Modelo %s \n' % (model_type))
    res.write('\n')
    res.write('Ajuste dropout: \n')
    res.write('\n')
    for drop in dropout:                                                       
      for train_index, test_index in folds.split(d.val):
        X_train = d.val.iloc[train_index]
        X_test = d.val.iloc[test_index]
        m = model.Model(model_type=model_type, train_dataset=X_train, neurons=128, dropout=drop, val_dataset=X_test, path=PATH)
        m.train(10,128)
        accuracy,f1_score,recall,precision = m.eval()
        cont=cont+f1_score
        cont_accuracy=cont_accuracy+accuracy
        cont_recall=cont_recall+recall
        cont_precision=cont_precision+precision
      
      cont=cont/l
      results.append((drop,cont))

      cont_accuracy=cont_accuracy/l
      results_accuracy.append((drop,cont_accuracy))

      cont_recall=cont_recall/l 
      results_recall.append((drop,cont_recall))

      cont_precision=cont_precision/l 
      results_precision.append((drop,cont_precision))

      cont=0
      cont_accuracy=0
      cont_precision=0
      cont_recall=0
    for r in results:
      res.write('Promedio de F1-score para cada valor del hiperparametro dropout: %s \n' % (r,))
    res.write('\n')
    for ra in results_accuracy:
      res.write('Promedio de accuracy para cada valor del hiperparametro dropout: %s \n' % (ra,))
    res.write('\n')
    for rp in results_precision:
      res.write('Promedio de precision para cada valor del hiperparametro dropout: %s \n' % (rp,))
    res.write('\n')
    for rc in results_recall:
      res.write('Promedio de recall para cada valor del hiperparametro dropoput: %s \n' % (rc,))
    results=[]
    results_accuracy=[]
    results_precision=[]
    results_recall=[]
    res.write('\n')
    res.write('Ajuste de epocas \n')
    res.write('\n')
    for e in epochs:                                                              
      l=len(list(folds.split(d.val)))
      for train_index, test_index in folds.split(d.val):
        X_train = d.val.iloc[train_index]
        X_test = d.val.iloc[test_index]
        m = model.Model(model_type=model_type, train_dataset=X_train, neurons=128, dropout=0.5, val_dataset=X_test, path=PATH)
        m.train(e,128)
        accuracy,f1_score,recall,precision = m.eval()
        cont=cont+f1_score
        cont_accuracy=cont_accuracy+accuracy
        cont_recall=cont_recall+recall
        cont_precision=cont_precision+precision
      
      cont=cont/l
      results.append((e,cont))
      
      cont_accuracy=cont_accuracy/l
      results_accuracy.append((e,cont_accuracy))

      cont_recall=cont_recall/l 
      results_recall.append((e,cont_recall))

      cont_precision=cont_precision/l 
      results_precision.append((e,cont_precision))

      cont=0
      cont_accuracy=0
      cont_precision=0
      cont_recall=0
    for r in results:
      res.write('Promedio de F1-score para cada valor del hiperparametro epocas: %s \n' % (r,))
    res.write('\n')
    for ra in results_accuracy:
      res.write('Promedio de accuracy para cada valor del hiperparametro epocas: %s \n' % (ra,))
    res.write('\n')
    for rp in results_precision:
      res.write('Promedio de precision para cada valor del hiperparametro epocas: %s \n' % (rp,))
    res.write('\n')
    for rc in results_recall:
      res.write('Promedio de recall para cada valor del hiperparametro epocas: %s \n' % (rc,))
    results=[]
    results_accuracy=[]
    results_precision=[]
    results_recall=[]
    res.write('\n')
    res.write('Ajuste de neuronas \n')
    res.write('\n')
    for n in neurons:                                                         
      l=len(list(folds.split(d.val)))
      for train_index, test_index in folds.split(d.val):
        X_train = d.val.iloc[train_index]
        X_test = d.val.iloc[test_index]
        m = model.Model(model_type=model_type, train_dataset=X_train, neurons=n, dropout=0.5, val_dataset=X_test, path=PATH)
        m.train(10,128)
        accuracy,f1_score,recall,precision = m.eval()
        cont=cont+f1_score
        cont_accuracy=cont_accuracy+accuracy
        cont_recall=cont_recall+recall
        cont_precision=cont_precision+precision
      
      cont=cont/l
      results.append((n,cont))
      
      cont_accuracy=cont_accuracy/l
      results_accuracy.append((n,cont_accuracy))

      cont_recall=cont_recall/l 
      results_recall.append((n,cont_recall))

      cont_precision=cont_precision/l 
      results_precision.append((n,cont_precision))

      cont=0
      cont_accuracy=0
      cont_precision=0
      cont_recall=0
    for r in results:
      res.write('Promedio de F1-score para cada valor del hiperparametro neuronas: %s \n' % (r,))
    res.write('\n')
    for ra in results_accuracy:
      res.write('Promedio de accuracy para cada valor del hiperparametro neuronas: %s \n' % (ra,))
    res.write('\n')
    for rp in results_precision:
      res.write('Promedio de precision para cada valor del hiperparametro neuronas: %s \n' % (rp,))
    res.write('\n')
    for rc in results_recall:
      res.write('Promedio de recall para cada valor del hiperparametro neuronas: %s \n' % (rc,))
    results=[]
    results_accuracy=[]
    results_precision=[]
    results_recall=[]
    res.write('\n')
    res.write('Ajuste de batchs \n') 
    res.write('\n')
    for b in batchs:                                                       
      l=len(list(folds.split(d.val)))
      for train_index, test_index in folds.split(d.val):
        X_train = d.val.iloc[train_index]
        X_test = d.val.iloc[test_index]
        m = model.Model(model_type=model_type, train_dataset=X_train, neurons=128, dropout=0.5, val_dataset=X_test, path=PATH)
        m.train(10,b)
        accuracy,f1_score,recall,precision = m.eval()
        cont=cont+f1_score
        cont_accuracy=cont_accuracy+accuracy
        cont_recall=cont_recall+recall
        cont_precision=cont_precision+precision
     
      cont=cont/l
      results.append((b,cont))
      
      cont_accuracy=cont_accuracy/l
      results_accuracy.append((b,cont_accuracy))

      cont_recall=cont_recall/l 
      results_recall.append((b,cont_recall))

      cont_precision=cont_precision/l 
      results_precision.append((b,cont_precision))

      cont=0
      cont_accuracy=0
      cont_precision=0
      cont_recall=0
    for r in results:
      res.write('Promedio de F1-score para cada valor del hiperparametro batchs: %s \n' % (r,))
    res.write('\n')
    for ra in results_accuracy:
      res.write('Promedio de accuracy para cada valor del hiperparametro batchs: %s \n' % (ra,))
    res.write('\n')
    for rp in results_precision:
      res.write('Promedio de precision para cada valor del hiperparametro batchs: %s \n' % (rp,))
    res.write('\n')
    for rc in results_recall:
      res.write('Promedio de recall para cada valor del hiperparametro batchs: %s \n' % (rc,))
    results=[]  
    results_accuracy=[] 
    results_precision=[]
    results_recall=[]
res.close()
