import numpy as np
import graphviz
import pydot

from sklearn.datasets import load_iris
from sklearn import tree 
from sklearn.externals.six import StringIO

iris = load_iris()
test_idx = [0,50,100]

#training data 
train_target = np.delete (iris.target,test_idx)
train_data = np.delete (iris.data, test_idx , axis=0)

#test data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print (test_target)
print(clf.predict(test_data))



  


dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 




#print (iris.feature_names)
#print (iris.target_names)
#print (iris.data[0])
#print (iris.target[0])

#for  i in range (len(iris.target)):
#	print ("Example %d  : label  %s , features %s" % (i,iris.target[i] , iris.data[i]) )