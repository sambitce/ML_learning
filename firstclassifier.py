from  sklearn import tree
##features = [[ 140,"smooth"] , [130,"smooth"] , [150,"Bumpy"] , [170,"Dumpy"]]
features = [[ 140,1] , [130,1] , [150,0] , [170,0]]
##labels = ["apple" , "apple" , "orange" "orange"]
labels = [0,0,1,1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)
output = clf.predict([[1000,1]])
print (output)
