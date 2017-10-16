from sklearn import tree
#sklearn prebuild models
# tree represents Decision trees

clf = tree.DecisionTreeClassifier()

# - Lets create 3 more classifiers...
# 1
# 2
# 3
    #1st variable list of lists handling sequence of values.
# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43],[190,78,42]]

#storing list of labels of body matrics in the previous list
#puts them as strings -(text only)

Y = ['Gentleman', 'Gentleman', 'lady', 'lady', 'Gentleman', 'Gentleman', 'lady', 'lady',
     'lady', 'Gentleman', 'Gentleman']




#variable of our decision tree classifier
#stores our decision tree classifier
# train on our data
#calling our tree dependency & initiating decision tree by calling method on tree object
clf = clf.fit(X, Y)


#train it on our treedataset with a classifier variable that takes two arguments
prediction = clf.predict([[190, 70, 43]])
  

# CHALLENGE compare their results then print the best one!

print(prediction)