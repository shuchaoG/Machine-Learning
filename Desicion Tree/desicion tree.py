from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

def func():
    dataset1=open(r'D:\Machine Learning\Desicion Tree\dataset1.csv','rb')
    reader=csv.reader(dataset1)
    headers=reader.next()
    print (headers)

    featureList=[]
    lableList=[]

    for row in reader:
        lableList.append(row[len(row)-1])
        rowDict={}
        for i in range(1,len(row)-1):
            rowDict[headers[i]]=row[i]
        featureList.append(rowDict)
    print(featureList)

    vec=DictVectorizer();
    dumX=vec.fit_transform(featureList).toarray()
    print("dumX:"+str(dumX))
    print(vec.get_feature_names())

    lb=preprocessing.LabelBinarizer()
    dumY=lb.fit_transform(lableList)
    print('dumY:'+str(dumY))

    clf=tree.DecisionTreeClassifier(criterion='entropy')
    clf=clf.fit(dumX,dumY)
    print('clf:'+str(clf))

    with open('dataset1.dot','w') as f:
        f=tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)

    oneRowX=dumX[0,:]
    print('oneRowX:'+str(oneRowX))

    newRowX=oneRowX
    newRowX[0]=1
    newRowX[2]=0

    predictY=clf.predict([newRowX])
    print('predictY:'+str(predictY))



if __name__=='__main__':
    print ('main')
    func()