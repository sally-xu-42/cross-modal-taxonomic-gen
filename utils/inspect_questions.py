import json
import random
import glob
import pandas as pd

if __name__=="__main__":
    lengthData = 20
    valList=[]
    with open('../data/CLEVR_v1.0/questions/CLEVR_val_questions.json') as f:
        data = json.load(f)
        for k in range(lengthData):
            i = data['questions'][random.randrange(20, 5000, 3)]
            temp=[]
            for path in glob.glob('../data/CLEVR_v1.0/images/val/'+i['image_filename']): 
                temp.append(path)
            temp.append(i['question'])
            temp.append(i['answer'])
            valList.append(temp)
    f.close()
    val_dataframe = pd.DataFrame(valList)#validation Dataframe
    del(data)
    del(valList)
    print(val_dataframe.head())
    val_dataframe.to_csv('./inspect.csv', index=False)