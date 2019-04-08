"""
This module descibes how to load a dataset from a  file.
fileid = 0 trains
       = 1 test
"""

def loaddata(filename,fileid):
    with open(filename,'r') as f:
        lines=f.readlines()
        f.close()
    data_list=[]

    index=0
    while index<len(lines):
        #get user id eg:'0|41'..
        line=lines[index].strip()
        index+=1
        if line=="":
            continue
        if '|' not in line:
            continue
        userid,n_rating=line.split('|')

        n=int(n_rating)

        #get score
        for i in range(n):
            line=lines[index+i].strip()
            if line=="":
                index+=1
                i-=1
                continue
            if fileid == '1':
                itemid=line
                data_list.append((userid,itemid))
                # if userid == '1':
                #     print(userid)
                # # print(userid)
            else:
                itemid,score=line.split('  ')
                data_list.append((userid,itemid,float(score)))

        index+=n
    print('load data finished')
    return data_list



if __name__ == "__main__":
    loaddata('../data-new/train.txt','0')