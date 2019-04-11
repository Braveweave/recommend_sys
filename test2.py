from surprise import SVD
from  surprise import  Dataset
from surprise import accuracy
from surprise import evaluate ,print_perf
from surprise import Reader
from surprise import KNNBasic
from surprise.model_selection import cross_validate,train_test_split

import  os
file_path=os.path.expanduser('process_data.txt')
reader=Reader(line_format='user item rating',sep=',',rating_scale=(0,100))

surprise_data=Dataset.load_from_file(file_path,reader=reader)
print('success load')
trainset,testset=train_test_split(surprise_data,test_size=0.25)
print('trainset')
model=SVD(n_factors=100)
model.fit(trainset)
predictions=model.test(testset)
accuracy.rmse(predictions)

# cross_validate(model, surprise_data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# pref=evaluate(model,surprise_data,measures=['RMSE','MAE'])
# print_perf(pref)

# print(model.qi.shape)



# all_trainset=surprise_data.build_full_trainset()
# algo=KNNBasic(k=40,min_k=3,sim_options={'user_based':True})
# algo.fit(all_trainset)

# data= Dataset.load_builtin('ml-100k')
# data.split(n_folds=3)
# algo=SVD()
# pref=evaluate(algo,data,measures=['RMSE','MAE'])
# print_perf(pref)