# Recommendation System

陈婕 1611522 信息安全与法学

### 一、 Basic statistics of the dataset

​	来自19834个用户，评分295086。评级值的范围在0到100之间。大多数评级都是给予曲目（47％）。该数据被认为是稀疏用户x项矩阵R = [r~ui~]，填充率为0.04％非常低。与其他协同过滤数据集相比例如，在Netflix奖数据集填充率约为1％。除了评级数据外，还有可用的分类信息。分类法是指项目之间的关系。

​	图一描绘了每个用户对项目的评级数量的分布，具有明显的幂律特征，具有“热门”项目的长尾和非常活跃的用户，以及与极少数评级相关联的大量项目和用户。

![image](https://ws4.sinaimg.cn/mw690/bd315bc6ly1g1xqepyvyvj20f90dhq3d.jpg)



##### 用户评分方面：

如图所示，`train.txt`数据集中的评分分布，得分在0，30，50，70，90 拥有较高的频率

![image](https://wx1.sinaimg.cn/mw690/bd315bc6ly1g1xon93z6qj20e90b8gln.jpg)



​	可见绝大多数评分是十的倍数。可以推测这种分布来自不同的用户界面，通过0,30,50,70和90处的峰值的优势来对应出1至5星级。

![image](https://wx2.sinaimg.cn/mw690/bd315bc6ly1g1xqr9a4paj20f70dfq3d.jpg)

​	如图为每个user评价item的平均值分布

![image](https://wx4.sinaimg.cn/mw690/bd315bc6ly1g1xr77zr2mj20ec0ba3yj.jpg)

​	如图为每个item被评分的平均值分布

​	描绘了用户和项目的平均评分的分布，两个分布的方差是非常不同的。此外，平均用户评级的分布明显更加偏离。

##### 总结

- 问题和Netflix Prize 相似
- 初步考虑的方法是应用collaborative filtering -> sparse matrix，应用Netflix Prize相同的方案，重写并优化模型。

### 二、 Details of the algorithms

> 1. baseline model
> 2. SVD
> 3. SVD+SGD
> 4. SVD+baseline
> 5. AFM
> 6. SVD++
> 7. ALS

#### 2.1 Baseline Model 

- **matrix factorization model:** 

  P~u~代表用户隐因子矩阵（表示用户u对因子k的喜好程度),Q~i~表示电影隐因子矩阵（表示电影i在因子k上的程度）

​      $\hat{R}_{ui} = Q_i^TP_u$

- **Baseline Predictors**

  不过有些评分与用户和产品的交互无关：有些用户偏向于给产品打高分，有些产品会收到高分。我们将这类不涉及用户产品交互的影响建模为baseline predictors。

  $\hat{r}_{ui} = \mu + b_u + b_i$ 

  μ是平均值，然后分别用b~i~和b~u~来代表具体用户和物品的整体偏差

  

  $b_i=\frac{\sum_{u \in R(i)}{(r_{ui}-\mu)}}{\lambda+|R(i)|}$

  $b_u=\frac{\sum_{u\in R(u)}{(r_{ui}-\mu-b_i)}}{\lambda+|R(u)|}$

#### 2.2 Matrix Factorization Models

- **SVD**

  自Netflix Prize Bennett等人以来，这一直是协作过滤中最受欢迎的模型之一。当使用梯度下降作为学习算法时，训练时间随着评级数| R |线性增长。

![image](https://wx1.sinaimg.cn/mw690/bd315bc6ly1g1xu6vamn6j204m01l0sj.jpg)

​	在求解上文中提到的这类无约束最优化问题时，**梯度下降法（Gradient Descent）**是最常采用的方法之一，其核心思想非常简单，沿梯度下降的方向逐步迭代。梯度是一个向量，表示的是一个函数在该点处沿梯度的方向变化最快，变化率最大，而梯度下降的方向就是指的负梯度方向。

根据梯度下降法的定义，其迭代最终必然会终止于一阶导数（对于多元函数来说则是一阶偏导数）为零的点，即驻点。

**SGD 部分代码：**

![image](https://ws2.sinaimg.cn/mw690/bd315bc6ly1g1xtydv8xsj20he085wfp.jpg)

- **ALS交替最小二乘**

推荐系统中，我们需要计算矩阵分解：

![image](https://ws1.sinaimg.cn/mw690/bd315bc6ly1g1z4ymxqhuj203401et8h.jpg)

因为这里$P$和$Q$同时都是变量，计算会比较复杂。一个简单的方法是，固定其中一个，计算另外一个。例如我们先随机产生$P_0$，然后固定$P_0$，求解

![image](https://ws4.sinaimg.cn/mw690/bd315bc6ly1g1z4zqc0z6j206r01emx0.jpg)

然后再固定$Q_1$，求解

![image](https://ws3.sinaimg.cn/mw690/bd315bc6ly1g1z50snja8j206i01c744.jpg)

之后再固定$P_1$，求解

![image](https://wx3.sinaimg.cn/mw690/bd315bc6ly1g1z50yrtyoj206l01dq2s.jpg)

这样交替进行，每次只更新P和Q的其中一个，每一步计算的过程就和最小二乘法一样；所以叫做交替最小二乘法。

- **ALS的进阶变形：ALS-WR，解决的问题背景：** 

  多数情况下，用户没有明确反馈对商品的偏好，也就是没有直接打分，我们只能通过用户的某些行为来推断他对商品的偏好。

  比如，在电视节目推荐的问题中，对电视节目收看的次数或者时长，这时我们可以推测次数越多，看得时间越长，用户的偏好程度越高，但是对于没有收看的节目，可能是由于用户不知道有该节目，或者没有途径获取该节目，我们不能确定的推测用户不喜欢该节目。

  ALS-WR通过置信度权重来解决这些问题：对于更确信用户偏好的项赋以较大的权重，对于没有反馈的项，赋以较小的权重。

  本质就是对loss function进行修正，使其更符合实际情况。 

- **SVD+ baseline Model:**

![image](https://wx4.sinaimg.cn/mw690/bd315bc6ly1g1xube084dj208601mt8j.jpg)

![image](https://ws1.sinaimg.cn/mw690/bd315bc6ly1g1xub5g7h5j20fq01rjrc.jpg)

加入防止过拟合的 λ 参数，最简单的SVD是优化下面的Loss function：

![image](https://wx2.sinaimg.cn/mw690/bd315bc6ly1g1xuavpirnj208e04o0sx.jpg)

- **AFM**

  此模型称为非对称因子模型，因为它只有与item有关参数。Paterek在Paterek（2007年）首次提到它。

  对于实际的应用场景中，经常有这样一种情况：用户点击查看了某一个物品，但是最终没有给出评分。实际上，对于用户点击查看物品这个行为，排除误操作的情况，在其余的情况下可以认为用户被物品的描述，例如贴图或者文字描述等所吸引。这些信息我们称之为隐式反馈。事实上，一个推荐系统中有明确评分的数据是很少的，这类隐式数据才占了大头。

  ![image](https://wx2.sinaimg.cn/mw690/bd315bc6ly1g1xu9b97xlj20av02s74b.jpg)

  关键性假设： 对于每个item，还是使用q~i~来给出对应的低维表示。对于每个用户，我们假设对应的低维向量可以由在训练中评价过的商品来表示。对于用户u，将其在训练集中评价过的集合记为I(u)，并使用I(u)中的商品来表示用户u。即：

  ![image](https://wx2.sinaimg.cn/mw690/bd315bc6ly1g1xuy0c8lqj207c01vdfo.jpg)

  对于相关的商品 $i \in I(u)$ ,用向量 $q_j^{(0)}$来表示该商品对于用户u对应的向量的贡献。

- **翻转的AFM 模型**

![image](https://wx2.sinaimg.cn/mw690/bd315bc6ly1g1xug0h5q7j20b402uaa3.jpg)

对于商品i ， 使用$p_u^{(0)}$ 来表示：

![image](https://ws2.sinaimg.cn/mw690/bd315bc6ly1g1xv68rngjj207501t3yc.jpg)

U(i)是训练集中评价过商品i 的用户集合， |U(i)| 是评价过商品 i 的用户总数

#### 2.3 ASVD(SVD++) 模型

​	ASVD模型将基本的SVD方法与AFM的想法相结合。我们可以在显式兴趣+偏置的基础上再添加隐式兴趣，即

![image](https://wx2.sinaimg.cn/mw690/bd315bc6ly1g1z4v54y1tj20c002yjrc.jpg)

求解公式如下

![image](https://wx2.sinaimg.cn/mw690/bd315bc6ly1g1z4vtmyokj20d406cwez.jpg)

#### 2.4 基于内容的推荐系统

`itemAttribution.txt`里提供了item 的两种属性，基于向量计算各个item 之间的相似度

距离测量方法：

- **Similarity Metric** 

**Euclidean Distance:**

 ![image](https://wx3.sinaimg.cn/mw690/bd315bc6ly1g1yhx6zljqj206r02rq2x.jpg)

> 注：数据处理： 将属性中为`None`的值记为平均值。

- **打分预测**

  ![image](https://wx2.sinaimg.cn/mw690/bd315bc6ly1g1yi8n3l1uj204q01it8i.jpg)

> 由于item-dataset 太大，训练需要多天，没有办法实现。

#### 2.5 模型融合

**线性加权融合**：

最常用的是采用加权型的混合推荐技术，将来自不同推荐算法生成的候选结果及结果的分数，进一步进行组合（Ensemble）加权，生成最终的推荐排序结果。

$w$为影响因子，通过最小二乘法计算w

![image](https://ws1.sinaimg.cn/mw690/bd315bc6ly1g1ynts6jwsj206z01dmwz.jpg)

使用相同的初始化参数再次训练每个模型，但现在使用训练集和验证集来执行训练。最后，计算每个模型mi的测试集预测（xi）并获得最终预测：

![image](https://wx2.sinaimg.cn/mw690/bd315bc6ly1g1z3x45ybfj203w01iwed.jpg)

### 三、 Experimental results (RMSE, training time, space consumption)  



![image](https://wx2.sinaimg.cn/mw690/bd315bc6ly1g1xurva9z0j20e90ast96.jpg)









### 四、 附录

对训练数据进行处理，得到用户ID、物品ID、用户ID与用户矩阵列号的对应关系、物品ID与物品矩阵列号的对应关系、评分矩阵的Shape、评分矩阵及评分矩阵的转置。

```
def _process_data(self, X):
    self.user_ids = tuple((set(map(lambda x: x[0], X))))
    self.user_ids_dict = dict(map(lambda x: x[::-1],
                                    enumerate(self.user_ids)))

    self.item_ids = tuple((set(map(lambda x: x[1], X))))
    self.item_ids_dict = dict(map(lambda x: x[::-1],
                                    enumerate(self.item_ids)))

    self.shape = (len(self.user_ids), len(self.item_ids))

    ratings = defaultdict(lambda: defaultdict(int))
    ratings_T = defaultdict(lambda: defaultdict(int))
    for row in X:
        user_id, item_id, rating = row
        ratings[user_id][item_id] = rating
        ratings_T[item_id][user_id] = rating

    err_msg = "Length of user_ids %d and ratings %d not match!" % (
        len(self.user_ids), len(ratings))
    assert len(self.user_ids) == len(ratings), err_msg

    err_msg = "Length of item_ids %d and ratings_T %d not match!" % (
        len(self.item_ids), len(ratings_T))
    assert len(self.item_ids) == len(ratings_T), err_msg
    return ratings, ratings_T
```

用户矩阵乘以评分矩阵：实现稠密矩阵与稀疏矩阵的矩阵乘法，得到用户矩阵与评分矩阵的乘积。

```
def _users_mul_ratings(self, users, ratings_T):

    def f(users_row, item_id):
        user_ids = iter(ratings_T[item_id].keys())
        scores = iter(ratings_T[item_id].values())
        col_nos = map(lambda x: self.user_ids_dict[x], user_ids)
        _users_row = map(lambda x: users_row[x], col_nos)
        return sum(a * b for a, b in zip(_users_row, scores))

    ret = [[f(users_row, item_id) for item_id in self.item_ids]
            for users_row in users.data]
    return Matrix(ret)
```

训练模型：

1. 数据预处理
2. 变量k合法性检查
3. 生成随机矩阵U
4. 交替计算矩阵U和矩阵I，并打印RMSE信息，直到迭代次数达到max_iter
5. 保存最终的RMSE

```
def fit(self, X, k, max_iter=10):
    ratings, ratings_T = self._process_data(X)
    self.user_items = {k: set(v.keys()) for k, v in ratings.items()}
    m, n = self.shape

    error_msg = "Parameter k must be less than the rank of original matrix"
    assert k < min(m, n), error_msg

    self.user_matrix = self._gen_random_matrix(k, m)

    for i in range(max_iter):
        if i % 2:
            items = self.item_matrix
            self.user_matrix = self._items_mul_ratings(
                items.mat_mul(items.transpose).inverse.mat_mul(items),
                ratings
            )
        else:
            users = self.user_matrix
            self.item_matrix = self._users_mul_ratings(
                users.mat_mul(users.transpose).inverse.mat_mul(users),
                ratings_T
            )
        rmse = self._get_rmse(ratings)
        print("Iterations: %d, RMSE: %.6f" % (i + 1, rmse))

    self.rmse = rmse
```

 

#### Reference

<https://blog.csdn.net/dark_scope/article/details/17228643>

孙亮、黄倩.实用机器学习[M].北京：人民邮电出版社，2017.5:204-252. 