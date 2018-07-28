---
layout: post
title:  "self_note_clustering"
date:   2018-7-28
comments: true
categories: machine_learning
---

# 聚类
---
聚类就是将无标签数据集，根据其内在的相似性，划分为多个类别，类别内的相似度较大，类别间的相似度较小。
聚类属于无监督机器学习算法，主要算法有：K-means聚类，层次聚类，密度聚类和谱聚类。
## k-means
又称k均值。思路：
1. 需要先验确定聚类簇的个数K;
2. 针对每个簇，选定一个中心点（在确定K后，初始中心点一般是随机分配）;
3. 对剩余的每个对象，计算其到各个中心的距离，距离哪个中心最近，就划分到对应的簇中。
4. 划分簇后，重新计算每个簇的中心点，再执行步骤3，4，知道中心点不在发生变化或变化小于指定阈值。

```
def k-means(int k, arr[][], threshold=0.1):
    k_center_list = random.random([i for i in range(len(arr))], k)
    clutering_k = dict()
    while True:
        for i in range(k):
            clutering_k[i] = []
        for obj in arr:
            shortest_distance = sys.maxint
            shortest_k = -1
            for k_i, k_center in enumerate(k_center_list):
                if shortest_distance > distance(obj, k_center):
                    shortest_k = k_i
            clutering_k[shortest_k].append(obj)
        k_center_list_new = get_center_list(clutering_k)
        if k_center_change(k_center_list_new, k_center_list) < 0.1:
            k_center_list = k_center_list_new
            break
        k_center_list = k_center_list_new
    return clustering_k, k_center_list
```
 * **特点：简单有效，但1、在簇的均值可被定义下使用有效；2、必须先验确定K；3、对初值（中心点）的选择敏感，选择的初值不同，最后聚出的簇也可能不同；4、不适用非凸集合; 5、对噪声和孤立点敏感**
 
## 层次聚类
根据定义好的策略，对数据集进行层次分解，直到满足条件为止。可以分为：

 * **凝聚的层次聚类：AGNES算法**
 开始将每个对象作为一个簇，根据策略一步步的聚合，直到满足停止聚合的条件。
 * **分解的层次聚类：DIANA算法**
 开始将整个数据集视为一个簇，根据策略一步步分解，直到满足停止分解的条件。

## 密度聚类
只要一个区域中的点的密度大于某个阈值，就把它加到与之相近的聚类中去。这种算法克服了基于距离的算法缺点——只能发现“类圆形”凸聚类，它可以发现任意形状的聚类，且对噪音数据不敏感，但计算复杂度大。
该类算法可分为：**DBSCAN算法 和 密度最大值算法**
### DBSCAN算法
可以把高密度的区域划分为簇，并不受噪音影响。

* 相关概念
**对象的\epsilon-邻域：**给定对象在半径\epsilon内的区域。
**核心对象：**如果一个对象的邻域中包含的对象数大于给定的数值m，该对象成为**核心对象**。
**直接密度可达：**如果对象p在核心对象q的邻域内，我们说从对象q出发到对象p是**直接密度可达**的（q->p）
**密度可达：** q->p1->p2->p3->p, 从q出发到p是密度可达。
**密度相连：** q<-q1<-o->p1->p2->p,对象q和对象p是从o出发，关于\epsilon和m的**密度相连**。
**簇：** 一个基于密度的簇是最大的密度相连对象集合。
**噪音：** 不包含在任何簇中的对象。

* 思想：
如果一个点p的**\epsilon-邻域**包含多于m个的对象，则创建一个p作为**核心对象**的新**簇**。然后DBSCAN反复寻找从这些核心对象**直接密度可达**的对象，这个过程可能涉及**密度可达**簇的合并。当没有新的点可以被添加到任何簇时，该过程结束。


### 密度最大值聚类
密度最大值聚类是一种简洁的聚类算法，既可以识别各种形状的簇，参数也很容易定义。

* 定义
**局部密度 \rho i**
\rho i = count( distance(d_ij, d_c) )
d_c是截断距离，\rho i 表示到对象i的距离小于d_c的对象的个数。一种推荐做法是选择d_c，使每个点的平均邻居数为所有点的1%-2%。
**高局部密度点距离\sigma i**
\sigma i = min(d_ij) (j: \rho j > \rho i)
在密度高于对象i的所有对象中，到对象i最近的距离

* 簇中心的识别
那些有着较大的**局部密度** \rho i 和 很大的的**高密距离** \sigma i的点被认为是簇的中心；如果高密距离很大，但局部密度很小的点一般是异常点。
确定簇中心后，再根据密度可达的方法进行分类操作
