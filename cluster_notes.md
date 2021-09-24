# Cluster Analysis 

### Sources:
1. [Zhihu](https://www.zhihu.com/question/19982667)
2. 

## Hypothesis:
1. 假设数据之间存在相似性，且是有价值的
2.  常见应用包括：
	* **用户分割**：将用户划分到不同的组别中，并根据簇的特性而推送不同的广告
	* 欺诈检测：发现正常与异常的用户数据，识别其中的欺诈行为


## How to Choose a Suitable Cluster Method
聚类算法的运算开销往往很高，所以最重要的选择标准往往是数据量
> ''在数据量不大时，优先采取别的算法。当数据量过大时，可以尝试HDBSCAN。仅当数据量巨大且无法降维或者降低数量时，再选择Kmeans''

首先要回答：<u>我们要解决什么问题</u>?

<mark>The intention here is not to divide users into different age and gender groups, but to find a cluster of participants with similar habits/preferences.</mark>

	1. 所以应该完全排除用户的个人信息进行聚类
	2. 无关变量不作为输入，应该在聚类完成后作为分析变量
	3. 方法不绝对，可以通过调整weight来实现个人信息的参与

## 分析变量的重要性
变量选择<u>主观</u>, 且聚类是无监督学习，因此很难评估变量的重要性

2 possible approaches:

1. 考虑变量的内在变化度与变量间的关联性：一个变量本身方差很小，那么不易对聚类气到很大的影响。如果变量间的相关性很高，那么相关性间的变量应该被合并处理。
2. 直接采用算法来对变量重要性进行排序：比如Principle Feature Analysis [[1]](https://dl.acm.org/doi/pdf/10.1145/1291233.1291297). 
	* 现成代码[[2]](https://stats.stackexchange.com/questions/108743/methods-in-r-or-python-to-perform-feature-selection-in-unsupervised-learning)



## How to Decide Whether the Results are Reliable
* 几个基本标准来定义好的聚类结果：

	1. 符合商业常识，大致方向上可以被领域专家所验证
	2. 可视化后有一定的区别，而并非完全随机且交织在一起
	3. 如果有预先设定的评估函数，评估结果较为优秀




