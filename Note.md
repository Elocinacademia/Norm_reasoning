#Note

###Data info:
---

+ `clean_data.csv` 只包含四个parameter和数字value  <mark>用于mine association rules</mark>  

*Example:*

| datatype| recipient| acceptability | transmission_principle|
| ----------- | ----------- |-----------|-----------|
| 2       | 1    | 4  | 3   |
| 1       | 13 |  2|  1|
| ...     | ...  |  ... |  ... |

+ `plain_data.csv` 是👆`clean_data.csv`的text版本

+ `data_ques_1.csv` 包含question body和question number  <mark>用于pandas</mark>

*Example:*

| Assume ...| Assume ...| Assume ... | Assume ...|
| ----------- | ----------- |-----------|-----------|
| Q31_1     | Q31_2  | Q31_3  | Q31_4   |
| empty     | neutral |  empty |  completely acceptable|
| ...     | ...  |  ... |  ... |

+ `data_ques_pure.csv` 只包含question text和具体的值 <mark>用于进行判断并生成user piece based dataset</mark>


*Example:*

| Assume ...| Assume ...| Assume ... | Assume ...|
| ----------- | ----------- |-----------|-----------|
| empty     | neutral |  empty |  completely acceptable|
| ...     | ...  |  ... |  ... |

+ `new_file.csv` 是1000+用户数据，每行是用户所有labeled information flow

*Example:*

| ['healthcare', 'your parents', 'none', 'no condition', 'Neutral']| ...| ... |  ...|
| ----------- | ----------- |-----------|-----------|
| ['email', 'assistant provider', 'none', 'if the data is kept confidential, i.e., not shared with others', 'Completely Acceptable']   | ...|  ... |  ...|
| ...     | ...  |  ... |  ... |


###Script info:
---

+ `main.py` 主要脚本 workspace


+ Data Processing

 * `to_user_data.py` 把最原始的文本数据转换为***以用户为单位的数据*** output是 `new_flie.csv`
 * fsdfsfs


	+ Word Embeddings

	* `glove_vec.py` 用于生成 `glove.6B.100d`

+ Test Scripts

	* `test_split.py` 用来测试划***划分数据集***的script
	*  `test_sort.py` 用来练习***sort***函数
	*  `test_mining.py` 用来测试***association rule mining***是否work
	*  `test_similarity.py` 实现***similarity calculation***的几个方法