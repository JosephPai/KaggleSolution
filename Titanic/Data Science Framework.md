大多数初学者在机器学习的领域迷失了方向，因为他们陷入了黑盒子(Black Box)的方式，使用了一些他们并不是很理解的lib和算法。 本教程通过提供一个框架，教会你如何像数据科学家一样思考和编程，从而使你独领风骚。 通过这次学习，将不仅完成这一道题目，你将能够解决任何问题。 我提供了清晰的解释，干净的代码和大量的资源链接。 不断改进中，欢迎指正！
---

这是一个经典的问题，预测二元事件的结果。通俗来讲，这意味着，它不是发生就是没有发生。 例如，你赢了或者没有赢，你通过了考试或者没有通过考试，你被接受或者不被接受。常见的实际应用是是客户的流失与保留。 另一个流行的用例是医疗保健的死亡率或生存分析。 二进制事件产生了一个有趣的动态，因为我们通过统计可以知道，随机猜测应该达到50％的准确率，而不需要创建一个单一的算法或编写一行代码。 但是，就像自动纠正拼写检查技术一样，有时我们人类可能对我们自己而言太聪明，实际上表现不及投掷硬币。 在这篇文章中，我使用了Kaggle的入门大赛Titanic：Machine Learning from Disaster来引导读者，如何使用数据科学框架来克服困难。

## A Data Science Framework
1. **确定问题**：如果数据科学，大数据，机器学习，预测分析，商业智能或其他技术是解决方案，那么问题是什么？古语云三思而后行。问题——需求——构想——技术实现，这是解决数据问题的正确流程，然而我们往往会急于使用一些比较花哨的技术来实现，并没有搞懂真正的问题。
2. **获取数据**：John Naisbitt在他的《1984》书中写道，我们正在“淹没在数据中，但是仍旧渴求知识。”因此，数据集其实可能已经以某种形式存在在某个地方。俗话说，你不必重新发明轮子，你只需要知道在哪里找到它，下一步，我们会重点将“脏数据”转换为“清理数据”。
3. **数据清洗**：这个步骤可以理解为是将“狂野”数据转化为“可管理”数据所需的过程，包括对数据进行结构化处理以便于后续存储和处理，为后续控制开发数据整理标准，提取数据（例如网页抓取）以及数据清理以识别异常，丢失或异常数据点。
4. **探索性分析**：任何曾经处理过数据的人都知道，不好的数据得不到好的结果（garbage-in, garbage-out）。 因此，对数据进行更直观的描述或者可视化处理，对于查找数据集中的潜在问题，模式，分类，相关性是非常重要的。 此外，数据分类（即定性与定量）对理解和选择正确的假设检验或数据模型也很重要。
5. **数据建模**：像描述性和推论性的统计数据一样，数据建模可以总结数据或预测未来的结果。数据集和预期结果将决定可供使用的算法。但是更重要的是，一定要记住，算法是工具，而不是魔法。你必须要能够选择出好的模型。 错误的模型可能导致很差的表现，会导致错误的结论。
6. **测试模型**：在使用部分数据集对模型进行训练之后，是时候测试模型了。这有助于防止模型过拟合，或者使其对于所选子集具有如此特定的效果，以至于不能精确地匹配来自同一数据集的另一个子集。
7. **继续优化**：在这个过程中迭代使其变得更好...更强大...比以前更快。 作为一名数据科学家，您的策略应该是将开发人员的操作和应用程序外包，这样您就有更多时间专注于推荐和设计。 一旦你能够打包你的想法，这成为你的硬通货。

## Step 1：确定问题

对于这个项目，简单来说就是预测泰坦尼克号上的人是否能生还。

**项目简介**：沉没的泰坦尼克号是历史上最著名的沉船事件之一。 1912年4月15日，在首航期间，泰坦尼克号撞上一座冰山后沉没，2224名乘客和机组人员中有1502人遇难。 这一耸人听闻的悲剧震撼了国际社会，并推动了船舶安全条例修正。

沉船导致生命损失的原因之一是乘客和船员没有足够的救生艇。 虽然幸存下来的运气有一些因素，但一些人比其他人更有可能生存，比如妇女，儿童和上层阶级。

在这个挑战中，我们要求你完成对什么样的人可能生存的分析。 特别是，我们要求你运用机器学习的工具来预测哪些乘客幸存下来。

需要技能：
- 二分类
- Python语言（或R语言）基础

## Step 2：获取数据
该网站已经提供了数据集：https://www.kaggle.com/c/titanic/data

## Step 3: 数据清洗
在step2给我们提供的数据已经有了非常好的格式，可以省去我们很多力气，这一步唯一要做的就是数据清洗。

### 3.1 Import Libraies
以下代码基于Python 3.x
``` Python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

#load packages
import sys #access to system parameters https://docs.python.org/3/library/sys.html
print("Python version: {}". format(sys.version))

import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
print("pandas version: {}". format(pd.__version__))

import matplotlib #collection of functions for scientific and publication-ready visualization
print("matplotlib version: {}". format(matplotlib.__version__))

import numpy as np #foundational package for scientific computing
print("NumPy version: {}". format(np.__version__))

import scipy as sp #collection of functions for scientific computing and advance mathematics
print("SciPy version: {}". format(sp.__version__)) 

import IPython
from IPython import display #pretty printing of dataframes in Jupyter notebook
print("IPython version: {}". format(IPython.__version__)) 

import sklearn #collection of machine learning algorithms
print("scikit-learn version: {}". format(sklearn.__version__))

#misc libraries
import timeit
import random
import time


#ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
```

### 3.11 导入数据建模 Libraries

我们将使用流行的scikit学习库来开发我们的机器学习算法。 在sklearn中，算法被称为Estimators，并在他们自己的类中实现。 为了数据可视化，我们将使用matplotlib和seaborn库。 以下是常见的类加载。

``` python
#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8
```

### 3.2 给数据“相个面”

这是与数据的初次见面，通过名字了解数据，有一个大概了解。 它看起来是什么样子（数据类型和值），是什么让它打勾（独立/特征变量），它的目标是什么（依赖/目标变量）。

要开始这一步，我们首先导入我们的数据。 接下来我们使用info（）和sample（）函数来获得变量数据类型（即定性和定量）的快速的概述。

1. **Survived**是我们的结果或因变量。 这是一个二进制数据类型,1为生存和0没有生存。所有其他变量都是潜在的预测变量或自变量。重要的是要注意，更多的预测变量不意味着有会有更好的模型，只有正确的变量才得出争取的结果。
2. **PassengerID**和**Ticket**变量被设定为随机的唯一标识符，对结果变量没有影响。因此，他们将被排除在分析之外。
3. **Class**变量代表船票的等级，是社会经济地位（SES）的代表，1 = 上层，2 = 中层，3 = 下层。
4. **Name**变量是一个字符串类型。 它可以用于特征工程，从称呼可以得出性别，从姓氏可以得出家庭成员数，从头衔得出社会经济地位（SES）。由于有些变量已经存在，我们主要考虑从头衔得出SES，来看看是否能够有效果。
5. **Sex**和**Embarked**是字符串类型。 他们将被转换为虚拟变量进行数学计算。
6. **Age**和**Fare**变量是连续的变量。
7. **SibSp**表示乘客的兄弟姐妹/配偶的人数，**Parch**表示乘客父母/子女的人数。 两者都是离散的定量数据类型。 这可以用于特征工程创建一个家庭大小，作为单独的变量。
8. **Cabin**变量是一种字符串数据类型，可以在特征工程中用于船舶发生事故时的近似位置，和甲板层次推断出SES。 但是，由于有许多空值，所以可能不会将这个变量纳入分析范围。

### 3.21 “4C方法数据清洗”：更正(Correcting)，补全(Completeing)，创建(Creating)和转换(Converting)
在这个阶段，我们将通过四步完成数据清洗 1）校正异常值和异常值，2）完成缺失信息，3）创建新的分析功能，4）将字段转换为正确的格式进行计算和表示。

1. **更正(Correcting)**：审查一下数据，似乎没有任何异常或不可接受的数据输入。 我们看到可能在年龄和票价上会有潜在的异常值。 但是，由于它们目前是合理的值，我们将等到完成我们的探索性分析后，再确定是否应包含或排除这一数据集。 应该指出的是，如果它们是不合理的价值，例如 年龄=800 而不是80，那么现在就可以做出安全的决定。 但是，当我们从原始值修改数据时，我们要谨慎，因为这可能会影响我们的模型的精确性。
2. **补全(Completeing)** ：在**Age**，**Cabin**和**Embarked**有null或缺失的数据。缺少值不是我们所希望的，因为一些算法不知道如何处理空值，会导致失败。当然也会有一些算法，如决策树，可以处理空值。因此，在我们开始建模之前对数据修复是很重要的，因为我们将对不同算法训练出的模型进行对比。有两种常用方法，一是删除记录，二是使用合理的输入填充缺失的值。不建议删除记录，特别是大部分记录，除非它确实代表不完整的记录。最好是对缺失值进行补偿。定性数据的基本方法是使用mode()方法。定量数据的基本方法是使用平均值，中位数或平均值+随机标准偏差进行估算。一种中间方法是使用基于特定标准的基本方法;比如按班级平均年龄或按票价和SES登机。还有更复杂的方法，但是在部署之前，应该将其与基础模型进行比较，以确定提高复杂性是否真正的有价值。对于这个数据集，**Age**将被中位数填补，**Cabin**这一特征将被放弃，**Embarked**使用mode()方法。后续的模型迭代可以修改这个决定，以确定它是否提高了模型的准确性。
3. **创建(Creating)**：特征工程就是当我们使用现有的特征来创建新的特征，以确定它们是否提供了新的信号来预测我们的结果。 对于这个数据集，我们将创建一个标题特征来确定它是否在最终是否生还的结果中起作用。
4. **转换(Converting)**：最后一点，但并非最不重要的，我们将处理格式。这个数据集当中没有日期或货币格式，但需要将数据类型标准化。数据集直接导入类别，这使得数学计算变得困难。对于这个数据集，我们将把类型数据类型转换为分类虚拟变量，用不同数字表示不同类别。

```python
print('Train columns with null values:\n', data1.isnull().sum())
print("-"*10)

print('Test/Validation columns with null values:\n', data_val.isnull().sum())
print("-"*10)

data_raw.describe(include = 'all')
```

### 3.22 数据清洗
刚刚大概确定了我们要做什么，现在开始coding吧。

```python
###COMPLETING: complete or delete missing values in train and test/validation dataset
for dataset in data_cleaner:    
    #complete missing age with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

    #complete embarked with mode
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)

    #complete missing fare with median
    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
    
#delete the cabin feature/column and others previously stated to exclude in train dataset
drop_column = ['PassengerId','Cabin', 'Ticket']
data1.drop(drop_column, axis=1, inplace = True)

print(data1.isnull().sum())
print("-"*10)
print(data_val.isnull().sum())
```

```python
###CREATE: Feature Engineering for train and test/validation dataset
for dataset in data_cleaner:    
    #Discrete variables
    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1

    dataset['IsAlone'] = 1 #initialize to yes/1 is alone
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1

    #quick and dirty code split title from name: http://www.pythonforbeginners.com/dictionary/python-split
    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]


    #Continuous variable bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut
    #Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)

    #Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)


    
#cleanup rare title names
#print(data1['Title'].value_counts())
stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
title_names = (data1['Title'].value_counts() < stat_min) #this will create a true false series with title name as index

#apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/
data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
print(data1['Title'].value_counts())
print("-"*10)


#preview data again
data1.info()
data_val.info()
data1.sample(10)
```

### 3.23 转换格式
下面，我们将分类数据转换为虚拟变量（数字化）进行数学分析。 有多种方法来编码分类变量。 我们将使用sklearn和pandas函数。

在这一步中，我们还将定义我们的x（独立/特征/解释/预测等）和y（依赖/目标/结果/响应/等）变量进行数据建模。

```python
#CONVERT: convert objects to category using Label Encoder for train and test/validation dataset

#code categorical data
label = LabelEncoder()
for dataset in data_cleaner:    
    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
    dataset['Title_Code'] = label.fit_transform(dataset['Title'])
    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])


#define y variable aka target/outcome
Target = ['Survived']

#define x variables for original features aka feature selection
data1_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone'] #pretty name/values for charts
data1_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare'] #coded for algorithm calculation
data1_xy =  Target + data1_x
print('Original X Y: ', data1_xy, '\n')


#define x variables for original w/bin features to remove continuous variables
data1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']
data1_xy_bin = Target + data1_x_bin
print('Bin X Y: ', data1_xy_bin, '\n')


#define x and y variables for dummy features original
data1_dummy = pd.get_dummies(data1[data1_x])
data1_x_dummy = data1_dummy.columns.tolist()
data1_xy_dummy = Target + data1_x_dummy
print('Dummy X Y: ', data1_xy_dummy, '\n')



data1_dummy.head()
```
### 3.24 二次检查数据
现在对我们完成清洗的数据再检查一遍吧。
```python
print('Train columns with null values: \n', data1.isnull().sum())
print("-"*10)
print (data1.info())
print("-"*10)

print('Test/Validation columns with null values: \n', data_val.isnull().sum())
print("-"*10)
print (data_val.info())
print("-"*10)

data_raw.describe(include = 'all')
```

### 3.25 切分训练集和测试集

如前所述，提供的测试文件实际上是本次竞赛所要提交的验证数据。 因此，我们将使用sklearn函数将训练数据分成两个数据集; 75/25分割。 这很重要，我们不要过度使用我们的模型。 这意味着，该算法对于给定的子集是特定的，不能从相同的数据集准确地概括另一个子集。 重要的是我们的算法没有看到我们将用来测试的子集，所以它不会通过记住答案来“作弊”。 我们将使用sklearn的train_test_split函数。 在后面的章节中，我们还将使用sklearn的交叉验证函数，将我们的数据集分解为训练集和测试集以进行数据建模比较。

```python
#split train and test data with function defaults
#random_state -> seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation
train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc], data1[Target], random_state = 0)
train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin], data1[Target] , random_state = 0)
train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], data1[Target], random_state = 0)


print("Data1 Shape: {}".format(data1.shape))
print("Train1 Shape: {}".format(train1_x.shape))
print("Test1 Shape: {}".format(test1_x.shape))

train1_x_bin.head()
```

## Step 4: 用统计学进行探索性分析

现在我们的数据已经清理完毕，我们将用描述性和图形化的统计数据来探索我们的数据来描述和总结我们的变量特征。 在这个阶段，你会发现自己将要分类特征，并确定它们与目标变量之间的相互关系。

```python
#Discrete Variable Correlation by Survival using
#group by aka pivot table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html
for x in data1_x:
    if data1[x].dtype != 'float64' :
        print('Survival Correlation by:', x)
        print(data1[[x, Target[0]]].groupby(x, as_index=False).mean())
        print('-'*10, '\n')
        

#using crosstabs: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.crosstab.html
print(pd.crosstab(data1['Title'],data1[Target[0]]))
```