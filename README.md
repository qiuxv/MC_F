项目名：类的强弱关系指导对抗样本攻击

MCF

假设：只能访问模型的接口，模型黑盒。

类的强弱关系的定义：
对于任意两个类，类i和类j
假设图像C=类i的图像a*lamda+类j的图像b*（1-lamda）
P=f(lamda),0<=lamda<=1
f(lamda)为C被分类为i的概率
由于对于任意lamda，事实上P总存在
所以可以得到函数图像在定义域上是连续的。    （1）
当lamda=0时，C=类j的图像b
此时P=1;
所以（0，1）一定在图像上。           （2）
同理可以得到（1，0）一定在图像上。  （3）
由（1）（2）（3）得到
一定存在至少一个lamda使P=0.5
我们称此lamda为边界lim
如果lim>0.5,则称i类对j类是弱势的，反之是强势的
强弱的程度称为强度差=|lim-0.5|

实验过程：

·可迁移性（更换数据集后，当强度差不是很小的时候，强弱的关系不变）

用途：
对攻击方有指导意义：
优先攻击强度差大的类别对
使用普遍强势的类别图像的局部值作为扰动量
其他：
普遍弱势的类别往往准确率较低，学习不够充分，可以作为模型对类别学习程度的指征。
对fariness的折射
防御手段：
对强度较弱的类别进行进一步训练，尽量是类别间强弱均匀。



