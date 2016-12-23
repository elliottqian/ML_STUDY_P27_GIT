#!/home/elliottqian/d/ubuntu/anaconda3/bin/python3.5
# -*- coding:utf-8 -*-s


import numpy as np
import matplotlib.pyplot as plt

"""
❶首先载入matplotlib的绘图模块pyplot,并且重命名为plt。
pylab模块
matplotlib还提供了一个名为pylab的模块,其中包括了许多NumPy和pyplot模块中常用的函数,方便用戶快速进行计算和
绘图,十分适合在IPython交互式环境中使用。本书使用下面的方式载入pylab模块:
>>> import pylab as pl


❷调用figure()创建一个Figure(图表)对象,并且它将成为当前Figure对象。也可以不创建Figure对象直接调用接下来的plot()进行绘图,这时matplotlib会自动创建一个Figure对象。figsize参数指定Figure对象的宽度和高度,其单位为英寸。此外还可
以用dpi参数指定Figure对象的分辨率,即每英寸所表示的像素数,这里使用缺省值80。因此本例中所创建的Figure对象的宽
度为“8*80 = 640”个像素。
但是在显示出绘图窗口之后,用工具栏中的保存按钮将图表保存为图像时,所保存的图像的大小
是“800*400”像素。这是因为保存图像时会使用不同的dpi设置。这个设置保存在matplotlib的配置文件中,我们可以通过如下
语句查看它的值:
>>> import matplotlib
>>> matplotlib.rcParams["savefig.dpi"]
100
因为保存图像时的DPI设置为100,因此所保存的图像的宽度是“8*100 = 800”个像素。
rcParams是一个字典,其中保存着从
配置文件读入的所有配置,在调用各种绘图函数时,这些配置将会作为各种参数的缺省值。后面我们还会对matplotlib的配置
文件进行详细介绍。


❸创建Figure对象之后,接下来调用plot()在当前的Figure对象中绘图。实际上plot()是在Axes(子图)对象上绘图,如果当前的
Figure对象中没有Axes对象,将会为之创建一个几乎充满整个图表的Axes对象,并且使此Axes对象成为当前的Axes对
象。plot()的前两个参数是分别表示X、Y轴数据的对象,这里使用的是NumPy数组。使用关键字参数可以指定所绘制的曲线
的各种属性:
label:给曲线指定一个标签名称,此标签将在图示中显示。如果标签字符串的前后有字符’$’,则matplotlib会使用其内嵌
的LaTex引擎将其显示为数学公式。
color:指定曲线的颜色,颜色可以用英文单词,或者以’#’字符开头的三个16进制数,例如’#ff0000’表示红色。或者使用
值在0到1范围之内的三个元素的元组表示,例如(1.0, 0.0, 0.0)也表示红色。
linewidth:指定曲线的宽度,可以不是整数,也可以使用缩写形式的参数名lw。
使用LaTex语法绘制数学公式会极大地降低图表的描绘速度。


❹直接通过第三个参数’b–’指定曲线的颜色和线型,它通过一些易记的符号指定曲线的样式。其中’b’表示蓝色,’–’表示线型
为虚线。在IPython中输入“plt.plot?”可以查看格式化字符串以及各个参数的详细说明。

❺接下来通过一系列函数设置当前Axes对象的各个属性:
xlabel、ylabel:分别设置X、Y轴的标题文字。
title:设置子图的标题。
xlim、ylim:分别设置X、Y轴的显示范围。
legend:显示图示,即图中表示每条曲线的标签(label)和样式的矩形区域。

❻最后调用plt.show()显示出绘图窗口。在通常的运行情况下,show()将会阻塞程序的运行,直到用戶关闭绘图窗口。然而
在带“-wthread”等参数的IPython环境下,show()不会等待窗口关闭。
"""

if __name__ == "__main__":

    """创建两个函数"""
    x = np.linspace(0, 10, 1000)
    y = np.sin(x)
    z = np.cos(x ** 2)

    """图像大小"""
    plt.figure(figsize=(8, 4))

    """两条线"""
    plt.plot(x, y, label="$sin(x)$", color="red", linewidth=2)
    plt.plot(x, z, "b--", label="$cos(x^2)$")

    """标题名称"""
    plt.xlabel("Time(s)")
    plt.ylabel("Volt")
    plt.title("PyPlot First Example")

    """y轴范围和图示"""
    plt.ylim(-1.2, 1.2)
    plt.legend()

    """展示"""
    plt.show()
    pass