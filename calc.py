#------------UI------------------#
from tkinter import *
import tkinter
import tkinter.messagebox
from tkinter import ttk
#------------Package--------------------#
import sys
import re
import numpy as np
import scipy as sp
import scipy
import numpy
import scipy.stats as st
import matplotlib as mp
import matplotlib.pyplot as plt
import math
import pandas as pd
from pandas import read_csv
import statsmodels.stats.api as sms
import random
from scipy.stats import ks_2samp
#------------PLOT--------------------#
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
#------------DATA frame-------------------#
from sklearn.svm import SVC
from sklearn.externals import joblib
import pickle
#------------Normal tests---------"
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson
from scipy.stats import ttest_ind
from scipy.stats import t
#------------random data----------#
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import std
#------------KW-----------------#
from scipy.stats import kruskal
#------------Wilcox Signed Rank Test--------#
from scipy.stats import wilcoxon
#------------Leven´s Test----------------#
import scipy.stats as stats
#------------TIME----------------------#
import time

#--------------Functions---------------------#
def read1(f):                      #----Read data txt----#
    h = f.read()
    g = re.findall('\d+,\d+',h)
    for y in range(0, len(g)):
        g[y] = g[y].replace(",",".")

    for y in range(0, len(g)):
        g[y] = float(g[y])
    print("data read")
    
    return g
def read2(f):                      #----Read data excel----#

    #excel_file_1 = "f" 
    #df_all = pd.concat([df_firstdata, df_seconddata])
    #print(df_all)
    df = pd.read_excel(f, sheet_name = 0)
    
    print(df)
    print("data read")
    np_array = df.to_numpy()
    return np_array
def read3(f):                      #----Read data csv----#

    dataset = read_csv(f)
    array = dataset.values
    
    print("data read")
    return array
def proces():                      #----Calculator----#
    number1=Entry.get(E1)
    number2=Entry.get(E2)
    operator=Entry.get(E3)
    number1=float(number1)
    number2=float(number2)
    answer=0
    if operator =="+":
        answer=number1+number2
    if operator =="-":
        answer=number1-number2
    if operator=="*":
        answer=number1*number2
    if operator=="/":
        answer=number1/number2
    if operator=="**":
        answer=number1**number2
    Entry.insert(E4,0,answer)
    print(answer)
def makecsv(data):                 #----CSV----#

    ss = pd.DataFrame(data)
    ss.to_csv('ss.csv', index=False, header=False, sep=',')
    dataset = read_csv("ss.csv")
    return dataset   
def data():                        #----Take paths and send to read----#
    path1=Entry.get(G1)
    path2=Entry.get(G2)
    path3=Entry.get(G3)
    path4=Entry.get(G7)
    path5=Entry.get(G8)
    
    global g
    global i
    global k
    
    i#f path1.endswith("txt") == True or path2.endswith("txt") == True or path3.endswith(".txt") == True: 
    if path1.endswith("txt") == True:
        f = open(path1,"r")
        g = read1(f)
        bot = tkinter.Tk()
        L12 = Label(bot, text="Data read: " + str(path1),).grid(row=0,column=1)
    if path2.endswith("txt") == True:
        h = open(path2,"r")
        i = read1(h)
    if path3.endswith("txt") == True:
        j = open(path3, "r")
        k = read1(j)
            
    #if path1.endswith("xlsx") == True or path2.endswith("xlsx") == True or path3.endswith("xlsx") == True:
    if path1.endswith("xlsx") == True:
        g = read2(path1)
        bot = tkinter.Tk()
        L12 = Label(bot, text="Data read: " + str(path1),).grid(row=0,column=1)
    if path2.endswith("xlsx") == True:
        i = read2(path2)
    if path3.endswith("xlsx") == True:
        k = read2(path3)
            
    #if path1.endswith("csv") == True or path2.endswith("csv") == True or path3.endswith("csv") == True:
    if path1.endswith("csv") == True:
        g = read3(path1)
        bot = tkinter.Tk()
        L12 = Label(bot, text="Data read: " + str(path1),).grid(row=0,column=1)
    if path2.endswith("csv") == True:
        i = read3(path2)
    if path3.endswith("csv") == True:
        k = read3(path3)
def stats2():                      #----Generate std, mean & variance----#
    global g
    global i
    global k
    global mean
    global std
    global variance
    global mean2
    global std2
    global variance2
    global mean3
    global std3
    global variance3
    
    if g != "":
        mean, std, variance = stats1(g)
        Entry.insert(G4,0,variance)
        Entry.insert(G5,0,std)
        Entry.insert(G6,0,mean)
    if i != "":
        mean2, std2, variance2 = stats1(i)
    if k != "":
        mean3, std3, variance3 = stats1(k)
    
   
    print("\n")
def stats1(list):                  #----Generate std, mean & variance----#
    
    global samplesize
    
#---------------MEAN---------------#

    o = 0
    o = sum(list)
    mean = float(o/len(list))

#--------------SSe-----------------#

    sse = 0

    for y in range(0, len(list)):
        sse = sse + ((list[y]-mean)**2)

#-----------variance--------------#

    variance = float(sse/(len(list)))

#-----------std----------------#

    std = float(math.sqrt(variance))
    samplesize = len(list)
    
    print("Variance: " + str(variance))
    print("Standard deviation: " + str(std))
    print("Mean value: " + str(mean))
    print("sample size: " +str(samplesize))
    print("\n")
    return mean, std, variance   
def stdtot1():                     #----Generate std, mean & variance total----#
    global g
    global i
    global k
    global stdtot
    global meantot
    global samplesize
    global nparray
    if k == "":
        a = array1(g)
        b = array1(i)
        c = np.vstack((a,b,))
    
    if k != "":
        a = array1(g)
        b = array1(i)
        d = array1(k)
        c = np.vstack((a,b,d))
        
    c = c.reshape(-1)
#-----------variance--------------#
    nparray = c
    variance = np.var(c)

#-----------std----------------#
    meantot = np.mean(c)
    stdtot = math.sqrt(variance)
    samplesize = len(c)
    e = numpy.arange(1, len(c)+1, 1)
    
    print(samplesize)
    print("stdtot:",stdtot)
    print("meantot",meantot)
    print("\n")
    plt.plot(e,c,"bs")
    plt.show()   
def ci():                          #----Generate CI----#
    global g
    global i
    global k
    
    if g != "":
        ci2(g)
        
    if i != "":
        ci2(i)
    if k != "":
        ci2(k)
def ci2(g):                        #----Generate CI----#
   a = st.t.interval(0.995, len(g)-1, loc=np.mean(g), scale=st.sem(g))
   b = sms.DescrStatsW(g).tconfint_mean()
   bot = tkinter.Tk()
   L12 = Label(bot, text="CI computed"+ str(b),).grid(row=0,column=1)
   print(a)
   print(b)
   print("\n")
def array1(g):                     #----Make array from list----#

   # names = ["vikt"]
    #a = 0
    #data = np.zeros((200,1))
    data = np.array(g)

    #for x in range(0,len(g)):
        #data[x] = g[x]
        
    #print(a)
    #print(len(data))
    return data
def fit(data):                     #----Scipy fit & QQ plot ----#

    _, bins, _ = plt.hist(data, 15, density=1, alpha=0.5)
    mu, sigma = scipy.stats.norm.fit(data)
    best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
    plt.plot(bins, best_fit_line)
    plt.savefig("Histogram2.png")



    qqplot(data, line='s')
    pyplot.show()
def fitdata1():                    #----FIT DATA----#

    global g
    
    if g != "":
        a = array1(g)
    if a != "":
        fit(a)
def fitdata2():                    #----FIT DATA----#
    
    global i
    
    if i != "":
        b = array1(i)
    if b != "":
        fit(b)
def fitdata3():                    #----FIT DATA----#
    
    global k
    
    if k != "":
        c = array1(k)
    if c != "":
        fit(c)
def test(data):                    #----Test for Gaussian dist & Normal test ----#
    print("\n")
    print("Shapiro test")
    stat, p = shapiro(data)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    
    
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')
    
    print("\n")
    print("Normal test:")
    stat, p = normaltest(data)
    print('Statistics=%.3f, p=%.3f' % (stat, p))

    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H:0)')
    else:
        print('Sample does not look Gaussian (reject H:0)')
    print("\n")
    
    for y in range(0, len(g)):
        g[y] = float(g[y])
       
    result = anderson(g)
    print('Statistic: %.3f' % result.statistic)
    p = 0
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < result.critical_values[i]:
            print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
        else:
            print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
    print("\n")
def test1():                       #----Send to Gaussin test-----------------#
    global g
    global i
    global k
    
    if g != "":
        a = array1(g)
        test(a)
    if i != "":
        b = array1(i)
        test(b)
    if k != "":
        c = array1(i)
        test(c)
def optest(mean,std):              #----One population mean Hypotes  z test----#
    global g
    u = 1.10
    n = math.sqrt(len(g))

    z =math.sqrt(((mean-u)/(std/n))**2)

    Z = 2.81 #99.5 % confidence ( alpha = 0.005 / 2 ) = 0.0025 for each side p(-z < Z < z ) = 0.995

    print("Hypotes test mean = 1.10 (99,9% significance) ")
    

    if Z > z:
        print(str(z) +" mindre än " + str(Z))
        print("Hypotes H:0 not rejected, mean 1.10")
    if Z < z:
        print(str(Z) +" mindre än " + str(z))
        print("Hypotes H:0 rejected, mean not 1.10")
    print("\n")   
def optest1():
    global mean
    global std
    global variance
    global mean2
    global std2
    global variance2
    global mean3
    global std3
    global variance3
    
    if mean != "":
        optest(mean,std)
        
    if mean2 != "":
        optest(mean2,std2)
        
    if mean3 != "":
        optest(mean3,std3)
    if mean !="" or mean2 !="" or mean3 !="":
        bot = tkinter.Tk()
        L12 = Label(bot, text="One population Z test of H:0 computed",).grid(row=0,column=1)
    if mean == "" and mean2 == "" and mean3 == "":
         bot = tkinter.Tk()
         L12 = Label(bot, text="Calculate mean & std first",).grid(row=0,column=1)
def kw2(a,b):                      #----Krukskal wallice----_#
    #seed(1)
    #data1 = 5 * randn(100) + 50
    #data2 = 5 * randn(100) + 50
    #data3 = 5 * randn(100) + 52
    stat, p = kruskal(a,b)
    print("Krukskal Wallice test")
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    print("\n")
def kw3(a,b,c):                    #----Krukskal wallice----_#
    #seed(1)
    #data1 = 5 * randn(100) + 50
    #data2 = 5 * randn(100) + 50
    #data3 = 5 * randn(100) + 52
    stat, p = kruskal(a,b,c)
    print("Krukskal Wallice test")
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    print("\n")
def kw():
    global g
    global i
    global k
    
    if g != "":
        a = array1(g)
    if i != "":
        b = array1(i)
    if k != "":
        c = array1(k)
    
    if g != "" and i != "" and k == "":
        kw2(a,b)
    if g != "" and i != "" and k != "":
        kw3(a,c,b)   
def lev2(a,b):                     #----Leven´s----------#
    #seed(1)
    #data1 = 5 * randn(100) + 50
    #data2 = 5 * randn(100) + 50
    #data3 = 5 * randn(100) + 52 
    print("leven´s test")
    print(stats.levene(a,b, center='median'))
    print(stats.levene(a,b, center='mean'))   #Mean trimmed opt for center#
    print("\n")
def lev3(a,b,c):                   #----Leven´s----------#
    seed(1)
    #data1 = 5 * randn(100) + 50
    #data2 = 5 * randn(100) + 50
    #data3 = 5 * randn(100) + 52 
    print("leven´s test")
    print(stats.levene(a,b,c, center='median'))
    print(stats.levene(a,b,c, center='mean'))   #Mean trimmed opt for center#
    print("\n")
def lev():
    global g
    global i
    global k
    
    if g != "":
        a = array1(g)
    if i != "":
        b = array1(i)
    if k != "":
        c = array1(k)
    
    if g != "" and i != "" and k == "":
        lev2(a,b)
    if g != "" and i != "" and k != "":
        lev3(a,c,b)
def wrst2(a,b):                    #----Wilcoxons signed rank test----#
    #seed(1) 
    #data1 = 5 * randn(100) + 50
    #data2 = 5 * randn(100) + 51 
    
    stat, p = wilcoxon(a, b)
    print("Wilcoxon signed rank test")
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')
    print("\n") 
def wsrt():                        #----Wilcoxons signed rank test----#
    global g
    global i
    global k
    
    if g != "":
        a = array1(g)
    if i != "":
        b = array1(i)
    if k != "":
        c = array1(k)
    
    if g != "" and i != "" and k == "":
        wrst2(a,b)
        print("\n")
    if g != "" and i != "" and k != "":
        wrst2(a,b)
        print("\n")
    if g != "" and i != "" and k != "":
        wrst2(b,c)
        print("\n")
    if g != "" and i != "" and k != "":
        wrst2(a,c)
        print("\n")
def histo(dataset):                #----Plot & save historgram----#
    dataset.hist()
    pyplot.show()
    plt.hist(data,15)
    plt.savefig("Histogram.png")
def makecsv(data):                 #----CSV----#

    ss = pd.DataFrame(data)
    ss.to_csv('ss.csv', index=False, header=False, sep=',')
    dataset = read_csv("ss.csv")
    return dataset
def hist1():
    global g
    
    
    if g != "":
        a = array1(g)
        a = makecsv(a)
        histo(a)
def hist2():
    
    global i
        
    if i != "":
        b = array1(i)
        b = makecsv(b)
        histo(b)
def hist3():
    
    global k
    
    
    if k != "":
        c = array1(k)
        c = makecsv(c)
        histo(c)  
def ss1():                         #----Sample_size calculation----#

    
    print("\n")
    
    N = 7000
    z = 1.96
    z2 = z**2
    p = 0.999
    e = 0.025
    e2 = e**2
    p2 = 1-p

    sample_size = ((z2*p*p2)/(e2)) / (1+((z2*p*p2)/(e2*N)))
    #print(((z**2)*p*(1-p)/e**2) /(1 + ((z**2)*p*(1-p)/((e**2)*N))))
    print(sample_size)
    print("\n")       
def SRT():                         #----Success-Run Therem----#
    #---- 95% conf, 90% Reliability-----#
    a = math.log(1-0.95)
    b = math.log(0.99)
    n = a/b
    bot = tkinter.Tk()
    L12 = Label(bot, text="Sample size = "+ str(n),).grid(row=0,column=1)
    print(n)
def SRT2():
    #95% conf, 90% Reliability,df = 2(r+1) = 3, X^2 = 7.825
    a = 0.5*7.825
    b = 1-0.9
    n = a/b
    bot = tkinter.Tk()
    L12 = Label(bot, text="Sample size = "+ str(n),).grid(row=0,column=1)
    print(n)
    print(n)
    
def cpktotLog():
    global nparray
    global samplesize
    
    nparray = np.log(nparray)
    meantot = np.mean(nparray)
    stdtot = np.std(nparray)
    UL = np.log(1.1275)
    LL = np.log(1.0725)
    if meantot - LL <= UL - meantot:
        cpkvalue = (meantot - LL) / (3*stdtot)
    else:
        cpkvalue = (UL - meantot) / (3*stdtot)
    print("Cpk value(Log Log): ",cpkvalue)
    x = 2.58
    y = samplesize
    L = cpkvalue - (x*(math.sqrt((1/(9*y))+((cpkvalue**2)/(2*y-2)))))
    print("Lower bond Cpk 99 % confidence",L)
    width = cpkvalue - L
    print("width of confidence interval",width)
    print("\n")
def cpktot():
    global meantot
    global stdtot
    global samplesize
    #Cpk = (Mean-LSL) / (3 Stdev)
    if meantot - 1.0725 <= 1.1275 - meantot:
        cpkvalue = (meantot - 1.0725) / (3*stdtot)
    else:
        cpkvalue = (1.1275 - meantot) / (3*stdtot)
    print("Cpk value: ",cpkvalue)
    x = 1.96
    y = samplesize
    L = cpkvalue - (x*(math.sqrt((1/(9*y))+((cpkvalue**2)/(2*y-2)))))
    print("Lower bond Cpk, 99 % confidence",L)
    width = cpkvalue - L
    print("width of confidence interval",width)
    print("\n")
def cpk1PK():
    global mean
    global std
    global samplesize
    #X = Z-value, 80% - 1,28, 90 % -1,645, 95% - 1,96. 98% - 2,33, 99% - 2,58
    
    x = 1.96
    y = float(samplesize)
    
    
    if mean - 3.4125 <= 3.5875 - mean:
        cpkvalue = (mean - 3.4125) / (3*std)
    else:
        cpkvalue = (3.5875 - mean) / (3*std)
    print("Cpk value: ",cpkvalue)
    L = cpkvalue - (x*(math.sqrt((1/(9*y))+((cpkvalue**2)/(2*y-2)))))
    print("Lower bond Cpk, 95 % confidence",L)
    width = cpkvalue - L
    print("width of confidence interval",width)
    print("\n")
def cpk2():
    
    global mean2
    global std2
    x = 1.96
    y = 200
    
    if mean2 - 1.0725 <= 1.1275 - mean2:
        cpkvalue = (mean2 - 1.0725) / (3*std2)
    else:
        cpkvalue = (1.1275 - mean2) / (3*std2)
    print("Cpk value: ",cpkvalue)
    L = cpkvalue - (x*(math.sqrt((1/(9*y))+((cpkvalue**2)/(2*y-2)))))
    print("Lower bond Cpk, 95 % confidence",L)
    width = cpkvalue - L
    print("width of confidence interval",width)
    print("\n")
def cpk3():
    global mean3
    global std3
    x = 1.96
    y = 200
    
    if mean3 - 1.0725 <= 1.1275 - mean3:
        cpkvalue = (mean3 - 1.0725) / (3*std3)
    else:
        cpkvalue = (1.1275 - mean3) / (3*std3)
    print("Cpk value: ",cpkvalue)
    L = cpkvalue - (x*(math.sqrt((1/(9*y))+((cpkvalue**2)/(2*y-2)))))
    print("Lower bond Cpk, 95 % confidence",L)
    width = cpkvalue - L
    print("width of confidence interval",width)
    print("\n")
def cpk1APTT():
    global mean
    global std
    global samplesize
    #X = Z-value, 80% - 1,28, 90 % -1,645, 95% - 1,96. 98% - 2,33, 99% - 2,58
    
    x = 1.96
    y = float(samplesize)
    
    
    if mean - 10.00 <= 10.20 - mean:
        cpkvalue = (mean - 10.00) / (3*std)
    else:
        cpkvalue = (10.20 - mean) / (3*std)
    print("Cpk value: ",cpkvalue)
    L = cpkvalue - (x*(math.sqrt((1/(9*y))+((cpkvalue**2)/(2*y-2)))))
    print("Lower bond Cpk, 95 % confidence",L)
    width = cpkvalue - L
    print("width of confidence interval",width)
    print("\n")
def cpk1NKP():
    global mean
    global std
    global samplesize
    #X = Z-value, 80% - 1,28, 90 % -1,645, 95% - 1,96. 98% - 2,33, 99% - 2,58
    
    x = 1.96
    y = float(samplesize)
    
    
    if mean - 1.0725 <= 1.1275 - mean:
        cpkvalue = (mean - 1.0725) / (3*std)
    else:
        cpkvalue = (1.1275 - mean) / (3*std)
    print("Cpk value: ",cpkvalue)
    L = cpkvalue - (x*(math.sqrt((1/(9*y))+((cpkvalue**2)/(2*y-2)))))
    print("Lower bond Cpk, 95 % confidence",L)
    width = cpkvalue - L
    print("width of confidence interval",width)
    print("\n")
def cpk1OKP():
    global mean
    global std
    global samplesize
    #X = Z-value, 80% - 1,28, 90 % -1,645, 95% - 1,96. 98% - 2,33, 99% - 2,58
    
    x = 1.96
    y = float(samplesize)
    
    
    if mean - 0.5265 <= 0.5535 - mean:
        cpkvalue = (mean - 0.5265) / (3*std)
    else:
        cpkvalue = (0.5535 - mean) / (3*std)
    print("Cpk value: ",cpkvalue)
    L = cpkvalue - (x*(math.sqrt((1/(9*y))+((cpkvalue**2)/(2*y-2)))))
    print("Lower bond Cpk, 95 % confidence",L)
    width = cpkvalue - L
    print("width of confidence interval",width)
    print("\n")
    
def tptest():                      #----Two pop mean test----#
    global g
    global i
    data1 = array1(g)
    data2 = array1(i)
    tptest1(data1,data2)
def tptest1(data1,data2):
    #------compare means between two populations---#
    alpha = 0.95
    #calculate means
    mean1, mean2 = scipy.mean(data1), scipy.mean(data2)
    #calculate standard errors
    se1, se2 = stats.sem(data1), stats.sem(data2)
	# standard error on the difference between the samples
    sed = math.sqrt(se1**2.0 + se2**2.0)
	# calculate the t statistic
    t_stat = (mean1 - mean2) / sed
	# degrees of freedom
    df = len(data1) + len(data2) - 2
	# calculate the critical value
    cv = t.ppf(1.0 - alpha, df)
	# calculate the p-value
    p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
	# return everything
    print("T-stats",t_stat)
    print("DF",df)
    print("CV",cv)
    print("P-value",p)
    if abs(t_stat) <= cv:
        print('Accept null hypothesis that the means are equal.')
    else:
        print('Reject the null hypothesis that the means are equal.')
def cp():
    global stdtot
    global samplesize
    
    if stdtot != "":
        cp = (3.5875-3.4125) / (6*float(stdtot))
    else:
        cp = (3.5875-3.4125) / (6*float(std))
    print("Cp value: ",str(cp))
    print("\n")
def random1337():
    
    data1 = np.random.randint(10, size=1) 
    
    if data1[0] == 1:
        print("du fick siffra ",data1, "funktion Cp")
        cp()
    if data1[0] == 2:
        print("du fick siffra ",data1, "funktion Ci")
        ci()
    if data1[0] == 3:
        print("du fick siffra ",data1, "funktion Histogram")
        hist1()
    if data1[0] == 4:
        print("du fick siffra ",data1, "funktion SRT")
        SRT()
    if data1[0] == 5:
        print("du fick siffra ",data1, "funktion Fitdata")
        fitdata1()
    if data1[0] == 6:
        print("du fick siffra ",data1, "funktion Stats")
        stats2()
    if data1[0] == 7:
        print("du fick siffra ",data1, "funktion One population test")
        optest1()
    if data1[0] == 8:
        print("du fick siffra ",data1, "funktion Cpk")
        cpk1()
    if data1[0] == 9:
        print("du fick siffra ",data1, "funktion ?")
        random1337()
    print("\n")
def kstp1(data1,data2):            #----Kolmogorov-Smirnov Two population test----#  
    print(ks_2samp(data1, data2))
    print("if p value < alpha, reject hypothesis that samples comes from the same distribution")
def ksop1(data):                   #----Kolmogorov-Smirnov one population test----#
    print("One sample Kolmogorov-Smirnov test")
    q = stats.kstest(data, 'norm')
    print(q)
    print("if p-value < alpha, reject normal dist hypothesis")
    print("\n")
def ksop():
    global g
    global i
    global k
    
    if g != "":
        ksop1(g)
    if i != "":
        ksop1(i)
    if k != "":
        ksop1(k)
def kstp():
    global g
    global i
    global k
    
    if g != "" and i !="":
        kstp1(g,i)
    if i != "" and k !="":
        kstp1(i,k)
    if k != "" and g !="":
        kstp1(k,g)
def plot1():
    global g
    global mean
    global std
    data = array1(g)
    y = np.arange(0, len(g), 1)
    
    print(len(data))
    fig, axs = plt.subplots(2, 2, figsize=(5, 5))
    axs[0, 0].hist(data)
    axs[1, 0].scatter(data,y)
    axs[0, 1].plot(data,y)
    axs[1, 1].hist2d(data,y)
    plt.show()  
def plotN():
    global g
    global mean
    global std
    
    variance = std**2
    x = np.arange(1.0725,1.1275,.01)
    f = np.exp(-np.square(x-mean)/2*variance)/(np.sqrt(2*np.pi*variance))

    plt.plot(x,f)
    plt.ylabel('gaussian distribution')
    plt.show()
def plot2():
    global i
    data = array1(i)
    y = np.arange(0, len(i), 1)
    
    print(len(data))
    fig, axs = plt.subplots(2, 2, figsize=(5, 5))
    axs[0, 0].hist(data)
    axs[1, 0].scatter(data,y)
    axs[0, 1].plot(data,y)
    axs[1, 1].hist2d(data,y)
    plt.show()   
def plot3():

    global k
    data = array1(k)
    y = np.arange(0, len(k), 1)
    
    print(len(data))
    fig, axs = plt.subplots(2, 2, figsize=(5, 5))
    axs[0, 0].hist(data)
    axs[1, 0].scatter(data,y)
    axs[0, 1].plot(data,y)
    axs[1, 1].hist2d(data,y)
    plt.show()
def quit():                        #----KILL MID------------------------------#
        mid.destroy()
def export():
    pass
def quit2():                       #----KILL TOP------------------------------#
        top.destroy()        

#--------------------Pop up---------------------#

def view1():
    mid = tkinter.Tk()
    scrollbar = Scrollbar(mid)
    scrollbar.pack(side=RIGHT, fill=Y)

    listbox = Listbox(mid, yscrollcommand=scrollbar.set)
    for y in range(len(g)):
        listbox.insert(END, str(g[y]))
    listbox.pack(side=LEFT, fill=BOTH)

    scrollbar.config(command=listbox.yview)
def view2():
    mid = tkinter.Tk()
    scrollbar = Scrollbar(mid)
    scrollbar.pack(side=RIGHT, fill=Y)

    listbox = Listbox(mid, yscrollcommand=scrollbar.set)
    for y in range(len(i)):
        listbox.insert(END, str(i[y]))
    listbox.pack(side=LEFT, fill=BOTH)

    scrollbar.config(command=listbox.yview)      
def view3():
    mid = tkinter.Tk()
    scrollbar = Scrollbar(mid)
    scrollbar.pack(side=RIGHT, fill=Y)

    listbox = Listbox(mid, yscrollcommand=scrollbar.set)
    for y in range(len(k)):
        listbox.insert(END, str(k[y]))
    listbox.pack(side=LEFT, fill=BOTH)

    scrollbar.config(command=listbox.yview)

#---------------------MAIN----------------------# 

if __name__ == '__main__':
    #-------------Global variables--------------#
    g = ""
    i = ""
    k = ""
    mean = ""
    std = ""
    variance = ""
    mean2 = ""
    std2 = ""
    variance2 = ""
    mean3 = ""
    std3 = ""
    variance3 = ""
    meantot = ""
    stdtot = ""
    samplesize = 0
    nparray = ""
    
    
    #------------------------------------------UI LAYERS--------------------#
    #------------------------------------------TOP(Root)--------------------#

    top = tkinter.Tk()
    top.title("Stefans Statistical Software V. Alpha 0.1")
    top.configure(bg="azure")


    #------------------------------------------LABLES----------------------------------------------#
    L1 = Label(top, text="Inputs",bg="aquamarine").grid(row=0,column=1)

    L6 = Label(top, text="Path to Data 1",bg="light cyan2").grid(row=5,column=0)
    L7 = Label(top, text="Path to Data 2",bg="light cyan2").grid(row=6, column=0)
    L8 = Label(top, text="Path to Data 3",bg="light cyan2").grid(row=7, column=0)
    L12 = Label(top, text="Path to Data 4",bg="light cyan2").grid(row=8, column=0)
    L13 = Label(top, text="Path to Data 5",bg="light cyan2").grid(row=9, column=0)
    L14 = Label(top, text="Outputs",bg="aquamarine").grid(row=10,column=1)
    L9 = Label(top, text="Variance",bg="light cyan3").grid(row=11, column=0)
    L10 = Label(top, text="STD",bg="light cyan3").grid(row=12, column=0)
    L11 = Label(top, text="Mean",bg="light cyan3").grid(row=13, column=0)


    #-------------------------------------------ENTRYS-----------------------------------------------------------#
    G1 = Entry(top, bd =5)
    G1.grid(row=5, column=1)
    G2 = Entry(top, bd=5)
    G2.grid(row=6, column=1)
    G3 = Entry(top, bd=5)
    G3.grid(row=7, column=1)
    G7 = Entry(top, bd=5)
    G7.grid(row=8, column=1)
    G8 = Entry(top, bd=5)
    G8.grid(row=9, column=1)
    G4 = Entry(top, bd=10)
    G4.grid(row=11, column=1)
    G5 = Entry(top, bd=10)
    G5.grid(row=12, column=1)
    G6 = Entry(top, bd=10)
    G6.grid(row=13, column=1)

    #-------------------------------------------BUTTONS--------------------------------------------------------------------------#
    
    #B=Button(top, text ="         Calculator        ",width=25,height=1,command = calculator, bg="cadetblue1").grid(row=4,column=2,)
    C=Button(top, text ="          Read data        ",width=25,height=1,command = data, bg="floral white").grid(row=5,column=2,)
    D=Button(top, text ="    Variance, Std & Mean   ",width=25,height=1,command = stats2, bg="floral white").grid(row=5,column=3,)
    E=Button(top, text ="            CI             ",width=25,height=1,command = ci, bg="RosyBrown3").grid(row=12,column=3,)
    F=Button(top, text ="        FIT data 1         ",width=25,height=1,command = fitdata1, bg="linen").grid(row=6,column=2,)
    G=Button(top, text ="        FIT data 2         ",width=25,height=1,command = fitdata2, bg = "linen").grid(row=6,column=3,)
    H=Button(top, text ="        FIT data 3         ",width=25,height=1,command = fitdata3, bg="linen").grid(row=6,column=4,)
    I=Button(top, text ="          N test           ",width=25,height=1,command = test1, bg="papaya whip").grid(row=7, column=2,)
    J=Button(top, text ="  One pop H:0 test, Z dist ",width=25,height=1,command = optest1, bg="papaya whip").grid(row=7, column=3,)
    Q=Button(top, text ="   Samplesize, Proportion  ",width=25,height=1,command = ss1, bg="papaya whip").grid(row=7, column=4,)
    K=Button(top, text ="   Krukskal Wallice Test   ",width=25,height=1,command = kw, bg="bisque").grid(row=8, column=2,)
    L=Button(top, text ="        Leven´s Test       ",width=25,height=1,command = lev, bg="bisque").grid(row=8, column=3,)
    M=Button(top, text =" Wilcoxons Signed Rank Test",width=25,height=1,command = wsrt, bg="bisque").grid(row=8, column=4,)
    N=Button(top, text ="       Histogram data 1    ",width=25,height=1,command = hist1, bg="navajo white").grid(row=9, column=2,)
    O=Button(top, text ="       Histogram data 2    ",width=25,height=1,command = hist2, bg="navajo white").grid(row=9, column=3,)
    P=Button(top, text ="       Histogram data 3    ",width=25,height=1,command = hist3, bg="navajo white").grid(row=9, column=4,)
    Q=Button(top, text ="Success-Run theorem, 0 fail",width=25,height=1,command = SRT, bg="RosyBrown1").grid(row=10, column=2,)
    R=Button(top, text ="Success-Run theorem, x fail",width=25,height=1,command = SRT2, bg="RosyBrown1").grid(row=10, column=3,)
    X=Button(top, text ="            Random Func    ",width=25,height=1,command = random1337, bg="RosyBrown1").grid(row=10, column=4,)
    S=Button(top, text ="          CPK total        ",width=25,height=1,command = cpktot, bg="RosyBrown2").grid(row=11, column=2,)
    T=Button(top, text ="            CPK PK          ",width=25,height=1,command = cpk1PK, bg="RosyBrown2").grid(row=12, column=2,)
    U=Button(top, text ="            CPK 2          ",width=25,height=1,command = cpk2, bg="RosyBrown2").grid(row=12, column=3,)
    AJ=Button(top, text ="            CPK 3          ",width=25,height=1,command = cpk3, bg="RosyBrown2").grid(row=12, column=4,)
    V=Button(top, text ="       Two pop t test      ",width=25,height=1,command = tptest, bg="RosyBrown3").grid(row=13, column=2,)
    C=Button(top, text ="       STD & mean tot      ",width=25,height=1,command = stdtot1, bg="floral white").grid(row=5,column=4,)
    W=Button(top, text ="            Cp             ",width=25,height=1,command = cp, bg="RosyBrown2").grid(row=11,column=4,)
    AA=Button(top, text="         KS one pop        ",width=25,height=1,command = ksop,bg="RosyBrown4").grid(row=14,column=2,)
    AB=Button(top, text="         KS two pop        ",width=25,height=1,command = kstp,bg="RosyBrown4").grid(row=14,column=3,)
    AC=Button(top, text="            Cpk APTT            ",width=25,height=1,command = cpk1APTT,bg="RosyBrown3").grid(row=13,column=3,)
    Y=Button(top, text= "         View data 1       ",width=25,height=1,command = view1,bg="lightpink4").grid(row=15,column=2,)
    Z=Button(top, text= "         View data 2       ",width=25,height=1,command = view2,bg="lightpink4").grid(row=15,column=3,)
    Å=Button(top, text= "         View data 3       ",width=25,height=1,command = view3,bg="lightpink4").grid(row=15,column=4,)
    AD=Button(top, text="         Cpk tot Log       ",width=25,height=1,command = cpktotLog,bg="RosyBrown2").grid(row=11,column=3,)
    AE=Button(top, text="            Plots 1        ",width=25,height=1,command = plot1,bg="RosyBrown2").grid(row=16,column=2,)
    AF=Button(top, text="            Plots 2        ",width=25,height=1,command = plot2,bg="RosyBrown2").grid(row=16,column=3,)
    AG=Button(top, text="            Plots 3        ",width=25,height=1,command = plot3,bg="RosyBrown2").grid(row=16,column=4,)
    AH=Button(top, text ="            Stäng         ",width=25,height=1,command = quit2, bg="red").grid(row=19,column=2,)
    AI=Button(top, text ="       Plot Normaldist    ",width=25,height=1,command = plotN, bg="RosyBrown1").grid(row=18,column=2,)
    AJ=Button(top, text ="       Cpk OKP    ",width=25,height=1,command = cpk1OKP, bg="RosyBrown3").grid(row=13,column=4,)
    AK=Button(top, text ="       Cpk NKP    ",width=25,height=1,command = cpk1NKP, bg="RosyBrown4").grid(row=14,column=4,)
    #-------------------------------------------CALCULATOR----------------------------------------------------------------------------#
    
    mid = tkinter.Toplevel()
    
    L2 = Label(mid, text="Siffra 1",bg="light cyan").grid(row=1,column=0)
    L3 = Label(mid, text="Siffra 2",bg="light cyan").grid(row=2,column=0)
    L4 = Label(mid, text="Operator",bg="light cyan").grid(row=3,column=0)
    L5 = Label(mid, text="Svar",bg="light cyan").grid(row=4,column=0)
    E1 = Entry(mid, bd =5)
    E1.grid(row=1,column=1)
    E2 = Entry(mid, bd =5)
    E2.grid(row=2,column=1)
    E3 = Entry(mid, bd =5)
    E3.grid(row=3,column=1)
    E4 = Entry(mid, bd =10)
    E4.grid(row=4,column=1)
    C1=Button(mid, text ="           Beräkna         ",width=25,height=1,command = proces, bg="cadetblue1").grid(row=4,column=2,)
    C1=Button(mid, text ="            Stäng          ",width=25,height=1,command = quit, bg="red").grid(row=5,column=2,)

    #--------------------------------------------Top loop-------------------------------------------------------------------------------#
    
    top.mainloop()
    #----------------------------------------------END----------------------------------------------------------------------------------#