import numpy 
from numpy import exp,array,random,dot,mean,abs


def UnitStep(x):
    if x>=0:
        return 1
    else:
        return -1

def UnitStepDerivative(x):
    return 0

def sigmoid(x):
    a = 1/(1+(exp(-Lamda*x)))
    return a

def tanh(x):
    a = (exp(Lamda*x)-exp(-Lamda*x))/(exp(Lamda*x)+exp(-Lamda*x))
    return a

def Relu(x):
    if x<=0:
        return 0
    else:
        return x

def ReluDerivatie(x):
    if x<0:
        return 0
    else:
        return 1

LearningRules = int(input("Öğrenme Kuralı Numarasını Seçiniz\n1.Hebb\n2.Perceptron\n3.Delta\n4.Widrow-Hoff\n"))

ActivationFunction = int(input("Aktivasyon Fonksiyonu Numarasını Seçiniz\n1.Birim Basamak Fonksiyon\n2.Sigmoid Fonksiyon\n3.Hiperbolik Tanjant Fonksiyonu\n4.Relu Fonksiyonu\n"))

İteration=int(input("Iterasyon Sayısını Seçiniz"))

LearningConstant = float(input("Lütfen Öğrenme Sabitini Giriniz."))

Lamda = int(input("Lamda Değerini Giriniz.Eğer Lamda Değeri Yoksa Bir Sayı ile Geçiniz"))

max_input = int(input("Lütfen Kaç Adet X Giriş Değeri Olacağını Seçiniz (**En Fazla 6 Giriş**)"))

while True:
    if max_input >6:
        print("Maksimum 6 Giriş Yapmanız Gerekmektedir. Lütfen Doğru Girişi Yapınız.")
    elif max_input == 2:
        my_array = []
        a = int(input("1. Matris Boyutu Nedir"))
        for i in range(a):
            my_array.append(float(input("Değerler:")))
        x1 = numpy.array(my_array)
        my_array = []
        a = int(input("2. Matris Boyutu Nedir"))
        for i in range(a):
            my_array.append(float(input("Değerler:")))
        x2 = numpy.array(my_array)
        break
    elif max_input == 3:
        my_array = []
        a = int(input("1. Matris Boyutu Nedir"))
        for i in range(a):
            my_array.append(float(input("Değerler:")))
        x1 = numpy.array(my_array)
        my_array = []
        a = int(input("2. Matris Boyutu Nedir"))
        for i in range(a):
            my_array.append(float(input("Değerler:")))
        x2 = numpy.array(my_array)
        my_array = []
        a = int(input("3. Matris Boyutu Nedir"))
        for i in range(a):
            my_array.append(float(input("Değerler:")))
        x3 = numpy.array(my_array)
        break
  
    elif max_input == 4:
        my_array = []
        a = int(input("1. Matris Boyutu Nedir"))
        for i in range(a):
            my_array.append(float(input("Değerler:")))
        x1 = numpy.array(my_array)
        my_array = []
        a = int(input("2. Matris Boyutu Nedir"))
        for i in range(a):
            my_array.append(float(input("Değerler:")))
        x2 = numpy.array(my_array)
        my_array = []
        a = int(input("3. Matris Boyutu Nedir"))
        for i in range(a):
            my_array.append(float(input("Değerler:")))
        x3 = numpy.array(my_array)
        my_array = []
        a = int(input("4. Matris Boyutu Nedir"))
        for i in range(a):
            my_array.append(float(input("Değerler:")))
        x4 = numpy.array(my_array)
        break
 
    elif max_input == 5:
        my_array = []
        a = int(input("1. Matris Boyutu Nedir"))
        for i in range(a):
            my_array.append(float(input("Değerler:")))
        x1 = numpy.array(my_array)
        my_array = []
        a = int(input("2. Matris Boyutu Nedir"))
        for i in range(a):
            my_array.append(float(input("Değerler:")))
        x2 = numpy.array(my_array)
        my_array = []
        a = int(input("3. Matris Boyutu Nedir"))
        for i in range(a):
            my_array.append(float(input("Değerler:")))
        x3 = numpy.array(my_array)
        my_array = []
        a = int(input("4. Matris Boyutu Nedir"))
        for i in range(a):
            my_array.append(float(input("Değerler:")))
        x4 = numpy.array(my_array)
        my_array = []
        a = int(input("5. Matris Boyutu Nedir"))
        for i in range(a):
            my_array.append(float(input("Değerler:")))
        x5 = numpy.array(my_array)
        break
    
    elif max_input == 6:
        my_array = []
        a = int(input("1. Matris Boyutu Nedir"))
        for i in range(a):
            my_array.append(float(input("Değerler:")))
        x1 = numpy.array(my_array)
        my_array = []
        a = int(input("2. Matris Boyutu Nedir"))
        for i in range(a):
            my_array.append(float(input("Değerler:")))
        x2 = numpy.array(my_array)
        my_array = []
        a = int(input("3. Matris Boyutu Nedir"))
        for i in range(a):
            my_array.append(float(input("Değerler:")))
        x3 = numpy.array(my_array)
        my_array = []
        a = int(input("4. Matris Boyutu Nedir"))
        for i in range(a):
            my_array.append(float(input("Değerler:")))
        x4 = numpy.array(my_array)
        my_array = []
        a = int(input("5. Matris Boyutu Nedir"))
        for i in range(a):
            my_array.append(float(input("Değerler:")))
        x5 = numpy.array(my_array)
        my_array = []
        a = int(input("6. Matris Boyutu Nedir"))
        for i in range(a):
            my_array.append(float(input("Değerler:")))
        x6 = numpy.array(my_array)
        break
    else:
        print("Yanlış Giriş Yaptınız Lütfen Doğru Girişi Yapınız")

    















