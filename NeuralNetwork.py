from numpy import exp,array,random,dot,mean,abs 

import numpy

giris = array([[0,0,1], [1,1,1] ,[1,0,1]])

gerceksonuc=array([[0,1,1]]).T

agirlik=array([1,1,1]).T

for x in range(100000):
    tahmin=1/(1+exp(-(dot(giris,agirlik))))
    agirlik = agirlik + dot(giris.T,((gerceksonuc-tahmin)*tahmin*(1-tahmin)))
    #print(str(numpy.mean(numpy.abs(gerceksonuc-tahmin))))


print(numpy.mean(1/(1+exp(-dot(array([1,0,0]),agirlik)))))







