import numpy as np
import NeuralNetworkWork as nnw

class Delta():
    my_array = []
    a = int(input("D matris Boyutu Nedir"))
    for i in range(a):
        my_array.append(float(input("Değerler:")))
    my_array = np.array(my_array)

    d = []
    for i in range((nnw.İteration+1)):
        for j in  my_array:
         d.append(j)

    my_array = []
    a = int(input("Ağırlık Matris Boyutu Nedir"))
    for i in range(a):
        my_array.append(float(input("Değerler:")))
    agirlik = np.array(my_array).T

    if nnw.max_input == 2:
        for i in range(nnw.İteration):
            if nnw.ActivationFunction == 1:
                net = np.dot(nnw.x1,agirlik)
                tahmin = nnw.UnitStep(net)    
                agirlik += (nnw.LearningConstant*(d[i]-tahmin)*nnw.UnitStepDerivative(tahmin)*nnw.x1)
                net2 = np.dot(nnw.x2,agirlik)
                tahmin =  nnw.UnitStep(net2)
                agirlik += (nnw.LearningConstant*(d[i+1]-tahmin)*nnw.UnitStepDerivative(tahmin)*nnw.x2)
                print(agirlik)
            elif nnw.ActivationFunction == 2 :
                net = np.dot(nnw.x1,agirlik)
                tahmin = nnw.sigmoid(net) 
                agirlik += (nnw.LearningConstant*(d[i]-tahmin)*(tahmin*(1-tahmin))*(nnw.x1))
                net2 = np.dot(nnw.x2,agirlik)
                tahmin =  nnw.sigmoid(net2)
                agirlik += (nnw.LearningConstant*(d[i+1]-tahmin)*(tahmin*(1-tahmin))*(nnw.x2))
                print(agirlik)
            elif nnw.ActivationFunction  == 3 :
                net = np.dot(nnw.x1,agirlik)
                tahmin = nnw.tanh(net)   
                agirlik += (nnw.LearningConstant*(d[i]-tahmin)*((1-tahmin**2))*(nnw.x1))
                net2 = np.dot(nnw.x2,agirlik)
                tahmin =  nnw.tanh(net2)
                agirlik += (nnw.LearningConstant*(d[i+1]-tahmin)*((1-tahmin**2))*nnw.x2)
                print(agirlik)
            elif nnw.ActivationFunction == 4 :
                net = np.dot(nnw.x1,agirlik)
                tahmin = nnw.Relu(net) 
                agirlik += (nnw.LearningConstant*(d[i]-tahmin)*nnw.ReluDerivatie(tahmin)*nnw.x1)
                net2 = np.dot(nnw.x2,agirlik)
                tahmin =  nnw.Relu(net2)
                agirlik += (nnw.LearningConstant*(d[i+1]-tahmin)*nnw.ReluDerivatie(tahmin)*nnw.x2)
                print(agirlik)
            else:
                print("Yanlış Data Değeri")

    elif nnw.max_input == 3:
        for i in range(nnw.İteration):
            if nnw.ActivationFunction == 1:
                net = np.dot(nnw.x1,agirlik)
                tahmin = nnw.UnitStep(net)   
                agirlik += (nnw.LearningConstant*(d[i]-tahmin)*nnw.UnitStepDerivative(tahmin)*nnw.x1)
                net2 = np.dot(nnw.x2,agirlik)
                tahmin =  nnw.UnitStep(net2)
                agirlik += (nnw.LearningConstant*(d[i+1]-tahmin)*nnw.UnitStepDerivative(tahmin)*nnw.x2)
                net3 = np.dot(nnw.x3,agirlik)
                tahmin = nnw.UnitStep(net3)
                agirlik += (nnw.LearningConstant*(d[i+2]-tahmin)*nnw.UnitStepDerivative(tahmin)*nnw.x3)
                print(agirlik)
            elif nnw.ActivationFunction == 2 :
                net = np.dot(nnw.x1,agirlik)
                tahmin = nnw.sigmoid(net) 
                agirlik += (nnw.LearningConstant*(d[i]-tahmin)*(tahmin*(1-tahmin))*(nnw.x1))
                net2 = np.dot(nnw.x2,agirlik)
                tahmin =  nnw.sigmoid(net2)
                agirlik += (nnw.LearningConstant*(d[i+1]-tahmin)*(tahmin*(1-tahmin))*(nnw.x2))
                net3 = np.dot(nnw.x3,agirlik)
                tahmin = nnw.sigmoid(net3)
                agirlik += (nnw.LearningConstant*(d[i+2]-tahmin)*(tahmin*(1-tahmin))*nnw.x3)
                print(agirlik)
            elif nnw.ActivationFunction  == 3 :
                net = np.dot(nnw.x1,agirlik)
                tahmin = nnw.tanh(net)    
                agirlik += (nnw.LearningConstant*(d[i]-tahmin)*((1-tahmin**2))*(nnw.x1))
                net2 = np.dot(nnw.x2,agirlik)
                tahmin =  nnw.tanh(net2)
                agirlik += (nnw.LearningConstant*(d[i+1]-tahmin)*((1-tahmin**2))*nnw.x2)
                net3 = np.dot(nnw.x3,agirlik)
                tahmin = nnw.tanh(net3)
                agirlik += (nnw.LearningConstant*(d[i+2]-tahmin)*((1-tahmin**2))*nnw.x3)
                print(agirlik)
            elif nnw.ActivationFunction == 4 :
                net = np.dot(nnw.x1,agirlik)
                tahmin = nnw.Relu(net)
                agirlik += (nnw.LearningConstant*(d[i]-tahmin)*nnw.ReluDerivatie(tahmin)*nnw.x1)
                net2 = np.dot(nnw.x2,agirlik)
                tahmin =  nnw.Relu(net2)
                agirlik += (nnw.LearningConstant*(d[i+1]-tahmin)*nnw.ReluDerivatie(tahmin)*nnw.x2)
                net3 = np.dot(nnw.x3,agirlik)
                tahmin = nnw.Relu(net3)
                agirlik += (nnw.LearningConstant*(d[i+2]-tahmin)*nnw.ReluDerivatie(tahmin)*nnw.x3)
                print(agirlik)
            else:
                print("Yanlış Data Değeri")
        
    elif nnw.max_input == 4:
        for i in range(nnw.İteration):
            if nnw.ActivationFunction == 1:
                net = np.dot(nnw.x1,agirlik)
                tahmin = nnw.UnitStep(net)    
                agirlik += (nnw.LearningConstant*(d[i]-tahmin)*nnw.UnitStepDerivative(tahmin)*nnw.x1)
                net2 = np.dot(nnw.x2,agirlik)
                tahmin =  nnw.UnitStep(net2)
                agirlik += (nnw.LearningConstant*(d[i+1]-tahmin)*nnw.UnitStepDerivative(tahmin)*nnw.x2)
                net3 = np.dot(nnw.x3,agirlik)
                tahmin = nnw.UnitStep(net3)
                agirlik += (nnw.LearningConstant*(d[i+2]-tahmin)*nnw.UnitStepDerivative(tahmin)*nnw.x3)
                net4 = np.dot(nnw.x4,agirlik)
                tahmin = nnw.UnitStep(net4)
                agirlik += (nnw.LearningConstant*(d[i+3]-tahmin)*nnw.UnitStepDerivative(tahmin)*nnw.x4)
                print("Agirlik",agirlik)
            elif nnw.ActivationFunction == 2 :
                net = np.dot(nnw.x1,agirlik)
                tahmin = nnw.sigmoid(net)   
                agirlik += (nnw.LearningConstant*(d[i]-tahmin)*(tahmin*(1-tahmin))*(nnw.x1))
                net2 = np.dot(nnw.x2,agirlik)
                tahmin =  nnw.sigmoid(net2)
                agirlik += (nnw.LearningConstant*(d[i+1]-tahmin)*(tahmin*(1-tahmin))*(nnw.x2))
                net3 = np.dot(nnw.x3,agirlik)
                tahmin = nnw.sigmoid(net3)
                agirlik += (nnw.LearningConstant*(d[i+2]-tahmin)*(tahmin*(1-tahmin))*nnw.x3)
                net4 = np.dot(nnw.x4,agirlik)
                tahmin = nnw.sigmoid(net4)
                agirlik += (nnw.LearningConstant*(d[i+3]-tahmin)*(tahmin*(1-tahmin))*nnw.x4)
                print("Agirlik",agirlik)
            elif nnw.ActivationFunction  == 3 :
                net = np.dot(nnw.x1,agirlik)
                tahmin = nnw.tanh(net)   
                agirlik += (nnw.LearningConstant*(d[i]-tahmin)*((1-tahmin**2))*(nnw.x1))
                net2 = np.dot(nnw.x2,agirlik)
                tahmin =  nnw.tanh(net2)
                agirlik += (nnw.LearningConstant*(d[i+1]-tahmin)*((1-tahmin**2))*nnw.x2)
                net3 = np.dot(nnw.x3,agirlik)
                tahmin = nnw.tanh(net3)
                agirlik += (nnw.LearningConstant*(d[i+2]-tahmin)*((1-tahmin**2))*nnw.x3)
                net4 = np.dot(nnw.x4,agirlik)
                tahmin = nnw.tanh(net4)
                agirlik += (nnw.LearningConstant*(d[i+3]-tahmin)*((1-tahmin**2))*nnw.x4)
                print("Agirlik",agirlik)
            elif nnw.ActivationFunction == 4 :
                net = np.dot(nnw.x1,agirlik)
                tahmin = nnw.Relu(net)
                agirlik += (nnw.LearningConstant*(d[i]-tahmin)*nnw.ReluDerivatie(tahmin)*nnw.x1)
                net2 = np.dot(nnw.x2,agirlik)
                tahmin =  nnw.Relu(net2)
                agirlik += (nnw.LearningConstant*(d[i+1]-tahmin)*nnw.ReluDerivatie(tahmin)*nnw.x2)
                net3 = np.dot(nnw.x3,agirlik)
                tahmin = nnw.Relu(net3)
                agirlik += (nnw.LearningConstant*(d[i+2]-tahmin)*nnw.ReluDerivatie(tahmin)*nnw.x3)
                net4 = np.dot(nnw.x4,agirlik)
                tahmin = nnw.Relu(net4)
                agirlik += (nnw.LearningConstant*(d[i+3]-tahmin)*nnw.ReluDerivatie(tahmin)*nnw.x4)
                print("Agirlik",agirlik)
            else:
                print("Yanlış Data Değeri")

    elif nnw.max_input == 5:
        for i in range(nnw.İteration):
            if nnw.ActivationFunction == 1:
                net = np.dot(nnw.x1,agirlik)
                tahmin = nnw.UnitStep(net)    
                agirlik += (nnw.LearningConstant*(d[i]-tahmin)*nnw.UnitStepDerivative(tahmin)*nnw.x1)
                net2 = np.dot(nnw.x2,agirlik)
                tahmin =  nnw.UnitStep(net2)
                agirlik += (nnw.LearningConstant*(d[i+1]-tahmin)*nnw.UnitStepDerivative(tahmin)*nnw.x2)
                net3 = np.dot(nnw.x3,agirlik)
                tahmin = nnw.UnitStep(net3)
                agirlik += (nnw.LearningConstant*(d[i+2]-tahmin)*nnw.UnitStepDerivative(tahmin)*nnw.x3)
                net4 = np.dot(nnw.x4,agirlik)
                tahmin = nnw.UnitStep(net4)
                agirlik += (nnw.LearningConstant*(d[i+3]-tahmin)*nnw.UnitStepDerivative(tahmin)*nnw.x4)
                net5 = np.dot(nnw.x5,agirlik)
                tahmin = nnw.UnitStep(net5)
                agirlik += (nnw.LearningConstant*(d[i+4]-tahmin)*nnw.UnitStepDerivative(tahmin)*nnw.x5)
                print("Agirlik",agirlik)
            elif nnw.ActivationFunction == 2 :
                net = np.dot(nnw.x1,agirlik)
                tahmin = nnw.sigmoid(net)   
                agirlik += (nnw.LearningConstant*(d[i]-tahmin)*(tahmin*(1-tahmin))*(nnw.x1))
                net2 = np.dot(nnw.x2,agirlik)
                tahmin =  nnw.sigmoid(net2)
                agirlik += (nnw.LearningConstant*(d[i+1]-tahmin)*(tahmin*(1-tahmin))*(nnw.x2))
                net3 = np.dot(nnw.x3,agirlik)
                tahmin = nnw.sigmoid(net3)
                agirlik += (nnw.LearningConstant*(d[i+2]-tahmin)*(tahmin*(1-tahmin))*nnw.x3)
                net4 = np.dot(nnw.x4,agirlik)
                tahmin = nnw.sigmoid(net4)
                agirlik += (nnw.LearningConstant*(d[i+3]-tahmin)*(tahmin*(1-tahmin))*nnw.x4)
                net5 = np.dot(nnw.x5,agirlik)
                tahmin = nnw.sigmoid(net5)
                agirlik += (nnw.LearningConstant*(d[i+4]-tahmin)*(tahmin*(1-tahmin))*nnw.x5)
                print("Agirlik",agirlik)
            elif nnw.ActivationFunction  == 3 :
                net = np.dot(nnw.x1,agirlik)
                tahmin = nnw.tanh(net)   
                agirlik += (nnw.LearningConstant*(d[i]-tahmin)*((1-tahmin**2))*(nnw.x1))
                net2 = np.dot(nnw.x2,agirlik)
                tahmin =  nnw.tanh(net2)
                agirlik += (nnw.LearningConstant*(d[i+1]-tahmin)*((1-tahmin**2))*nnw.x2)
                net3 = np.dot(nnw.x3,agirlik)
                tahmin = nnw.tanh(net3)
                agirlik += (nnw.LearningConstant*(d[i+2]-tahmin)*((1-tahmin**2))*nnw.x3)
                net4 = np.dot(nnw.x4,agirlik)
                tahmin = nnw.tanh(net4)
                agirlik += (nnw.LearningConstant*(d[i+3]-tahmin)*((1-tahmin**2))*nnw.x4)
                net5 = np.dot(nnw.x5,agirlik)
                tahmin = nnw.tanh(net5)
                agirlik += (nnw.LearningConstant*(d[i+4]-tahmin)*((1-tahmin**2))*nnw.x5)
                print("Agirlik",agirlik)
            elif nnw.ActivationFunction == 4 :
                net = np.dot(nnw.x1,agirlik)
                tahmin = nnw.Relu(net)
                agirlik += (nnw.LearningConstant*(d[i]-tahmin)*nnw.ReluDerivatie(tahmin)*nnw.x1)
                net2 = np.dot(nnw.x2,agirlik)
                tahmin =  nnw.Relu(net2)
                agirlik += (nnw.LearningConstant*(d[i+1]-tahmin)*nnw.ReluDerivatie(tahmin)*nnw.x2)
                net3 = np.dot(nnw.x3,agirlik)
                tahmin = nnw.Relu(net3)
                agirlik += (nnw.LearningConstant*(d[i+2]-tahmin)*nnw.ReluDerivatie(tahmin)*nnw.x3)
                net4 = np.dot(nnw.x4,agirlik)
                tahmin = nnw.Relu(net4)
                agirlik += (nnw.LearningConstant*(d[i+3]-tahmin)*nnw.ReluDerivatie(tahmin)*nnw.x4)
                net5 = np.dot(nnw.x5,agirlik)
                tahmin = nnw.Relu(net5)
                agirlik += (nnw.LearningConstant*(d[i+4]-tahmin)*nnw.ReluDerivatie(tahmin)*nnw.x5)
                print("Agirlik",agirlik)
            else:
                print("Yanlış Data Değeri")

    elif nnw.max_input == 6:
        for i in range(nnw.İteration):
            if nnw.ActivationFunction == 1:
                net = np.dot(nnw.x1,agirlik)
                tahmin = nnw.UnitStep(net)    
                agirlik += (nnw.LearningConstant*(d[i]-tahmin)*nnw.UnitStepDerivative(tahmin)*nnw.x1)
                net2 = np.dot(nnw.x2,agirlik)
                tahmin =  nnw.UnitStep(net2)
                agirlik += (nnw.LearningConstant*(d[i+1]-tahmin)*nnw.UnitStepDerivative(tahmin)*nnw.x2)
                net3 = np.dot(nnw.x3,agirlik)
                tahmin = nnw.UnitStep(net3)
                agirlik += (nnw.LearningConstant*(d[i+2]-tahmin)*nnw.UnitStepDerivative(tahmin)*nnw.x3)
                net4 = np.dot(nnw.x4,agirlik)
                tahmin = nnw.UnitStep(net4)
                agirlik += (nnw.LearningConstant*(d[i+3]-tahmin)*nnw.UnitStepDerivative(tahmin)*nnw.x4)
                net5 = np.dot(nnw.x5,agirlik)
                tahmin = nnw.UnitStep(net5)
                agirlik += (nnw.LearningConstant*(d[i+4]-tahmin)*nnw.UnitStepDerivative(tahmin)*nnw.x5)
                net6 = np.dot(nnw.x6,agirlik)
                tahmin = nnw.UnitStep(net5)
                agirlik += (nnw.LearningConstant*(d[i+5]-tahmin)*nnw.UnitStepDerivative(tahmin)*nnw.x6)
                print("Agirlik",agirlik)
            elif nnw.ActivationFunction == 2 :
                net = np.dot(nnw.x1,agirlik)
                tahmin = nnw.sigmoid(net)   
                agirlik += (nnw.LearningConstant*(d[i]-tahmin)*(tahmin*(1-tahmin))*(nnw.x1))
                net2 = np.dot(nnw.x2,agirlik)
                tahmin =  nnw.sigmoid(net2)
                agirlik += (nnw.LearningConstant*(d[i+1]-tahmin)*(tahmin*(1-tahmin))*(nnw.x2))
                net3 = np.dot(nnw.x3,agirlik)
                tahmin = nnw.sigmoid(net3)
                agirlik += (nnw.LearningConstant*(d[i+2]-tahmin)*(tahmin*(1-tahmin))*nnw.x3)
                net4 = np.dot(nnw.x4,agirlik)
                tahmin = nnw.sigmoid(net4)
                agirlik += (nnw.LearningConstant*(d[i+3]-tahmin)*(tahmin*(1-tahmin))*nnw.x4)
                net5 = np.dot(nnw.x5,agirlik)
                tahmin = nnw.sigmoid(net5)
                agirlik += (nnw.LearningConstant*(d[i+4]-tahmin)*(tahmin*(1-tahmin))*nnw.x5)
                net6 = np.dot(nnw.x6,agirlik)
                tahmin = nnw.sigmoid(net5)
                agirlik += (nnw.LearningConstant*(d[i+5]-tahmin)*(tahmin*(1-tahmin))*nnw.x6)
                print("Agirlik",agirlik)
            elif nnw.ActivationFunction  == 3 :
                net = np.dot(nnw.x1,agirlik)
                tahmin = nnw.tanh(net)   
                agirlik += (nnw.LearningConstant*(d[i]-tahmin)*((1-tahmin**2))*(nnw.x1))
                net2 = np.dot(nnw.x2,agirlik)
                tahmin =  nnw.tanh(net2)
                agirlik += (nnw.LearningConstant*(d[i+1]-tahmin)*((1-tahmin**2))*nnw.x2)
                net3 = np.dot(nnw.x3,agirlik)
                tahmin = nnw.tanh(net3)
                agirlik += (nnw.LearningConstant*(d[i+2]-tahmin)*((1-tahmin**2))*nnw.x3)
                net4 = np.dot(nnw.x4,agirlik)
                tahmin = nnw.tanh(net4)
                agirlik += (nnw.LearningConstant*(d[i+3]-tahmin)*((1-tahmin**2))*nnw.x4)
                net5 = np.dot(nnw.x5,agirlik)
                tahmin = nnw.tanh(net5)
                agirlik += (nnw.LearningConstant*(d[i+4]-tahmin)*((1-tahmin**2))*nnw.x5)
                net6 = np.dot(nnw.x6,agirlik)
                tahmin = nnw.tanh(net5)
                agirlik += (nnw.LearningConstant*(d[i+5]-tahmin)*((1-tahmin**2))*nnw.x6)
                print("Agirlik",agirlik)
            elif nnw.ActivationFunction == 4 :
                net = np.dot(nnw.x1,agirlik)
                tahmin = nnw.Relu(net)
                agirlik += (nnw.LearningConstant*(d[i]-tahmin)*nnw.ReluDerivatie(tahmin)*nnw.x1)
                net2 = np.dot(nnw.x2,agirlik)
                tahmin =  nnw.Relu(net2)
                agirlik += (nnw.LearningConstant*(d[i+1]-tahmin)*nnw.ReluDerivatie(tahmin)*nnw.x2)
                net3 = np.dot(nnw.x3,agirlik)
                tahmin = nnw.Relu(net3)
                agirlik += (nnw.LearningConstant*(d[i+2]-tahmin)*nnw.ReluDerivatie(tahmin)*nnw.x3)
                net4 = np.dot(nnw.x4,agirlik)
                tahmin = nnw.Relu(net4)
                agirlik += (nnw.LearningConstant*(d[i+3]-tahmin)*nnw.ReluDerivatie(tahmin)*nnw.x4)
                net5 = np.dot(nnw.x5,agirlik)
                tahmin = nnw.Relu(net5)
                agirlik += (nnw.LearningConstant*(d[i+4]-tahmin)*nnw.ReluDerivatie(tahmin)*nnw.x5)
                net6 = np.dot(nnw.x6,agirlik)
                tahmin = nnw.Relu(net5)
                agirlik += (nnw.LearningConstant*(d[i+5]-tahmin)*nnw.ReluDerivatie(tahmin)*nnw.x6)
                print("Agirlik",agirlik)
            else:
                print("Yanlış Data Değeri")