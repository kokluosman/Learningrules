import numpy as np
import NeuralNetworkWork as nnw
import Delta as dl
import WidrowHoff as wf 
import Hebb as hb
import Perceptron as pc


if nnw.LearningRules == 1:
    hb.Hebb()
elif nnw.LearningRules == 2:
    pc.Perceptron()
elif nnw.LearningRules == 3:
    dl.Delta()
elif nnw.LearningRules == 4:
    wf.WidrowHoff()
else:
    print("Yanlış Seçim")

    


