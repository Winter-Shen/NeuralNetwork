import NeuralNetwork as nn
import numpy as np
'''
c = nn.InputLayer(3).input(np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])).forwardPropogation().output()

l1 = nn.Layer(3, 4, 0.1)
c = l1.initializeWeight(np.array([[7,8,9,10],[11,12,13,14],[15,16,17,18]])).input(c).forwardPropogation().output()

l2 = nn.Layer(4, 5, 0.1)
c = l2.initializeWeight(np.array([[19,20,21,22,23],[23,24,25,26,27],[28,29,30,31,32],[33,34,35,36,37]])).input(c).forwardPropogation().output()

l3 = nn.Layer(5, 1, 0.1)
c = l3.initializeWeight(np.array([[37],[38],[39],[40],[41]])).input(c).forwardPropogation().output()


r = c - np.array([[23], [24],[25],[26]])
dy = l3.setDy(r*2/l3.size).backwardPropogation(4).getDx()
dy = l2.setDy(dy).backwardPropogation(1).getDx()
dy = l1.setDy(dy).backwardPropogation(1).getDx()
#print(l1.getWeight())
#print(dy)

p1 = l1.input(np.array([[13,14,15]])).forwardPropogation().output()
p2 = l2.input(p1).forwardPropogation().output()
p3 = l3.input(p2).forwardPropogation().output()
'''

m = nn.model()
m.addLayer(nn.InputLayer(3))
m.addLayer(nn.Layer(3, 4, 0.1).initializeWeight(np.array([[7,8,9,10],[11,12,13,14],[15,16,17,18]])))
m.addLayer(nn.Layer(4, 5, 0.1).initializeWeight(np.array([[19,20,21,22,23],[23,24,25,26,27],[28,29,30,31,32],[33,34,35,36,37]])))
m.addLayer(nn.Layer(5, 1, 0.1).initializeWeight(np.array([[37],[38],[39],[40],[41]])))

X = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
Y = np.array([[23], [24],[25],[26]])
m.fit0(X, Y)
print(m.predict(np.array([[13,14,15]])))