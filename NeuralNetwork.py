import numpy as np

def sigmoid(x):
    return(np.exp(x)/(1+np.exp(x)))

def forward0(x, w, b):
    v=w*x+b
    return(sigmoid(v))

def generatePar(l,x):
    w_list = []
    b_list = []
    for i, j in enumerate(l):
        if i:
            w = np.ones((j, l[i-1]))
        else:
            w = np.ones((j, x.shape[0]))
        w_list.append(np.matrix(w))

        b = np.ones(j)
        b_list.append(np.matrix(b).T)
    return([w_list, b_list])

def forward(x, l, w_list, b_list): 
    y_list = []

    x = np.matrix(x).T
    y_list.append(x)

    for i, j in enumerate(l):
        x = forward0(x, w_list[i], b_list[i])
        y_list.append(x)

    return(y_list)

def backwards0(gradientYL, x, y_hat, w):
    jacobianVY = np.multiply(y_hat, (1-y_hat))
    gradientVL = np.multiply(jacobianVY,gradientYL)
    gradientUL = gradientVL
    gradientXL = w.T * gradientUL
    gradientWL = gradientUL*x.T
    gradientBL = gradientVL
    return([gradientXL, gradientWL, gradientBL])

def backwards1(y, value, l, eta, w_list, b_List):
    updatedW_list=[]
    updatedB_list=[]
    gradientYL = 2*(y - value[len(l)])
    for ll in range(len(l)-1,-1,-1):
        x0 = value[ll]
        y_hat = value[ll+1]
        w = w_list[ll]
        b = b_List[ll]
        
        L = backwards0(gradientYL, x0, y_hat, w)

        gradientYL = L[0]
        w = w - eta*L[1]
        updatedW_list.append(w)
        b = b - eta*L[2]
        updatedB_list.append(b)
    return([updatedW_list, updatedB_list])

def trainNet(x, y, l):
    [w_list, b_list] = generatePar(l, x)
    for i in range(10):
        values = forward(x, l, w_list, b_list)
        [w_list, b_list] = backwards1(y, values, l, 0.5, w_list, b_list)
        w_list = list(reversed(w_list))
        b_list = list(reversed(b_list))
    return([w_list, b_list])

