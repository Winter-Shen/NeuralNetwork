import numpy as np

def sigmoid(x):
    return(np.exp(x)/(1+np.exp(x)))

def forward0(x, w, b):
    v=w*x+b
    return(sigmoid(v))

def forward(x, l):
    w_list = []
    b_list = []
    y_list = []

    x = np.matrix(x).T
    y_list.append(x)

    for i, j in enumerate(l):
        if i:
            w = np.ones((j, l[i-1]))
        else:
            w = np.ones((j, x.shape[0]))
        w_list.append(np.matrix(w))

        b = np.ones(j)
        b_list.append(np.matrix(b))

        x = forward0(x, np.matrix(w), np.matrix(b).T)
        y_list.append(x)

    #y = x[0, 0]
    return([y_list, w_list, b_list])

def backwards0(gradientYL, x, y_hat, w):
    jacobianVY = np.multiply(y_hat, (1-y_hat))
    gradientVL = np.multiply(jacobianVY,gradientYL)
    gradientUL = gradientVL
    gradientXL = w.T * gradientUL
    gradientWL = gradientUL*x.T
    gradientBL = gradientVL
    return([gradientXL, gradientWL, gradientBL])

def backwards1(y, net, l, eta):
    w_list=[]
    gradientYL = 2*(y - net[0][len(l)])
    for ll in range(len(l)-1,-1,-1):
        x0 = net[0][ll]
        y_hat = net[0][ll+1]
        w=net[1][ll]
        L = nn.backwards0(gradientYL, x0, y_hat, w)
        gradientYL = L[0]
        w = w - eta*L[1]
        w_list.append(w)
    return(w_list)

