import torch
import numpy as np

class Score_knn:
    acc = 0
    acc_real = 0
    acc_fake = 0
    precision = 0
    recall = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0


def knn(Mxx, Mxy, Myy, k):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1)))
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat(
        (Mxy.transpose(0, 1), Myy), 1)), 0)

    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1))
                ).topk(k, 0, False)
    
#     print(idx.size())
    count = torch.zeros(n0 + n1)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1)).float()

    s = Score_knn()
    s.tp = (pred * label).sum()
    s.fp = (pred * (1 - label)).sum()
    s.fn = ((1 - pred) * label).sum()
    s.tn = ((1 - pred) * (1 - label)).sum()
    s.precision = s.tp / (s.tp + s.fp + 1e-10)
    s.recall = s.tp / (s.tp + s.fn + 1e-10)
    s.acc_real = s.tp / (s.tp + s.fn)
    s.acc_fake = s.tn / (s.tn + s.fp)
    s.acc = torch.eq(label, pred).float().mean()
    s.k = k

    return s

#欧式距离
def distance(X, Y, sqrt):
    nX = X.size(0)
    nY = Y.size(0)
    X = X.view(nX,-1)
    X2 = (X*X).sum(1).resize_(nX,1)
    Y = Y.view(nY,-1)
    Y2 = (Y*Y).sum(1).resize_(nY,1)

    M = torch.zeros(nX, nY)
    M.copy_(X2.expand(nX, nY) + Y2.expand(nY, nX).transpose(0, 1) -
            2 * torch.mm(X, Y.transpose(0, 1)))

    del X, X2, Y, Y2

    if sqrt:
        M = ((M + M.abs()) / 2).sqrt()

    return M

def compute_score(real, fake):
#     if isinstance(real, torch.Tensor) == False:
#         real = torch.from_numpy(real)
#     if isinstance(fake, torch.Tensor) == False:
#         fake = torch.from_numpy(fake)
    
    Mxx = distance(real, real, False)
    Mxy = distance(real, fake, False)
    Myy = distance(fake, fake, False)
    
    s = knn(Mxx, Mxy, Myy, 1)

    return 2*s.acc_real-1,2*s.acc_fake-1

def nn1_acc(real, fake):
    '''
    生成数据要比原始数据多，acc_real、acc_fake越接近0说明两者分布越接近，
    acc_real负数，acc_fake正数说明model collapse
    acc_real正数，acc_fake正数说明生成效果不好，
    
    '''
    if isinstance(real, torch.Tensor) == False:
        real = torch.from_numpy(real)
    if isinstance(fake, torch.Tensor) == False:
        fake = torch.from_numpy(fake)
    
    acc_real = 0
    acc_fake = 0
    
    nf = fake.size(0)
    nr = real.size(0)
    
    if nf < nr:
        print("生成的样本不能比真实样本少")
        return

    np.random.shuffle(fake)
    for i in range(nf//nr-1):
        s1, s2 = compute_score(real, fake[nr*i:nr*(i+1),:])
        acc_real = acc_real + s1
        acc_fake = acc_fake + s2

    index = np.arange(0, nf)
    np.random.shuffle(index)
    index = index[:nr]
    s1, s2 = compute_score(real, fake[index, :])
    acc_real = acc_real + s1
    acc_fake = acc_fake + s2
    acc_real = acc_real / (1+nf//nr)
    acc_fake = acc_fake / (1+nf//nr)

    return acc_real, acc_fake


r = np.load('C:/Users/25100/Stars/M-GAN/m5.npy')#
g = np.load('C:/Users/25100/Stars/M-GAN/m595.npy')

print(r.shape, g.shape)
print(nn1_acc(r, g), nn1_acc(g[:750], g[750:]), nn1_acc(r[:64], r[64:]))

# (128, 3520) (1500, 3520)
# (tensor(0.3750), tensor(0.7148)) (tensor(0.2187), tensor(0.2453)) (tensor(0.2188), tensor(0.2656))
