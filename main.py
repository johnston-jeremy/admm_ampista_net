import os
import numpy as np
import tensorflow as tf
import gc
import time
from pdb import set_trace as bp
from ampistanet import AMPNet
import admmnet

def Phi_siso(M,N):
  m = np.arange(M)[:,None]
  n = np.arange(N)
  return np.exp(1j*2*np.pi*m*n/N)/np.sqrt(M)

class Problem():
  def __init__(self,N,L,M,Ng,K,Ntxrx,J):

    self.N = N
    self.M = np.prod(Ntxrx)
    self.L = L
    self.K = K
    self.Ng = Ng
    self.J = J

    # Pilot matrix
    A = CN(L, N, 1/L)

    # Array response matrix
    Phi = Phi_siso(M,Ng)

    w,v = np.linalg.eig( np.matmul( np.conj(Phi.T), Phi ) )
    alpha = 1/max(np.abs(w))  

    self.alpha = np.float32(alpha)
    self.alpha0 = np.float32(alpha)
    self.Phi = np.complex64(Phi)
    self.A = A

    self.M2 = np.matmul(np.conj(A.T), A)
    self.M3 = np.conj(A.T)

def CN(d1,d2,variance):
  return np.sqrt(variance/2) * (np.random.randn(d1,d2) + 1j*np.random.randn(d1,d2))

def scaled_noise_c(A,X,SNR):
  AX = np.matmul(A,X)
  norm_AX = np.linalg.norm(AX)
  L,M = AX.shape
  noise = CN(L,M,1);
  noise = 10**(-SNR/20) * norm_AX * noise/np.linalg.norm(noise)
  # SNR_emp = 20*np.log10(norm_AX/np.linalg.norm(noise))
  stdev = np.sqrt(np.mean(np.abs(noise - np.mean(noise))**2))
  return noise,stdev

def gen_c(p,Nsamp,channel_sparsity,N,L,M,Ng,K,SNR):
  Phi = p.Phi
  A = p.A

  Y = np.zeros((Nsamp,L,M),dtype=complex)
  X = np.zeros((Nsamp,N,M),dtype=complex)

  for j in range(Nsamp):
    s = np.zeros((Ng,K), dtype=complex)
    for i in range(K):
      s[np.random.permutation(Ng)[:channel_sparsity],i] \
        = np.random.normal(size=(channel_sparsity,)) \
        + 1j*np.random.normal(size=(channel_sparsity,))
    h = np.matmul(Phi, s).T
    ind = np.random.permutation(N)[:K]
    X[j,ind] = h
    

    Z, sigma = scaled_noise_c(A,X[j],SNR)

    Y[j] = np.matmul(A, X[j]) + Z

  return Y, X, sigma

def nmse(X,Xhat):
  X = X[0] + 1j*X[1]
  Xhat = Xhat[0].numpy() + 1j*Xhat[1].numpy()
  return 10*np.log10(np.mean(np.sum(np.abs(Xhat-X)**2,axis=(1,2))/np.sum(np.abs(X)**2,axis=(1,2))))

def train_loop(net, p, datatrain, datatest, lr_schedule, bs, savepath=None):
  
  Y,X = datatrain
  Ytest, Xtest = datatest
  Ntr = len(X[0])
  
  print('Initial test error:', nmse(Xtest, net(Ytest)), 'dB')
  print('Initial train error =', nmse(X, net(Y)), 'dB')
  net.compile(tf.keras.optimizers.Adam(learning_rate=lr_schedule[0]))
  # net.summary()
  for i, lr in enumerate(lr_schedule):
    gc.collect()
    net.optimizer.lr.assign(lr)
    print('Epoch', str(i+1), 'learning rate =', '{:.0e}'.format(lr))

    progbar = tf.keras.utils.Progbar(Ntr//bs)
    nb = 0
    for b in list(range(0,Ntr,bs)):
      v = net.train_step((Y[0][b:b+bs], Y[1][b:b+bs]), (X[0][b:b+bs],X[1][b:b+bs]))
      nb += 1
      if nb%100==0:
        error = nmse(Xtest, net(Ytest))
        progbar.update(nb, values=[('loss',v),('testNMSE',error)])
      else:
        progbar.update(nb, values=[('loss',v)])
    error = nmse(Xtest, net(Ytest))

    if savepath is not None:
      path = savepath \
           + 'Epoch=' + str(i+1) \
           + '_Rate=' + '{:.0e}'.format(lr) \
           + '_Batch=' + str(bs) \
           + '_Error=' + '{:.3f}'.format(error)
      save(net, p, path)
    
  return net

def get_filename(method, p, numlayers, snr):
  stamp = method \
        + '_{0}x{1}x{2}'.format(p.N,p.M,p.L) \
        + '_J=' + str(p.J) + '_K=' + str(p.K) \
        + '_SNR=' + str(snr) + 'dB_' \
        + str(numlayers) + 'layers_' \
        + time.strftime("%m_%d_%Y_%H_%M_%S",time.localtime())
  return stamp

def gen_data(p, Nsamp, Ntest, SNR):
  Yc,Xc,p.sigma_noise = gen_c(p, Nsamp, p.J, p.N, p.L, p.M, p.Ng , p.K, SNR)
  Y = [Yc.real, Yc.imag]
  X = [Xc.real, Xc.imag]
  
  Ytestc,Xtestc,_ = gen_c(p, Ntest, p.J, p.N, p.L, p.M, p.Ng , p.K, SNR)
  Ytest = [Ytestc.real, Ytestc.imag]
  Xtest = [Xtestc.real, Xtestc.imag]

  return Y,X,Ytest,Xtest

def save(net, p, filepath):
  net.save_weights(filepath+'/weights')
  np.save(filepath+'/A', p.A)
  np.save(filepath+'/Phi', p.Phi)

def load(net, weights_path):
  obj = net.load_weights(weights_path)
  obj.expect_partial()
  return net

def train(method):
  # learning rate schedule.
  schedule = {'ampista':10*[1e-3] + 20*[1e-4] + 10*[1e-5], 
              'admm':20*[1e-3] + 10*[1e-4] + 10*[1e-5]}

  batch_size = 100

  # number of stages
  n = 5

  p = Problem(N=50,L=12,M=8,Ng=2*8,K=3,Ntxrx=(8,1),J=2)
  SNR = 10
  Nsamp = 8*10**4 # train set size
  Ntest = 10**3 # test set size
  Y, X, Ytest, Xtest = gen_data(p, Nsamp, Ntest, SNR)
  print('N =', p.N, ', L =', p.L, ', M =', p.M, ', K =', p.K, ', J =', p.J)

  savepath = './save/' + get_filename(method, p, n, SNR) +'/'
  os.mkdir(savepath)

  net = gen_net(method, p, n)
  net = train_loop(net, p, (Y,X), (Ytest,Xtest), schedule[method], batch_size,savepath=savepath)

  print('Train error =', nmse(X, net(Y)), 'dB')
  print('Test error =', nmse(Xtest, net(Ytest)), 'dB')

  return savepath

def test(path, method):

  p = Problem(N=50,L=12,M=8,Ng=2*8,K=3,Ntxrx=(8,1),J=2)
  
  # load network weights
  net_path = path + os.listdir(path)[-1] # path to final epoch
  net = gen_net(method, p, n=5)
  net = load(net, net_path + '/weights')

  # generate test data
  Ntest = 2**10
  SNR = 10
  p.A = np.load(net_path + '/A.npy') # load pilot matrix
  _, _, Ytest, Xtest = gen_data(p, 1, Ntest, SNR)

  print('Test error =', nmse(Xtest, net(Ytest)), 'dB')

def gen_net(method, p, n):
  if method == 'admm':
    # initialize
    p.sigma, p.mu, p.rho, p.taux, p.tauz = 6.9e-01, 1.0e-02, 2.0e+00, 1.4e-01, 7.2e-01

    tf.keras.backend.set_floatx('float32')
    return admmnet.ADMMNet(p, n)
  elif method == 'ampista':
    # initialize
    p.lam = .1
    p.damp_init = 0.6

    tf.keras.backend.set_floatx('float32')
    num_stages = {'amp':n, 'ista':5}
    return AMPNet(p, num_stages)

if __name__ == '__main__':
  # Example
  test(path='./save/admm3_50x8x12_J=2_K=3_SNR=10dB_5layers_03_25_2021_03_29_34/', method='admm')
  test(path='./save/ampista_50x8x12_J=2_K=3_SNR=10dB_5layers_03_25_2021_03_14_09/', method='ampista')
  
  # train ADMM-Net
  path = train(method ='admm') 
  # test
  test(path=path, method='admm')

  # train VAMP-ISTA-Net
  path = train(method = 'ampista') 
  # test
  test(path=path, method='ampista')