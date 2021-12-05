import scipy.io
import numpy as np
import glob
import dislib as ds
from dislib.classification import CascadeSVM
import time
from pycompss.api.api import compss_barrier
from pycompss.api.api import compss_wait_on
from scipy import signal
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def zero_pad(data, length):
    extended = np.zeros(length)
    signal_length = np.min([length, data.shape[0]])
    extended[:signal_length] = data[:signal_length]
    return extended

def spectrogram(data, fs=300, nperseg=64, noverlap=32):
    f, t, Sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    Sxx = np.transpose(Sxx, [0, 2, 1])
    Sxx = np.abs(Sxx)
    mask = Sxx > 0
    Sxx[mask] = np.log(Sxx[mask])
    return f, t, Sxx

def load_n_preprocess(dataDir):

    dataDir = 'training2017/'

    max_length = 61
    freq = 300

    ## Loading labels and select A and N classes 
    import csv
    csvfile = list(csv.reader(open(dataDir+'REFERENCE.csv')))
    csvfile = [x for x in csvfile if x[1] == 'A' or x[1] == 'N']

    ## Loading time serie signals
    files = [dataDir+i[0]+".mat" for i in csvfile]
    trainset = np.zeros((len(files),18810))
    count = 0
    for f in files:
        mat_val = zero_pad(scipy.io.loadmat(f)['val'][0], length=max_length * freq)
        sx = spectrogram(np.expand_dims(mat_val, axis=0))[2] # generate spectrogram
        sx_norm = (sx - np.mean(sx)) / np.std(sx) # normalize the spectrogram
        trainset[count,] = sx_norm.flatten()
        count += 1
   
    traintarget = np.zeros((trainset.shape[0],1))
    classes = ['A','N']
    for row in range(len(csvfile)):
        traintarget[row, 0] = 0 if classes.index(csvfile[row][1]) == 0 else 1

    trainset = np.concatenate((trainset, trainset), axis=0)
    traintarget = np.concatenate((traintarget, traintarget), axis=0)

    return(trainset,traintarget)

if __name__ == "__main__":

    X_train, y_train = load_n_preprocess('training2017/')
    X_test, y_test = load_n_preprocess('validation2017/')

    x = ds.array(X_train, block_size=(500, 500))
    y = ds.array(y_train, block_size=(500, 1))

    csvm = CascadeSVM(kernel='rbf', c=1, gamma='auto', tol=1e-2, random_state=seed)
    csvm.fit(x, y)

    compss_barrier()
    fit_time = time.time()
    print("Fit time", fit_time - load_time)
    out = [csvm.iterations, csvm.converged, load_time, fit_time]
    
    x = ds.array(X_test, block_size=(500, 500))
    y = ds.array(y_test, block_size=(500, 1))

    out.append(compss_wait_on(csvm.score(x, y)))
    print("Test time", time.time() - fit_time)
    print("Score: ", out)

    labels_pred = csvm.predict(X_test)
    cm = confusion_matrix(y_test, labels_pred)
    print(cm)
    acc= accuracy_score(y_test,labels_pred)
    print(acc)
    print (classification_report(y_test, labels_pred))
