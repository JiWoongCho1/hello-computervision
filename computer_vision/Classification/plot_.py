import matplotlib.pyplot as plt

def plot_(hist_list, epoch):
    plt.plot(range(1, epoch+1), hist_list['train'], label = 'train')
    plt.plot(range(1, epoch + 1), hist_list['val'], label='validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss') # can be metric
    plt.legend()
    plt.show()