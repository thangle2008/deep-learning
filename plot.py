import matplotlib.pyplot as plt
import argparse
import csv


parser = argparse.ArgumentParser()

parser.add_argument('file', action='store')


if __name__ == '__main__':

    args = parser.parse_args()

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    epochs = []

    with open(args.file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        # skip the header row
        next(reader)

        for row in reader:
            epoch, acc, loss, lr, val_acc, val_loss = map(float, row)
            train_losses.append(loss)
            train_accs.append(acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            epochs.append(epoch)

    plt.figure(1)

    # display losses
    plt.subplot(211)
    plt.plot(epochs, train_losses, 'b', label='train loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r', label='val loss', linewidth=2)
    plt.legend(loc='best')

    # display accuracy
    plt.subplot(212)
    plt.plot(epochs, train_accs, 'b', label='train acc', linewidth=2)
    plt.plot(epochs, val_accs, 'r', label='val acc', linewidth=2)
    plt.legend(loc='best')

    plt.show()
