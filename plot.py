import matplotlib.pyplot as plt
import argparse
import csv
import os


parser = argparse.ArgumentParser()

parser.add_argument('file', nargs='*', action='store',
                    help='a list of training csv files to plot')


def plot_csv(filepath, compare=False, color='r'):
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    epochs = []

    with open(filepath, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')

        # skip the header row
        next(reader)

        for row in reader:
            if len(row) > 0:
                epoch, acc, loss, lr, val_acc, val_loss = map(float, row)
                train_losses.append(loss)
                train_accs.append(acc)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                epochs.append(epoch)

    filename = os.path.basename(filepath)
    line_label = filename.split('_')[0]

    plt.figure(1)

    # display losses
    plt.subplot(211)
    if not compare:
        plt.plot(epochs, train_losses, 'b', label='train loss', linewidth=1)
        plt.plot(epochs, val_losses, 'r', label='val loss', linewidth=1)
    else:
        plt.plot(epochs, val_losses, color, label=line_label, linewidth=1)

    plt.legend(loc='best')

    # display accuracy
    plt.subplot(212)
    if not compare:
        plt.plot(epochs, train_accs, 'b', label='train acc', linewidth=1)
        plt.plot(epochs, val_accs, 'r', label='val acc', linewidth=1)
    else:
        plt.plot(epochs, val_accs, color, label=line_label, linewidth=1)

    plt.legend(loc='best')


if __name__ == '__main__':

    args = parser.parse_args()

    if len(args.file) == 1:
        plot_csv(args.file[0], compare=False)
    else:
        color = ['r', 'g', 'b', 'y']
        color_idx = 0
        for f in args.file:
            plot_csv(f, compare=True, color=color[color_idx])
            color_idx = (color_idx + 1) % len(color)

    plt.show()

