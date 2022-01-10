import matplotlib.pyplot as plt


def plot_figure(data, xlabel, ylabel, filename):
    plt.figure()
    plt.ylim([0, 105])
    plt.cla()
    plt.plot(range(len(data)), data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f"{filename}", format="png")
    plt.close()
