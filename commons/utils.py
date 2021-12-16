import matplotlib.pyplot as plt


def plot_figure(rewards, label, filename):
    plt.figure()
    plt.ylim([0, 105])
    plt.cla()
    plt.plot(range(len(rewards)), rewards)
    plt.xlabel("episode")
    plt.ylabel(label)
    plt.savefig(f"{filename}.png", format="png")
    plt.close()
