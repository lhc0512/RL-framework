from multiprocessing import Pool
import os


def f(x):
    print('Child process id:', os.getpid())

    return x * 2


if __name__ == '__main__':
    print('Parent process id:', os.getpid())
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))
