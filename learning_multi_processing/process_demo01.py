from multiprocessing import Process
import os


def f(name):
    print('Child process id:', os.getpid())
    print('hello', name)


if __name__ == '__main__':
    print('Parent process id:', os.getpid())
    p = Process(target=f, args=('bob',))

    # start(): Start the process’s activity.
    #
    # This must be called at most once per process object.
    # It arranges for the object’s run() method to be
    # invoked in a separate process.
    p.start()

    # join(timeout): Wait until the thread terminates.

    # This blocks the calling thread until the thread
    # whose join() method is called terminates – either
    # normally or through an unhandled exception – or
    # until the optional timeout occurs.
    p.join()
