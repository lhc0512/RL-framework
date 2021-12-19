from multiprocessing import Process, Queue
import os


def f(q):
    '''
    q: a Queue
    '''

    print('parent process:', os.getppid())
    print('process id:', os.getpid())

    # Add elements to the queue.
    q.put([42, None, 'hello'])


if __name__ == '__main__':
    q = Queue()
    # The subprocess here adds elements to the queue.
    p = Process(target=f, args=(q,))
    p.start()
    # Retreive the array from the queue.
    print(q.get())  # prints "[42, None, 'hello']"
    p.join()
