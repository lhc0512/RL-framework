from multiprocessing import Process, Pipe
import os


def f(connection):
    print('parent process:', os.getppid())
    print('process id:', os.getpid())
    connection.send('hello')
    connection.close()


if __name__ == '__main__':
    parent_conn, child_conn = Pipe()
    p = Process(target=f, args=(child_conn,))
    p.start()
    print(parent_conn.recv())  # prints
    p.join()
