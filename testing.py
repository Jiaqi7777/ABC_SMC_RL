from time import sleep
from random import random
from multiprocessing import Process
from multiprocessing import Event, current_process
 
# target task function

class main_cls():
    def __init__(self):
        self.tcls = t()
        self.b = 0

    def fn(self):
        event = Event()
        # create a suite of processes
        processes = [Process(target=self.task, args=(event, i)) for i in range(9)]
        # start all processes
        for process in processes:
            process.start()
        # block for a moment
        print('Main process blocking...')
        sleep(4)
        # trigger all child processes
        event.set()
        # wait for all child processes to terminate
        for process in processes:
            process.join()

    def update_main(self):
        self.b += 1

    def task(self, event, number):
        # wait for the event to be set
        print(f'Process {number} waiting...', flush=True)
        event.wait()
        # begin processing
        value = random()
        self.tcls.update()
        self.update_main()
        sleep(value)
        print(f'Process {number} got {value}' + "here" + str(self.tcls.a) + "id" + str(id(self.tcls)) + "herehere" + str(self.b) + "id2" + str(id(self)), flush=True)

class t():
    def __init__(self):
        self.a = 0
    
    def update(self):
        self.a += 2


# entry point
if __name__ == '__main__':
    # create a shared event object
    cls = main_cls()
    cls.fn()
    print("final", cls.b)