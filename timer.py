from mpi4py import MPI

class Timer:

        def __init__(self, P_x):
            
                self.P_x = P_x
                self.start_times = {}
                self.stop_times = {}
                self.messages = {}

        def start(self, key):

                if key in self.start_times:
                        raise ValueError('Duplicate timing key {key} in start times')

                self.P_x._comm.Barrier()
                self.start_times[key] = MPI.Wtime()

        def stop(self, key, message=None):

                if key not in self.start_times:
                        raise ValueError('Missing timing key {key} in start times')

                self.P_x._comm.Barrier()
                self.stop_times[key] = MPI.Wtime()
                self.messages[key] = message

        def dump_times(self, f):

                for k, stop in sorted(self.stop_times.items(), key=lambda x: x[1]):
                        start = self.start_times[k]
                        dt = stop-start

                        line = f'{k}, {start}, {stop}, {dt}'

                        m = self.messages[k]
                        if m is not None:
                                line += f', {m}'

                        f.write(f'{line}\n')

                self.start_times.clear()
                self.stop_times.clear()
                self.messages.clear()