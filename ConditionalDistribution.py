from queue import Queue, Empty
import imageio

import numpy as np
import multiprocessing as mp

from commonFunctions import MakeNewDir, RemoveFiles
from .plot import RankHist2, PLOT_DATA, PlotXYErrBars, Plot


class Conditional():
    def __init__(self,times, returns,signals):
        '''
        :param times: (T,1)stamps
        :param returns: (T,N) returns
        :param signals: (T,N) signals
        '''
        ord = np.argsort(times)
        self.times = times[ord]
        self.returns = returns[ord]
        self.signals = signals[ord]

    def FutureMove(self,name,chance,steps_fwd=10,take_abs=True):
        T = len(self.times)
        targets = self.returns  # (T,N) , return in BP for each stock
        strength = self.signals  # (T,N) , zeros where not traded
        if take_abs:
            strength = np.abs(strength)
        pool_strength = strength.flatten()
        targets = np.cumsum(targets, axis=0)
        pool_strength = pool_strength[pool_strength>0]
        id = int(chance * len(pool_strength))
        thresh = np.sort(pool_strength)[id]
        sums = np.zeros(steps_fwd, dtype=np.float)
        sums2 =np.zeros(steps_fwd, dtype=np.float)
        nums = np.zeros(steps_fwd, dtype=np.float)
        for t1 in range(1,T-steps_fwd):
            filter = (strength[t1]>thresh).astype(np.float)
            sgn = np.sign(self.signals[t1])
            num = filter.sum()
            for dt in range(steps_fwd):
                delta = targets[t1+dt]- targets[t1-1]
                delta *= sgn
                sums[dt] += delta.dot(filter)
                sums2[dt] += (delta**2).dot(filter)
                nums[dt] += num
            pass
        pass
        nums[nums==0] = 1
        means = sums/nums
        errs = np.sqrt((sums2/nums - means**2)/nums)
        steps = np.arange(steps_fwd,dtype=np.float)
        PlotXYErrBars(steps,means,errs,PLOT_DATA + '%s_mean_pnl(%d).png'%(name,int(chance*100)),title='MeanPnl(%d)'%int(chance*100))
        PlotXYErrBars(steps, sums, errs, PLOT_DATA + '%s_net_pnl(%d).png' % (name, int(chance * 100)),
                      title='NetPnl(%d)'%int(chance*100))


    def FutureMovesParallel(self,name,chances,num_cores = 16,steps_fwd=10):
        if num_cores <=0:
            for chance in chances:
                self.FutureMove(name, chance, steps_fwd)
            pass
        else:
            queue = mp.Queue()

            for chance in chances:
                queue.put(chance)

            def work(core):
                print
                "starting core %d" % core
                while True:
                    try:
                        chance = queue.get_nowait()
                        self.FutureMove(name,chance,steps_fwd)
                    except Empty:
                        break

            jobs = []
            for rank in range(num_cores):
                job = mp.Process(target=work, name="worker_%s" % rank, args=(rank,))
                job.start()
                jobs.append(job)
            for job in jobs:
                job.join()
            pass
        pass
        images = []
        for chance in chances:
            filename = PLOT_DATA + '%s_mean_pnl(%d).png' % (name, int(chance * 100))
            images.append(imageio.imread(filename))
        imageio.mimsave(PLOT_DATA + '%s_mean_pnl.movie.gif' % name, images)


    def TargetAboveThresholdParallel(self, name, chances, num_cores=16, take_abs=True):
        targets = self.returns.flatten() # (T*N) , return in BP for each stock
        strength = self.signals.flatten() #(T*N) , zeros where not traded
        if take_abs:
            targets *= np.sign(strength)
            strength = np.abs(strength)
        ord = np.argsort(-strength) ## decreasing order in strength, regardless day
        strength = strength[ord]
        targets = targets[ord]
        sum_nz = np.cumsum((strength>0).astype(strength.dtype))
        means = np.cumsum(targets)/ sum_nz
        errors = np.sqrt((np.cumsum(targets**2)/ sum_nz - means**2)/sum_nz)
        PlotXYErrBars(strength, means, errors, PLOT_DATA + '%s_pnl_above_thresh.png'%(name),title='PnlBPvsThresh')
        lims = sum_nz[-1]
        if num_cores <= 0:
            for c in chances:
                n = (c * lims).astype(np.int64)
                RankHist2(means[:n],
                          plotpath=PLOT_DATA + '%s_cond_hist_chance_%d.png' % (name, int(100 * c)),
                          name='RankHist(%d)'%int(100 * c))
            pass
        else:
            queue = mp.Queue()

            for chance in chances:
                n = (chance * lims).astype(np.int64)
                queue.put((chance,n))

            def work(core):
                print
                "starting core %d" % core
                while True:
                    try:
                        c,n = queue.get_nowait()
                        RankHist2(means[:n],
                                  plotpath=PLOT_DATA + '%s_cond_hist_chance_%d.png' % (name, int(100 * c)),
                                  name= 'RankHist(%d)'% (int(100 * c)))
                    except Empty:
                        break

            jobs = []
            for rank in range(num_cores):
                job = mp.Process(target=work, name="worker_%s" % rank, args=(rank,))
                job.start()
                jobs.append(job)
            for job in jobs:
                job.join()
            pass
        pass

        images = []
        for chance in chances:
            filename = PLOT_DATA + '%s_cond_hist_chance_%d.png'% (name, int(chance * 100))
            images.append(imageio.imread(filename))
        imageio.mimsave(PLOT_DATA + '%s_cond_hist_chance.movie.gif' % name, images)


def test_FutureMoves():
    T = 200
    N = 1000
    signals = np.random.normal(scale =1., size=(T, N))
    noise = np.random.normal(scale=0.1, size=(T, N))
    returns = signals + noise
    times = np.arange(T, dtype=np.int32)
    CND = Conditional(times, returns, signals)
    chances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.92, 0.95, 0.97, 0.99]
    RemoveFiles(PLOT_DATA,"mean_pnl")
    CND.FutureMovesParallel(name='test',chances=chances,num_cores=16, steps_fwd=10 )


def test_TestCondHist():
    '''
       :param T: number of days
       :param N: number of stocks
       :return:
       '''
    T = 200
    N = 1000
    signals = np.random.normal(scale=1, size=(T, N))
    noise = np.random.normal(scale=0.25, size=(T, N))
    returns = signals + noise
    times = np.arange(T, dtype=np.int32)
    CND = Conditional(times, returns, signals)
    chances = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.92, 0.95, 0.97, 0.99]
    RemoveFiles(PLOT_DATA,"cond_hist_chance")
    RemoveFiles(PLOT_DATA, "pnl_above_thresh")
    CND.TargetAboveThresholdParallel('test', chances, num_cores=16, take_abs=True)


def test_roll_cumsum():
    num_steps_fwd = 5
    sub_targets = np.arange(30,dtype=np.float).reshape((10,3))
    sum_targets= np.cumsum(sub_targets,axis=0)
    data = sum_targets.mean(axis=1)
    print(data)
