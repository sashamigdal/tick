__author__ = 'sasha'

import calendar
import csv
import ctypes
import datetime
import mmap
import os
import shutil
import stat
import warnings
from datetime import date
from fractions import Fraction

import itertools
import numpy as np
import pytz as pytz
from time import time, sleep, localtime, strftime

MAT_DGTS = 32

if MAT_DGTS == 16:
    MAT_TYPE = np.float16
    MAT_FMT = '<f4'  # cannot read money and volume and flow unless use 1M as a unit
    c_mat_p = ctypes.POINTER(ctypes.c_int16)
elif MAT_DGTS == 32:
    MAT_TYPE = np.float32
    MAT_FMT = '<f4'
    c_mat_p = ctypes.POINTER(ctypes.c_float)
elif MAT_DGTS == 64:
    MAT_TYPE = np.float64
    MAT_FMT = '<f8'
    c_mat_p = ctypes.POINTER(ctypes.c_double)
else:
    MAT_TYPE = np.float
    MAT_FMT = '<f8'

STR_LEN = 8
SYM_FMT = 'S%d' % STR_LEN

DEBUG_PRINT = False
PRINT_TIME = True


def TMS():
    return  strftime("%H:%M:%S :", localtime())
#
# def print(*args, **kwargs):
#     if DEBUG_PRINT:
#         msg = __builtins__.print(args, kwargs)
#         if PRINT_TIME: msg = "%s: %s" % (TMS(), msg)
#         return msg
#
class Beacon():
    def __init__(self, name, period = 60, action = None):
        self.mm = int(time()/period)
        self.name = name
        self.action = action
        self.period = period
    def __call__(self, *args, **kwargs):
        mm = int(time()/self.period)
        if mm > self.mm:
            self.mm = mm
            print("%s: sitting in %s" % (TMS(), self.name))
            if self.action is not None:
                self.action()
            return True
        return False

class memoryCheck():
    """Checks memory of a given system"""

    def __init__(self):

        if os.name == "posix":
            self.value = self.linuxRam()/1000
            print("linux ram = ", self.value)
        elif os.name == "nt":
            self.value = 1e5#self.windowsRam()
            print("windows crashes here on windowsRam() call")  # ram = ", self.value/1000

        else:
            print("I only work with Win or Linux :P")

    def windowsRam(self):
        """Uses Windows API to check RAM in this OS"""
        kernel32 = ctypes.windll.kernel32
        c_ulong = ctypes.c_ulong
        class MEMORYSTATUS(ctypes.Structure):
            _fields_ = [
                ("dwLength", c_ulong),
                ("dwMemoryLoad", c_ulong),
                ("dwTotalPhys", c_ulong),
                ("dwAvailPhys", c_ulong),
                ("dwTotalPageFile", c_ulong),
                ("dwAvailPageFile", c_ulong),
                ("dwTotalVirtual", c_ulong),
                ("dwAvailVirtual", c_ulong)
            ]
        memoryStatus = MEMORYSTATUS()
        memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUS)
        kernel32.GlobalMemoryStatus(ctypes.byref(memoryStatus))

        return int(memoryStatus.dwTotalPhys/1024**2)

    def linuxRam(self):
        """Returns the RAM of a linux system"""
        totalMemory = os.popen("free -m").readlines()[1].split()[1]
        return int(totalMemory)

    def __call__(self, *args, **kwargs):
        return self.value

# GetRAM = memoryCheck()
GIGABYTE = 1073741824
def HasClassName(A,C):
    return A.__class__.__name__ == C.__name__

def IsString(A):
    return A.__class__.__name__ in ('str','string_')

def GetMaxLenFromMem(dataType, mem):
    return int(mem /  ( np.dtype(dataType).itemsize + 2 * np.dtype(np.int32).itemsize))


def GetNNZFromDSMatrixPath(fname):
    if os.path.exists(fname + '.data'):
        path = fname + '.data'
    elif os.path.exists(fname + '.data.z'):
        path = fname + '.data.z'
    else:
        return 0
    return os.path.getsize(path) // np.dtype(MAT_TYPE).itemsize
MAX_UNIX_MINUTE = (2050 -2000)*366*24*60#max value of minutes since 2000
MAX_TICKER = 200000  #hardly there will be more than 20K distinct histories of US equities

def testMatType():
    s = MAT_TYPE.itemsize
    x = MAT_TYPE(5.5)

    t = type(x)
    pass

def fromCstring(s):
    return ''.join(itertools.takewhile('\0'.__ne__,s))

warnings.filterwarnings('ignore')

def ListDirAlphabethical(dir):
    return list(sorted(os.listdir(dir)))

def ListDirChronological(path):
    mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
    return list(sorted(os.listdir(path), key=mtime))

def GetSize(path):
    if not os.path.isfile(path): return 0
    return os.stat(path).st_size

def GetTimestamp(path):
    if not os.path.isfile(path): return 0
    return os.path.getmtime(path)

def NumLines(path):
    if GetSize(path) ==0: return 0
    f = open(path, "r+")
    buf = mmap.mmap(f.fileno(), 0)
    lines = 0
    readline = buf.readline
    while readline():
        lines += 1
    return lines

def makeRWDir(dirname):
    os.mkdir(dirname)
    os.chmod(dirname,stat.S_IRUSR | stat.S_IWUSR)



# BAD_SUFFIXES = set(['WD', 'WI', 'WS', 'Q', 'S' 'U', 'RT', 'AWI'])

GOOD_SUFFIXES = set(['A', 'B', 'D', 'P', 'O', 'M', 'N','CL'])

# GOOD_SUFFIXES = set()
# with open('/data/COMMON_DATA/suffixes.csv') as input:
#     for s in input.readlines():
#         if s == '\n':continue
#         GOOD_SUFFIXES.add(s.split('\n')[0])


def WriteAnimationFiles(t, u, name, Dir, ext = 't', frame = 0):

    assert len(t) == len(u), "WriteAnimationFiles: Arrays must be the same length"
    assert u.dtype == MAT_TYPE

    path = Dir + "Movie/"
    MakeDir(path)
    t.toFile(path + name + ".gpl_" + ext + "%s"%frame)
    u.toFile(path + name +  ".gpl_u" + "%s"%frame)

def MeanSharpe(pnls,yearly=True):
    if len(pnls) ==0: return 0,0
    m,e = pnls.meanStd()
    if e <=0: return 0,0
    N = 251. if yearly else len(pnls) -1.
    return m, m/e*np.sqrt(N)



def NormalizedData(data, scale=0):
    if scale ==0: return data.sign()
    return  data/ (data.sqr() + scale*scale).sqrt()

def RecoverDot(S):
    if '/' in S:
        return S.replace('/','.')
    elif '$' in S:
        return S.replace('$','.')
    elif '-'  in S:
        return S.replace('-','.')
    elif '_'  in S:
        return S.replace('_','.')
    elif ' ' in S:
        return S.replace(' ', '.')
    else:
        return S

def recover_dots(symbols):
    return np.array(map(RecoverDot,symbols),dtype=SYM_FMT)

def IsValidUSTicker(S):
    S = RecoverDot(S)
    if S == '': return False
    if len(S) > 8: return False
    if S[-1] == '.': return False
    if S.startswith('TEST'): return False
    if S.endswith('ZZT'): return False
    if S.startswith('Z') and S.endswith('ZZ'): return False
    return S.replace('.','').isalpha()

def ReadColumnsWithHeader(filename):
    with open(filename, 'r') as input:
        L = list(csv.reader(input))
    header = L[0]
    Cols = zip(*L[1:])
    D = dict(zip(header,Cols))
    return D

def testSuffixes():
    todaySymbolsArray = np.array(['CPA.WS', 'SSS.WI', 'MSFT'])
    OK = np.array(map(IsValidUSTicker, todaySymbolsArray), dtype=np.bool)
    todaySymbols = set(todaySymbolsArray[OK])
    pass

##  makes ticker or Cusip from the daily filename. Returns either like "ABCD" or "Cusip.12354678"
def getSymbolFromDailyFilename(filename):
    if not filename.startswith("Daily_"): return None
    if not filename.endswith(".csv"): return None
    BloomSym = filename[6:].replace(".csv", "")
    if BloomSym.startswith(".Cusip"):
        #.Cusip.12354678
        return BloomSym.split('.')[-1][:8]
    return None




def Overlap(a,b,c,d):
    '''
    do intervals [a,b) and [c,d) overlap?
     we do not use this function, keep it just in case
    '''

    if a is None or b is None: return False
    if a < c:
        return c <= b and c <= d
    elif c < a:
        return a <= d and a <= b
    else:# a == c
        return a <= b and c <= d



def DailyDataIntegrityCheck(close, ids, factors, date):
    assert close.shape == ids.shape and factors.shape == ids.shape, \
        'shape mismatch in IntegrityCheck close: %s, factors:%s, ids:%s' % (close.shape, factors.shape, ids.shape)
    num_good_ids = (ids >= 0).sum()
    assert (close > 0).sum() >= num_good_ids, 'date = %s, close prices (%s) vs ids (%s) mismatch' % (
        date, (close > 0).sum(), num_good_ids)
    assert (factors > 0).sum() >= num_good_ids, 'date = %s, eod_multipliers (%s) vs ids (%s) mismatch' % (
        date, (factors > 0).sum(), num_good_ids)


def SelectGoodList(lst, Good, axis=0):
    if axis == 0:
        return map(lambda x: x[Good], lst)
    elif axis == 1:
        return map(lambda x: x[:, Good], lst)
    else:
        return map(lambda x: x[..., Good], lst)


def testSelectGood():
    A = np.arange(10)
    B = np.ones(10)
    C = np.zeros(10)
    OK = (A < 5)
    A, B, C = SelectGoodList([A, B, C], OK)
    pass


def SelectGoodArray(ar, Good):
    assert Good.ndim == 1 and ar.ndim > 1
    return ar[..., Good]


def testSelectGoodArrayInPlace():
    A = np.arange(24).reshape((2, 3, 4))
    OK = np.arange(3)
    A = SelectGoodArray(A, OK)
    pass


def GetNumberFromDate(dt):
    #'YYYY-MM-DD' to int YYYYMMDD
    if len(dt) == 10 and dt[4] == '-' and dt[7] == '-':
        return int(dt.replace("-", ""))  # AE: 58 s vs 59. same speed actually
        #return (int(dt[0:4]) * 100 + int(dt[5:7])) * 100 + int(dt[-2:])
    else:
        return -1


def GetDateFromNumber(N):
    D = N % 100
    Y = (N - D) / 100
    M = Y % 100
    Y = (Y - M) / 100
    return date(Y, M, D)



def TestDaysFromNumber(init_date='2007-11-10'):
    N = GetNumberFromDate(init_date)
    DT = GetDateFromNumber(N)
    new_date = DT.isoformat()
    assert init_date == new_date, 'wrong date conversion'


from scipy.stats import rankdata


def stockWeight(money):
    assert money.ndim == 1, ' wrong dimension in stock weight'
    return rankdata(money) / len(money)


def stockWeightsPerMinute(minute_money):
    assert minute_money.ndim == 2, ' wrong dimension in stock weight'
    # R = np.data(map(stockWeight,[minute_money[k, :] for k in range(minute_money.shape[0])]))
    R1 = np.array(map(lambda k: stockWeight(minute_money[k, :]), range(minute_money.shape[0])))
    # assert (R1 == R).all()
    return R1


def test_rank2():
    mm = np.random.uniform(size=(100, 10))
    ranks = stockWeightsPerMinute(mm)
    pass


def filterPrices(close_prices):
    return close_prices > 1.


def filterVolume(volume):
    return volume > 10000.


def filterMoney(today_money):
    return today_money > 1e4


def pf(f, x):
    z = np.zeros_like(x)
    pos = x > 0
    z[pos] = f(x[pos])
    return z


def pinv(A):
    return pf(lambda x: 1 / x, A)


def test_Pinv():
    minute_close = np.random.normal(size=(10, 100))
    dp = np.diff(minute_close, axis=0)
    ip = pinv(minute_close[:-1])
    assert dp.shape == ip.shape
    dp_over_p = dp * ip
    pass


def plog(A):
    return pf(lambda x: np.log(x), A)

def VolPack(v):
    return pf(lambda x: np.log(1.+x), v)

def VolUnPack(p):
    return np.exp(p) -1.

def testPexp():
    a = np.arange(-5,5,1,dtype=MAT_TYPE)
    b = VolPack(a)
    c = VolUnPack(b)
    pass
def pAlog(A):
    return pf(lambda x: x * np.log(x), A)


def pAlog2(A, T):
    return pf(lambda x: x * np.log2(x * T), A)


def pf2(f, x, y):
    '''
    this is instead of using np.nan_to_num, which makes npinfty large numbet like log(0.)
    '''
    assert x.shape == y.shape, ' x y mismatch'
    z = np.zeros_like(x)
    pos = np.logical_and(x > 0, y > 0)
    z[pos] = f(x[pos], y[pos])
    return z


def plogRatio(x, y):
    return pf2(lambda a, b: np.log(a / b), x, y)

def pRatio(x,y):
    z = np.zeros_like(x)
    nz = np.logical_and(x!= 0, y != 0)
    z[nz] = x[nz]/ y[nz]
    return z

def pDiff(x,y):
    return pf2(lambda a, b: a - b, x, y)





def SubtractShifted(a):
    """
        Makes an array of differences `b`, such that:

        b[0] = a[0],
        b[i] = a[i] - a[i-1], for i = 1,..., N-1.

        Sum of elements of `b` will be equal to a[N-1]. Some elements in `b` can be zero.
        :param a: input array
        :return: numpy array of differences
    """
    b = np.zeros_like(a)
    b[0] = a[0]
    b[1:] = a[1:] - a[:-1]
    return b


def GetCopyZeroedAfterTime(array, timestamps, ts_zero_after):
    """
        Filters an array by time. Both `timestamps` and `ts_zero_after` must be timestamps or nanotimestamps,
        or anything time-like.
        :param array:           Must be parallel to `timestamps`.
        :param timestamps:      Array of Unix (nano)timestamps.
        :param ts_zero_after:   Unix (nano)timestamp.
        :return:
    """
    assert len(array) == len(timestamps), 'GetCopyZeroedAfterTime(): Mismatch of arrays length'
    time_filter = (timestamps >= ts_zero_after)
    result = array.copy()
    result[time_filter] = 0
    return result

def CyclicAppendInPlace(X, A):
    L = A.shape[0]
    X[:-L] = X[L:]
    X[-L:] = A

def ProbNegative(data):
    return (data<0).sum()/np.float(len(data))

def ProbPositive(data):
    if len(data) ==0: return 0
    return (data>0).sum()/np.float(len(data))

def MeanSign(data):
    if len(data) ==0: return 0
    return data.sign().mean()

def Quantized(var,quant):
    return (var/quant).round() *quant


def MakeDir(dir):
    try:
        if not os.path.isdir(dir):
            os.makedirs(dir)
    except Exception as e:
        print("makedirs exception :", e)
        pass
    # try:
    #     os.chmod(dir,0777)
    # except Exception as e:
    #     # print "chmod exception :", e
    #     pass

    return dir


def RemDir(dir):
    if os.path.isdir(dir):
        try:
            shutil.rmtree(dir)
        except OSError:
            print("[ERROR] RemDir({})".format(dir))


def MakeNewDir(dir):
    RemDir(dir)
    try:
        os.makedirs(dir)
    except Exception as e:
        print("makedirs exception :", e)
        pass

def RemoveFiles(dir, word):
    for filename in os.listdir(dir):
        if word in filename:
            path = dir + filename
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)



def ArgAndMax(lst):
    M = max(lst)
    return lst.index(M), M

def testArgMax():
    X = range(10)
    i,M = ArgAndMax(X)
    pass

def testNone():
    x = (None in ('a','b'))
    pass


def IsaacSorter(D):
    keys = D.keys()
    values = D.values()
    before=[]
    after=[]
    oper =0
    lk=len(keys)+1
    while len(keys)>0:
        if len(keys) >= lk:
            raise ValueError('Cyclical renames.')
        lk = len(keys)
        for ind, key in enumerate(keys):
            if key not in values:
                before.append(key)
                after.append(values[ind])
                keys.pop(ind)
                values.pop(ind)
                oper += 1
                break
    # print "Isaac reordered in %d operations"%oper
    before=before[::-1]
    after=after[::-1]
    return before,after

def testFractionLimit(x=3.14):
    print(Fraction(x).limit_denominator(10))


def ArchiveFolder(folder, Dir):
    if not os.path.isdir(folder): return
    archive_dir = Dir + 'archive/%s/'%strftime('%Y-%m-%d:%H:%M:%S')
    try:
        MakeDir(archive_dir)
        shutil.move(folder,archive_dir)
    except OSError:
        print("stepan!!!!!!!!!!!!!!!! permissions")


class Alarm():
    def __init__(self,name):
        self._name = name
        self._snooze = False
        self._delay =0
        self.HMS = np.infty

    def Set(self, H=-1, M=0, S=0,wait=0.):
        self.HMS = (H*60 +M)*60 + S
        self._delay = wait
        self._snooze = False

    def Snooze(self):
        self._snooze = True

    def __call__(self, tt=None):
        if self._snooze : return False
        if tt is None:
            tt = localtime()
        if (tt.tm_hour*60 +tt.tm_min)*60 + tt.tm_sec < self.HMS:
            sleep(self._delay)
            return False
        self._snooze = True
        print("%02d:%02d:%02d Alarm [%s] delay=%s" % (tt.tm_hour, tt.tm_min, tt.tm_sec, self._name, self._delay))
        return True

    def Passed(self):
        return self._snooze

class EachSlice():
    def __init__(self,name, slice=60):
        self._name = name
        self._delay =0
        self._slice = slice
        self.HMS = np.infty

    def Set(self, H=0, M=0, S=0,wait=0.):
        self.HMS = H*3600 + M * 60 + S
        self._delay = wait


    def __call__(self, tt=None):
        if tt is None:
            tt = localtime()
        MS = tt.tm_hour*3600 + tt.tm_min*60 + tt.tm_sec
        if  MS < self.HMS:
            sleep(self._delay)
            return False
        self.HMS = ((MS + self._slice)//60)*60
        # print "%02d:%02d:%02d EachSlice [%s] delay=%s"%(tt.tm_hour,tt.tm_min,tt.tm_sec, self._name,self._delay)
        return True

class MultiAlarm():
    def __init__(self,name):
        self._name = name
        self._snooze = False
        self._delay =0
        self._alarms = set()

    def Set(self, H=-1, M=0, S=0,wait=0.):
        a = Alarm(self._name)
        a.Set(H,M,S,wait)
        self._alarms.add(a)

    def Snooze(self):
        self._snooze = True

    def __call__(self, tt=None):
        if self._snooze : return False
        for a in self._alarms:
            if a(tt): return True
        return False

    def Passed(self):
        return self._snooze

def IsSplit(factor,eps =1e-3):
    approx = Fraction(np.float(factor)).limit_denominator(10)
    return (approx != 1.) and abs(approx-factor) < eps


def AreSplits(factors,eps=1e-3):
    ok = map(lambda f:IsSplit(f,eps),factors)
    return np.array(ok)

def saveMaxID(dir,ID):
    filename = dir + 'maxid.txt'
    file(filename, 'w').write(str(ID) + "\n")
    print('saved MAX_ID = %s to %s' % (ID, filename))


def loadMaxID(dir):
    filename = dir + 'maxid.txt'
    if os.path.isfile(filename):
        MAX_ID = int(file(filename).read())
        # print 'reading MAX_ID = %s from %s'%(MAX_ID, filename)
        return MAX_ID
    else:
        return 0


########################################################################################################################
def getDayMonthYear(int_date):
    day = int_date%100
    YYYYMM = int_date//100
    year = YYYYMM//100
    month = YYYYMM%100
    return day,month,year

def TodayInt():
    return int(strftime('%Y%m%d'))

def YesterdayInt():
    return int((datetime.datetime.now() - datetime.timedelta(1)).strftime('%Y%m%d'))


def DateTimeFromTimestamp64(timestamp64):
    """
    Formats timestamp to a human-readable format, preserving at least one digit after point.
    Uses system timezone.
    Examples:
        1564581600000000000 --> ('2019-07-31', '10:00:00.0')
        1564581600100000000 --> ('2019-07-31', '10:00:00.1')
        1564581600123456789 --> ('2019-07-31', '10:00:00.123456789')
    :param timestamp64: UNIX nanotimestamp
    :return: tuple of strings
    """
    if isinstance(timestamp64, np.uint64):
        timestamp64 = int(timestamp64)
    timestamp = timestamp64 / 1000000000
    nanoseconds = timestamp64 % 1000000000
    dt = datetime.datetime.fromtimestamp(timestamp)

    sDate = dt.strftime('%Y-%m-%d')
    sTime = dt.strftime('%H:%M:%S')
    subsecond = '.{:0>9.0f}'.format(nanoseconds)
    subsecond = subsecond.rstrip('0')
    if subsecond.endswith('.'):
        subsecond += '0'
    return sDate, sTime + subsecond


def Timestamp64_from_ET_DateTime(int_date, hour, minute, second=0):
    """
        Returns Unix nanotimestamp from datetime of ET timezone
        :param int_date: int, YYYYMMDD
        :param hour: int
        :param minute: int
        :param second: int
    """
    year = int_date // 10000
    month = int_date // 100 % 100
    day = int_date % 100

    dt = datetime.datetime(year, month, day, hour, minute, second)

    tz_eastern = pytz.timezone('US/Eastern')
    tz_utc = pytz.timezone('UTC')
    dt_utc = tz_eastern.localize(dt).astimezone(tz_utc)
    dt_utc = tz_utc.normalize(dt_utc)

    ts = calendar.timegm(dt_utc.utctimetuple())
    return ts * 1000000000


########################################################################################################################
def NormalizeInvestment(V):
    norm = np.abs(V).sum()
    if norm > 0:
        V /=norm

def MaybeRemove(path):
    if os.path.isfile(path):
        os.remove(path)

def testNorm():
    V = np.array([1,-2,3],dtype=MAT_TYPE)
    NormalizeInvestment(V)
    pass
def testCumsum():
    v = np.arange(10,dtype=MAT_TYPE)
    vv = np.cumsum(v, axis=0)
    pass



def testSpeedSortRenames(N=1000):
    from random import shuffle
    r = zip(range(N), range(1, N+1))
    shuffle(r)
    D = dict(r)
    t0 = time()
    i0, i1 = IsaacSorter(D)
    t1 = time()
    print("N=%d Isaac: %.4f secs" % (N, t1 - t0))
    t0 = time()
    s0, s1 = IsaacSorter(D)
    t1 = time()
    print("N=%d sasha: : %.4f secs" % (N, t1 - t0))


def test_slash():
    name = "sasha/migdal"
    s = ('/')

def test_multi_alarm():
    multialarm = MultiAlarm('test_multialarm')

    now = datetime.datetime.now()
    now += datetime.timedelta(minutes=1)
    print('multialarm: set {}'.format(now))
    multialarm.Set(H=now.hour, M=now.minute, S=now.second)

    now += datetime.timedelta(minutes=1)
    print('multialarm: set {}'.format(now))
    multialarm.Set(H=now.hour, M=now.minute, S=now.second)

    while not multialarm():
        sleep(0.1)
    print('multialarm: passed {}'.format(datetime.datetime.now()))


def testSaleCond():
    ALLTAQ = "@ BCEFHIKLMNOPTUVXZ456789"
    EXCLUDETAQ = "GPZBDHKMOQL"
    left = ''
    for a in ALLTAQ:
        if not a in EXCLUDETAQ:
            left += a
        pass
    print("TAQ allowed sale cond: %s" % left)
    ALLLS = "@ A B C D E F G H K L M N O P Q R S T U V W X Y Z 1 2 3 4 5 6 7 8 9"
    ALLLS = ALLLS.replace(' ','')
    ALLLS = ALLLS.replace('A',' A')
    good_ls = ''
    for b in ALLLS:
        if b in left:
            good_ls += b
        pass
    print("good ls = %s" % good_ls)


if __name__ == '__main__':
    print("WARNING: This __main__ has not been verified to work since PR698 cleaned up the imports.")
    print("     ..: This __main__ will need to be run from the magadan root directory.")
    print("     ..: See https://fresnelresearch.slack.com/archives/C09SH2EPK/p1556512199006900")
    print("     ..: and https://github.com/Migdal-Research/magadan/pull/698")
    print("     ..: When you verify this __main__ works, please remove this message.")
    ok =IsValidUSTicker('ZVZZ.T')
    exit()
    testNorm()
    exit()
    test_multi_alarm()
    exit()
    testSaleCond()
    test_slash()
    # testSpeedSortRenames(10000)
    # exit()
    testCumsum()
    ok = IsSplit(11/10.+0.0001)
    testFractionLimit()
    exit()
    testNone()
    testArgMax()

    testPexp()
    testMatType()


    # testRepeat()

