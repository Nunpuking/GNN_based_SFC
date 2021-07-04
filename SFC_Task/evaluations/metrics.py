

def ComputeFails(TDset, req_step, B):
    '''
    B is the batch_size
    N is the number of nodes in the topology

    - INPUT
    TDset    : <B> class TD, TopologyDrivers
    req_step : int         , time index of the current request in request sequences

    - OUTPUT
    n_reqs  : int, number of requests that are included in the failure ratio computation
    n_fails : int, number of failed requests

    '''

    n_reqs = 0
    n_fails = 0

    for b in range(B):
        TD = TDset[b]
        if req_step >= len(TD.reqs.keys()):
            continue
        n_reqs += 1
        req_idx = list(TD.reqs.keys())[req_step]
        req = TD.reqs[req_idx]

        if req['complete'] == False:
            n_fails += 1

    return n_reqs, n_fails

def ComputeDelayRatios(TDset, req_step, B):
    '''
    B is the batch_size
    N is the number of nodes in the topology

    - INPUT
    TDset    : <B> class TD, TopologyDrivers
    req_step : int         , time index of the current request in request sequences

    - OUTPUT
    n_reqs       : int  , number of requests that are included in the failure ratio computation
    delay_ratios : float, ratio of generation path's delay over label path's delay

    '''

    n_reqs = 0
    delay_ratios = 0.0

    for b in range(B):
        TD = TDset[b]
        if req_step >= len(TD.reqs.keys()):
            continue
        req_idx = list(TD.reqs.keys())[req_step]
        req = TD.reqs[req_idx]

        if req['complete'] == False:
            continue

        n_reqs += 1

        label_delay = TD.ComputeDelay(req_idx, mode='Label', predict_mode=TD.predict_mode)
        gen_delay = TD.ComputeDelay(req_idx, mode='Generation', predict_mode=TD.predict_mode)

        delay_ratios += float(gen_delay/label_delay)

    return n_reqs, delay_ratios

def ComputeDelays(TDset, req_step, B):
    '''
    B is the batch_size
    N is the number of nodes in the topology

    - INPUT
    TDset    : <B> class TD, TopologyDrivers
    req_step : int         , time index of the current request in request sequences

    - OUTPUT
    n_reqs : int, number of requests that are included in the failure ratio computation
    delays : int, sum of generation path's delay

    '''

    n_reqs = 0
    delays = 0

    for b in range(B):
        TD = TDset[b]
        if req_step >= len(TD.reqs.keys()):
            continue
        req_idx = list(TD.reqs.keys())[req_step]
        req = TD.reqs[req_idx]

        if req['complete'] == False:
            continue

        n_reqs += 1

        gen_delay = TD.ComputeDelay(req_idx, mode='Generation')

        delay_ratios += gen_delay

    return n_reqs, delays

