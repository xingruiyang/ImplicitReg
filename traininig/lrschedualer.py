class BaseSchedualer():
    def get_lr(self, n_epoch):
        return None


class DecaySchedualer(BaseSchedualer):
    def __init__(self, start, step_size, step) -> None:
        self.start = start
        self.step_size = step_size
        self.step = step

    def get_lr(self, n_epoch):
        return self.start*(self.step_size**(n_epoch//self.step))


class ConstantSchedualer(BaseSchedualer):
    def __init__(self, value) -> None:
        self.lr = value

    def get_lr(self, n_epoch):
        return self.lr


class ManualStepSchedualer(BaseSchedualer):
    def __init__(self, start, step, lr) -> None:
        self.start_lr = start
        self.step = step
        self.lr = lr

    def get_lr(self, n_epoch):
        num_step = len(self.step)
        if num_step == 0 or n_epoch < self.step[0]:
            return self.start_lr
        for i in range(num_step-1):
            if n_epoch >= self.step[i] and n_epoch < self.step[i+1]:
                return self.lr[i]
        if n_epoch >= self.step[num_step-1]:
            return self.lr[num_step-1]
        raise ValueError("ManualStepSchedualer can't parse lr!")


def get_schedualer(cfg_str):
    stype = cfg_str["type"]
    param = cfg_str["param"]
    if stype == 'decay':
        return DecaySchedualer(**param)
    if stype == 'const':
        return ConstantSchedualer(**param)
    if stype == 'manual':
        return ManualStepSchedualer(**param)
    else:
        raise ValueError("unkown schedualer type {}".format(stype))


def load_schedualers(slist):
    return {k: get_schedualer(s) for k, s in slist.items()}
