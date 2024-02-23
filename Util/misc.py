#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch
import numpy as np
import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist
from torch import inf


# In[9]:


class SmoothedValue(object):
    """
        Track a series of values and provide access to smoothed values over a
        window or the global series average.
    """

    def __init__(self, window_size=25, fmt=None):
        if fmt is None:
            fmt = "med {median:.4f} (avg {global_avg:.4f})"
        # collections.deque()是双端队列，可以实现左右两端添加或取出元素的功能。最大长度为maxlen
        # deque可以构造一个固定大小的队列,当超过队列之后,会把前面的数据自动移除掉
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    # 1。保护类的封装特性 2。让开发者可以使用“对象.属性”的方式操作操作类属性
    # 通过 @property 装饰器，可以直接通过方法名来访问方法，不需要在方法名后添加一对“（）”小括号。
    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def nsum(self, n = 1):
        d = torch.tensor(list(self.deque)[-n:], dtype=torch.float32)
        return torch.sum(d, 0).item()

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)




# In[10]:


class MetricLogger(object):
    # object写不写没有区别
    def __init__(self, delimiter="\t"):
        """
            Python中通过Key访问字典，当Key不存在时，会引发‘KeyError’异常。
            为了避免这种情况的发生，可以使用collections类中的defaultdict()方法来为字典提供默认值
            字典的value值类型是传入的参数。
        """
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    # print或者return时，返回的就不是内存地址，显示更友好，实现了类到字符串的转化。
    # 直接使用str(object)即可调用
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}', window_size=25)
        data_time = SmoothedValue(fmt='{avg:.4f}', window_size=25)
        # 计算iterable的总长度，在format的时候保留几位整数。如：50 就是2
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        # eta: 预计到达时间
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'iter time: {time}',
            'load data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        # 把log_msg使用 分隔符 组合成一条字符串
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        num_sum = 0
        for obj in iterable:
            # 从iterable中取出一组数据所用的时间
            data_time.update(time.time() - end)
            # 返回obj，并在那边处理完后再执行下部分的代码。就像插了一个断点一样
            yield obj
            # 从iterable中 取出一组数据 并 传回 并 将传回的数据处理完 所用的全部时间
            iter_time.update(time.time() - end)
            # 第一个以及每print_freq个以及最后一个batch时执行，打印最近20个iter的运行状况
            num_sum += 1
            if i % print_freq == 0 or i == len(iterable) - 1:
                # 计算还需要多少秒结束本epoch的运算
                eta_seconds = iter_time.global_avg * (len(iterable) - (i + 1))
                # 转换为本次epoch结束的时间
                today = datetime.datetime.now()
                eta_second = datetime.timedelta(seconds=int(eta_seconds))
                eta_time = str((today + eta_second).strftime("%H:%M:%S"))
                eta_string = str(eta_second) + "  " + eta_time
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i + 1, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=iter_time.nsum(num_sum), data=data_time.nsum(num_sum),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i + 1, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=iter_time.nsum(num_sum), data=data_time.nsum(num_sum)))
                num_sum = 0
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))




# In[11]:


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process

    builtins是Python中的一个模块。该模块提供对Python的所有“内置”标识符的直接访问；
    例如，builtins.open 是内置函数的全名 open() 。
    description: 修改python的内置函数print，使其附带时间戳，且当前进程不是主进程时不打印。
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        """
        pop(key[,default])
        参数
        key: 要删除的键值
        default: 如果没有 key，返回 default 值
        返回值
        返回被删除的值
        """
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now()
            builtin_print('[{}] '.format(now.strftime("%Y-%m-%d %H:%M:%S")), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print



# In[12]:


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True



# In[ ]:


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


# In[ ]:


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


# In[ ]:


def is_main_process():
    return get_rank() == 0


# In[ ]:


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


# In[13]:


def init_distributed_mode(args):
    if args.dist_on_itp: #感觉应该是多机多卡
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ: # 应该是单机多卡
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE']) # gpu数目
        args.gpu = int(os.environ['LOCAL_RANK']) # 当前第几个卡上
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else: # 单卡或者cpu
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier() # 等待所有的任务都已经分布式初始化完成之后，再执行下面的任务，一般放在分布式训练的实例初始化之后的最后一行
    setup_for_distributed(args.rank == 0)
    
    


# In[14]:


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        """
        amp : 全称为 Automatic mixed precision，自动混合精度，
        可以在神经网络推理过程中，针对不同的层，采用不同的数据精度进行计算，从而实现节省显存和加快速度的目的
        Tensor精度：
        torch.FloatTensor（浮点型 32位）(torch默认的tensor精度类型是torch.FloatTensor)
        torch.HalfTensor（半精度浮点型 16位）优势就是存储小、计算快、更好的利用CUDA设备的Tensor Core. 劣势就是：数值范围小、舍入误差,导致一些微小的梯度信息丢失
        解决方案:
        torch.cuda.amp.GradScaler，通过放大loss的值来防止梯度消失underflow。真正更新权重的时候会自动把放大的梯度再unscale回去，所以对用于模型优化的超参数不会有任何影响
        """
        self._scaler = torch.cuda.amp.GradScaler()


    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        """
            作用其实是把一个类的实例化对象变成了可调用对象.
            a = People('无忌！')
            a.__call__()       # 调用方法一
            a()                # 调用方法二
        """
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            """
            为了尽可能的减少梯度underflow，scaler应该更大；
            但是如果太大的话，半精度浮点型的tensor又容易overflow（变成inf或者NaN）。
            所以动态估计的原理就是在不出现inf或者NaN梯度值的情况下尽可能的增大scaler的值——
            在每次scaler.step(optimizer)中，都会检查是否又inf或NaN的梯度出现
                1，如果出现了inf或者NaN，scaler.step(optimizer)会忽略此次的权重更新（optimizer.step() )，
                   并且将scaler的大小缩小（乘上backoff_factor）；

                2，如果没有出现inf或者NaN，那么权重正常更新，并且当连续多次（growth_interval指定）没有出现inf或者NaN，
                   则scaler.update()会将scaler的大小增加（乘上growth_factor）。

            """
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)



class background_step():

    def __init__(self):
        super(background_step, self).__init__()


    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        loss.backward()
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                norm = get_grad_norm_(parameters)
            optimizer.step()
        else:
            norm = None
        return norm

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        return {}


# In[15]:


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm



# In[16]:


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, mAP):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    # loss_scaler不是none表示为训练过程暂存的checkpoint
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s-%s-(%.4f).pth' % (args.model, epoch_name, mAP.item()))]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_status': mAP,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    # 表示测试阶段暂存模型的状态，方便下次从断点处继续测试？？？
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)



# In[17]:


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        # 从检查点恢复
        # 从网络导入
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        # 从本地导入
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        if 'model' in checkpoint:
            model_without_ddp.load_state_dict(checkpoint['model'])
            print("Resume checkpoint %s" % args.resume)
            # 确保是训练阶段不是eval，并且元素都在
            if  'best_status' in checkpoint:
                args.best_status = checkpoint['best_status']
            if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch']
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
        else:
            model_state = model_without_ddp.state_dict()
            # check
            for k in list(checkpoint.keys()):
                if k in model_state:
                    shape_model = tuple(model_state[k].shape)
                    shape_checkpoint = tuple(checkpoint[k].shape)
                    if shape_model != shape_checkpoint:
                        checkpoint.pop(k)
                else:
                    print(k)
                    checkpoint.pop(k)
            msg = model_without_ddp.load_state_dict(checkpoint, strict=False)
            print("Resume checkpoint %s" % args.resume)
            print(msg)


# In[1]:


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


# In[ ]:


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            if len(loss_dict[k].size()) == 0:
                all_losses.append(loss_dict[k].unsqueeze(0))
            else:
                all_losses.append(loss_dict[k])
        
        all_losses = torch.stack(all_losses, dim=0)
        torch.distributed.reduce(all_losses, dst=0)
        if torch.distributed.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

