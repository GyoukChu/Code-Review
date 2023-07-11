import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import argparse
import torch.distributed as dist
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from MySampler import DistributedEvalSampler
# See: https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/sampler.py

class BasicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.act = F.relu

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.dropout2(x)
        output = self.fc2(x)
        return output

def main(opts):
	# init dist
    init_for_distributed(opts)
    local_gpu_id = opts.gpu

    # data set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307), (0.3081))
        ])

    train_set = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('data', train=False, transform=transform)

    train_sampler = DistributedSampler(dataset=train_set, shuffle=True)
    '''
    https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/sampler.py
    '''
    '''
    DistributedEvalSampler is different from DistributedSampler.
    It does NOT add extra samples to make it evenly divisible.
    DistributedEvalSampler should NOT be used for training. The distributed processes could hang forever.
    See this issue for details: https://github.com/pytorch/pytorch/issues/22584
    '''
    test_sampler = DistributedEvalSampler(dataset=test_set, shuffle=False)
    test_sampler2 = DistributedSampler(dataset=test_set, shuffle=False)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=int(64 / opts.world_size), # batch_size=64
                               shuffle=False,
                               num_workers=int(opts.num_workers / opts.world_size),
                               sampler=train_sampler,
                               pin_memory=True)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=int(64 / opts.world_size),
                             shuffle=False,
                             num_workers=int(opts.num_workers / opts.world_size),
                             sampler=test_sampler,
                             pin_memory=True)

    # model
    model = BasicNet()
    model = model.cuda(local_gpu_id)
    model = DDP(module=model,
                device_ids=[local_gpu_id])

    # criterion
    criterion = torch.nn.CrossEntropyLoss().to(local_gpu_id)

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # scheduler
    scheduler = StepLR(optimizer=optimizer, step_size=30, gamma=0.1)
    
    for epoch in range(1): # training for one epoch
        # train
        model.train()
        '''
        In distributed mode, calling the set_epoch() method at the beginning of each epoch 
        before creating the DataLoader iterator is necessary to make shuffling work properly 
        across multiple epochs. Otherwise, the same ordering will be always used.
        '''
        train_sampler.set_epoch(epoch)

        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(local_gpu_id), target.to(local_gpu_id)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # test
        model.eval()

        correct = 0
        with torch.no_grad():
            for _, (data, target) in enumerate(test_loader):
                data = data.to(opts.rank)
                labels = target.to(opts.rank)
                outputs = model(data)
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
        correct = torch.Tensor([correct]).to(opts.rank)
        global_result = [torch.zeros_like(correct) for _ in range(opts.world_size)]
        
        
        if opts.rank == 0:
            dist.gather(correct, gather_list=global_result)
        else:
            dist.gather(correct, dst=0)

        if opts.rank == 0:
            global_result_tensor = torch.cat(global_result)
            global_correct = torch.sum(global_result_tensor)
            #print(train_sampler.total_size)
            #print(test_sampler.total_size)
            #print(test_sampler2.total_size)
            print(f'Accuracy: {100. * global_correct / len(test_loader.dataset)}')
            
            scheduler.step()

    cleanup()
    return


def init_for_distributed(opts):

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        opts.rank = int(os.environ["RANK"])
        opts.world_size = int(os.environ['WORLD_SIZE'])
        opts.gpu = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(opts.gpu)

    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '23456'

    dist.init_process_group(backend='nccl', 
                            world_size=opts.world_size, 
                            rank=opts.rank)

    dist.barrier()
    setup_for_distributed(opts.rank == 0)


def setup_for_distributed(is_master):
    """
    Disable printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--gpu_ids', nargs="+", default=['6', '7'])
    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = "1" # Single Thread
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(v) for v in args.gpu_ids])

    args.world_size = len(args.gpu_ids)
    args.num_workers = len(args.gpu_ids) * 4
    main(args)

# # RUN: torchrun --nproc_per_node=2 --nnodes=1 DDP_example.py