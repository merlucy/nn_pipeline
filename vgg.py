import os
import threading
import time
from functools import wraps

import torch
import torch.nn as nn
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef

num_classes = 1000


class Stage0(torch.nn.Module):
    def __init__(self):
        super(Stage0, self).__init__()
        self._lock = threading.Lock()
        self.layer2 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer3 = torch.nn.ReLU(inplace=True)
        self.layer4 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self._initialize_weights()
        self.layer5 = torch.nn.ReLU(inplace=False)
        self.layer6 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    def forward(self, x_rref):
        x = x_rref.to_here().to("cpu")
        with self._lock:
            out2 = self.layer2(x)
            out3 = self.layer3(out2)
            out4 = self.layer4(out3)
            out5 = self.layer5(out4)
            out6 = self.layer6(out5)

        print("end of stage 0 forward")
        return out6

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]


class Stage1(torch.nn.Module):
    def __init__(self):
        super(Stage1, self).__init__()
        self._lock = threading.Lock()

        self.layer3 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer4 = torch.nn.ReLU(inplace=True)
        self.layer5 = torch.nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer6 = torch.nn.ReLU(inplace=True)
        self.layer7 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer8 = torch.nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer9 = torch.nn.ReLU(inplace=True)
        self._initialize_weights()
        self.layer11 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer12 = torch.nn.ReLU(inplace=True)
        self.layer13 = torch.nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer14 = torch.nn.ReLU(inplace=True)
        self.layer15 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer16 = torch.nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer17 = torch.nn.ReLU(inplace=True)
        self.layer18 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x_rref):
        x = x_rref.to_here().to("cpu")
        with self._lock:
            out3 = self.layer3(x)
            out4 = self.layer4(out3)
            out5 = self.layer5(out4)
            out6 = self.layer6(out5)
            out7 = self.layer7(out6)
            out8 = self.layer8(out7)
            out9 = self.layer9(out8)

            out11 = self.layer11(out9)
            out12 = self.layer12(out11)
            out13 = self.layer13(out12)
            out14 = self.layer14(out13)
            out15 = self.layer15(out14)
            out16 = self.layer16(out15)
            out17 = self.layer17(out16)
            out18 = self.layer18(out17)

        print("end of stage 1 forward")
        return out18

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]


class Stage2(torch.nn.Module):
    def __init__(self):
        super(Stage2, self).__init__()
        self._lock = threading.Lock()

        self.layer7 = torch.nn.ReLU(inplace=False)
        self.layer8 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer9 = torch.nn.ReLU(inplace=True)

        self._initialize_weights()
        self.layer11 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer12 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer13 = torch.nn.ReLU(inplace=True)
        self.layer14 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer15 = torch.nn.ReLU(inplace=True)
        self.layer16 = torch.nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer17 = torch.nn.ReLU(inplace=True)
        self.layer18 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer19 = torch.nn.Linear(in_features=8192, out_features=4096, bias=True)
        self.layer20 = torch.nn.ReLU(inplace=True)
        
    def forward(self, x_rref):
        x = x_rref.to_here().to("cpu")
        with self._lock:

            out7 = self.layer7(x)
            out8 = self.layer8(out7)
            out9 = self.layer9(out8)

            out11 = self.layer11(out9)
            out12 = self.layer12(out11)
            out13 = self.layer13(out12)
            out14 = self.layer14(out13)
            out15 = self.layer15(out14)
            out16 = self.layer16(out15)
            out17 = self.layer17(out16)
            out18 = self.layer18(out17)
            out19 = out18.size(0)
            out20 = out18.view(out19, -1)
            out21 = self.layer19(out20)
            out22 = self.layer20(out21)

        print("end of stage 2 forward")
        return out22

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]

class Stage3(torch.nn.Module):
    def __init__(self):
        super(Stage3, self).__init__()
        self._lock = threading.Lock()

        self.layer13 = torch.nn.Dropout(p=0.5)
        self.layer14 = torch.nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.layer15 = torch.nn.ReLU(inplace=True)
        self.layer16 = torch.nn.Dropout(p=0.5)
        self.layer17 = torch.nn.Linear(in_features=4096, out_features=1000, bias=True)
        self._initialize_weights()        

    def forward(self, x_rref):
        x = x_rref.to_here().to("cpu")
        with self._lock:

            out13 = self.layer13(x)
            out14 = self.layer14(out13)
            out15 = self.layer15(out14)
            out16 = self.layer16(out15)
            out17 = self.layer17(out16)
        
        print("end of stage 3 forward")

        return out17

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]


class DistVggNet(nn.Module):
    """
    Assemble two parts as an nn.Module and define pipelining logic
    """
    def __init__(self, workers, *args, **kwargs):
        super(DistVggNet, self).__init__()

        # Put the first stage on workers[0]
        self.p1_rref = rpc.remote(
            workers[0],
            Stage0,
            args = args,
            kwargs = kwargs,
            timeout=0
        )
        # Put the second stage on workers[1]
        self.p2_rref = rpc.remote(
            workers[1],
            Stage1,
            args = args,
            kwargs = kwargs,
            timeout=0
        )
        # Put the third stage on workers[2]
        self.p3_rref = rpc.remote(
            workers[2],
            Stage2,
            args = args,
            kwargs = kwargs,
            timeout=0
        )
        # Put the fourth stage on workers[3]
        self.p4_rref = rpc.remote(
            workers[3],
            Stage3,
            args = args,
            kwargs = kwargs,
            timeout=0
        )

    def forward(self, x):

        out_futures = []    # List to store futures received from the workers
        tensors = x[0]
        offset = x[1]
        length = x[2]
        index = length + offset

        # Executes each stages depending on the stage each tensor is supposed to be fed into
        if offset < 1:
            rref_1 = RRef(tensors[0])
            p1_out_rref = self.p1_rref.rpc_async().forward(rref_1)
            out_futures.append(p1_out_rref)     # Maintains the order of input list stages
        if index >= 2 and offset < 2:
            rref_2 = RRef(tensors[1 - offset])
            p2_out_rref = self.p2_rref.rpc_async().forward(rref_2) 
            out_futures.append(p2_out_rref)
        if index >= 3 and offset < 3:
            rref_3 = RRef(tensors[2 - offset ])
            p3_out_rref = self.p3_rref.rpc_async().forward(rref_3) 
            out_futures.append(p3_out_rref)
        if index >= 4 and offset < 4:
            rref_4 = RRef(tensors[3 - offset])
            p4_out_rref = self.p4_rref.rpc_async().forward(rref_4) 
            out_futures.append(p4_out_rref)

        out_tensors = torch.futures.wait_all(out_futures)   # Collects all the output tensors

        for i in range(len(out_tensors)):
            out_tensors[i] = torch.cat([out_tensors[i]])    # Prevents the problem from each tensor being a leaf tensor

        return out_tensors  # Changed from returning concatenated tensors to returning list of output tensors

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.p1_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p2_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p3_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p4_rref.remote().parameter_rrefs().to_here())
        return remote_params


#########################################################
#                   Run RPC Processes                   #
#########################################################

num_batches = 128
batch_size = 1
image_w = 128
image_h = 128


def run_master():

    # put the two model parts on worker1 and worker2 respectively
    model = DistVggNet(["worker1", "worker2", "worker3", "worker4"])
    loss_fn = nn.MSELoss()
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
    )

    one_hot_indices = torch.LongTensor(batch_size) \
                           .random_(0, num_classes) \
                           .view(batch_size, 1)

    # Uses input_list to store the intermediary tensors, and input_list feeds the tensors to the model
    input_list = []
    offset = 0  # Index to track which earlier stages are not executed. Also used to shift the input_list
    # batch_no = 0

    # Runs the loop num_batches + 3 times to execute the remaining + 3 stages of forwarding.
    for i in range(num_batches + 3):

        input_col = []
        inputs = ''

        if i < num_batches:
            print(f"Processing batch {i}")
            # generate random inputs and labels
            inputs = torch.randn(batch_size, 3, image_w, image_h)
            labels = torch.zeros(batch_size, num_classes) \
                        .scatter_(1, one_hot_indices, 1)

            input_list = [inputs] + input_list
            offset = 0
        else:
            offset += 1

        # Append the offset, index, and length data to send them to the model
        input_col.append(input_list)
        input_col.append(offset)
        input_col.append(len(input_list))

        results = model(input_col) # Runs the Stages with the given inputs

        # The distributed autograd context is the dedicated scope for the
        # distributed backward pass to store gradients, which can later be
        # retrieved using the context_id by the distributed optimizer.
        with dist_autograd.context() as context_id:
            if offset + len(results) >= 4:
                # print("backward for batch ", batch_no )
                dist_autograd.backward(context_id, [loss_fn(results[-1], labels)])
                opt.step(context_id)
                # batch_no += 1

        input_list = []     # Reset the input_list

        for j in range(0, len(results), 1):
            if j + offset + 1 == 4:     # Drop the last element of results, which is already backward propagated
                break
            input_list.append(results[j])   # Re-insert the results


def run_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=600)

    import psutil
    p = psutil.Process()
    
    if rank == 0:
        p.cpu_affinity([0])
        print(f"Child #{rank}: Set my affinity to {rank}, affinity now {p.cpu_affinity()}", flush=True)

        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_master()
    else:
        p.cpu_affinity([rank-1])
        print(f"Child #{rank}: Set my affinity to {rank}, affinity now {p.cpu_affinity()}", flush=True)

        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        pass

    # block until all rpcs finish
    rpc.shutdown()


if __name__=="__main__":
    world_size = 5
    tik = time.time()
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
    tok = time.time()
    print(f"execution time = {tok - tik}")