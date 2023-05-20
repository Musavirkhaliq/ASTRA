import crypten
import crypten.communicator as comm
import torch
from crypten.encoder import FixedPointEncoder
from crypten import generators
from crypten.mpc.primitives import AstraSharedTensor
from crypten.common.tensor_types import is_float_tensor, is_int_tensor, is_tensor
import crypten.mpc as mpc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def check():
    rank = comm.get().get_rank()
    if rank == 0:
        x = crypten.cryptensor([1,2,3],src=0, ptype=crypten.mpc.astraB)
        return x
    if rank == 1:
        x = crypten.cryptensor([0,0,0],src=0, ptype=crypten.mpc.astraB)
        return x
    if rank == 2:
        x = crypten.cryptensor([0,0,0],src=0, ptype=crypten.mpc.astraB)
        return x