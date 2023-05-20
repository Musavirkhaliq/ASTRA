
import sys
sys.path.insert(0, '/home/musa/Documents/distributed/crypten')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
#import the libraries
import crypten
import torch
import crypten.mpc as mpc
import crypten.communicator as comm 
from crypten.mpc.primitives.comparator import*
from crypten.mpc.primitives.sampletestcomp import*

#initialize crypten
crypten.init()
#Disables OpenMP threads -- needed by @mpc.run_multiprocess which uses fork
torch.set_num_threads(1)
@mpc.run_multiprocess(world_size=3)
def examine_arithmetic_shares():
    x_enc = crypten.cryptensor([5,2,3],ptype=crypten.mpc.astra)
    # print(x_enc.shape)
    y_enc = crypten.cryptensor([4,2,6], ptype=crypten.mpc.astra)
    # # print(y_enc.shape)
    # # w_enc = crypten.cryptensor([4,120,3], ptype=crypten.mpc.astra)
    z_enc = x_enc - y_enc
    w_enc = bitext(x_enc,y_enc,z_enc)

    # rank = comm.get().get_rank()
    # crypten.print(f"\nRank {rank}:\n {w_enc.get_plain_text()}\n", in_order=True)
    # # rank = comm.get().get_rank()
    # # crypten.print(f"\nRank {rank}:\n {y_enc}\n", in_order=True)
    # rank = comm.get().get_rank()
    # crypten.print(f"\nRank {rank}:\n {z_enc.ptype}\n", in_order=True)
    rank = comm.get().get_rank()
    crypten.print(f"\nRank {rank}:\n {w_enc.get_plain_text()}\n", in_order=True)
        
x = examine_arithmetic_shares()