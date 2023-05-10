import sys
sys.path.insert(0, '/home/musa/Documents/distributed/crypten')
import torch
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm 
crypten.init()
torch.set_num_threads(1)

#Constructing CrypTensors with ptype attribute

#arithmetic secret-shared tensors
x_enc = crypten.cryptensor([1.0, 5.0, 3.0], ptype=crypten.mpc.arithmetic)
print("x_enc internal type:", x_enc)

#binary secret-shared tensors
y = torch.tensor([1, 2, 1], dtype=torch.int32)
y_enc = crypten.cryptensor(y, ptype=crypten.mpc.arithmetic)
print("y_enc internal type:", y_enc)


z = x_enc + y_enc
print("z=",z.get_plain_text())

# @mpc.run_multiprocess(world_size=3)
# def examine_arithmetic_shares():
#     x_enc = crypten.cryptensor([1, 2, 3], ptype=crypten.mpc.arithmetic)
#     # x_enc = crypten.cryptensor([-6239757564308448057, -1129533114741977653,  5749444835519892985])
#     print("plaintext:", x_enc.get_plain_text())

#     rank = comm.get().get_rank()
#     crypten.print(f"\nRank {rank}:\n {x_enc}\n", in_order=True)
        
# x = examine_arithmetic_shares()
