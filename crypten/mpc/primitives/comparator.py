import crypten
import crypten.communicator as comm
import torch
from crypten.encoder import FixedPointEncoder
from crypten import generators
from crypten.mpc.primitives import AstraSharedTensor
from crypten.common.tensor_types import is_float_tensor, is_int_tensor, is_tensor
import crypten.mpc as mpc
from crypten.mpc.primitives import sharingastra
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_msb(tensor):
    msb_tensor = torch.empty_like(tensor)  # Create an empty tensor of the same shape as the input tensor

    for i, val in enumerate(tensor):
      val = val >> 63
      msb_tensor[i] = val
    return (msb_tensor*(-1)).long()



def bitext(x,y,a,precision=None):
    x, y = x.share, y.share
    rank = comm.get().get_rank()

    a_share = a.share

    zero =torch.zeros_like(x[0])
    size = x[0].size()

    rs = AstraSharedTensor.astrashares(size, device=x.device).share
    # print(rs)
    if rank == 0:
        ra1 = torch.zeros_like(x[0])
        ra2 = torch.zeros_like(x[0])
        req2 = comm.get().irecv(ra1, src=1)
        req3 = comm.get().irecv(ra2, src=2)
        req2.wait()
        req3.wait()

        
        ra = ra1 + ra2
        
        
        # print(compute_msb(ra))
        # print(ra)
        
        p_share = torch.stack([torch.zeros_like(x[0]), torch.zeros_like(x[0])])

        # encode the input tensor:
        encoder = FixedPointEncoder(precision_bits=precision)
        if p_share is not None:
            if is_int_tensor(p_share) and precision != 0:
                p_share = p_share.float()
            p_share = encoder.encode(p_share)
            p_share = p_share.to(device=device)


        q = compute_msb(ra)
        print([1,q])

        q_share = crypten.cryptensor(q,src=0, ptype=crypten.mpc.astraB)
        # print(p_share)

        #this is the msb share
        q_share.share = p_share & q_share.share

        # hello = compute_msb(q_share.share)
        # print(hello)

        # #truncation 
        # if encoder.scale > 1:
        #     q_share.share = sharingastra.truncation(q_share.share, encoder.scale).share.data
        # print(q_share)

        return q_share
    

    if rank == 1:
        r = rs[1]
        r_ = rs[2]
        p = compute_msb(r)
        
        p_share = torch.stack([torch.zeros_like(p),p])
    
        print(p_share)

        # encode the input tensor:

        encoder = FixedPointEncoder(precision_bits=precision)
        if p_share is not None:
            if is_int_tensor(p_share) and precision != 0:
                p_share = p_share.float()
            p_share = encoder.encode(p_share)
            p_share = p_share.to(device=device)


        ap1 = a_share[1]-a_share[0] #ma - lamdba a1
        rap1 = (r * ap1) + r_
        # print(rap1)
        req0 = comm.get().isend(rap1, dst=0)
        req0.wait()

        #r is given as as agrument for running sake
        q_share = crypten.cryptensor(zero,src=0, ptype=crypten.mpc.astraB)
        


    

        #this is the msb share
        q_share.share = p_share & q_share.share
        # print(q_share)

        # #truncation 
        # if encoder.scale > 1:
        #     q_share.share = sharingastra.truncation(q_share.share, encoder.scale).share.data
    
        # hello = compute_msb(q_share.share)
        # print(hello)

        return q_share
    

    if rank == 2:
        r = rs[0]
        r_ = rs[2]
        p = compute_msb(r)
        # p = r


        p_share = torch.stack([torch.zeros_like(p),p])
        # encode the p_share tensor:
        encoder = FixedPointEncoder(precision_bits=precision)
        if p_share is not None:
            if is_int_tensor(p_share) and precision != 0:
                p_share = p_share.float()
            p_share = encoder.encode(p_share)
            p_share = p_share.to(device=device)


        ap2 = -a_share[0]  #lamdba a2
        
        rap2 = (r*ap2) - r_

        # print(rap2)
        req1 = comm.get().isend(rap2, dst=0)
        req1.wait()
        q_share = crypten.cryptensor(zero,src=0, ptype=crypten.mpc.astraB)


    
        
        

        #this is the msb share
        q_share.share = p_share & q_share.share

        # hello = compute_msb(q_share.share)
        # print(hello)
        # print(q_share)
        # #truncation 
        # if encoder.scale > 1:
        #     q_share.share = sharingastra.truncation(q_share.share, encoder.scale).share.data
        
        return q_share
    