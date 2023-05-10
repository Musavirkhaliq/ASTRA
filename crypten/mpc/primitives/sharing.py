        rank = comm.get().get_rank()
        # if source is zero 
        if src == 0:
            if rank == 0:
                lambda_v1 = next_share
                lambda_v2 = current_share
                mv = V + lambda_v1 + lambda_v2
                # send to 1 and 2 
                return lambda_v1, lambda_v2
            elif rank == 1:
                lambda_v1 = current_share
                mv = 0
                #recieve mv from 0 
                return lambda_v1, mv
            elif rank == 2:
                lambda_v2 = next_share
                #recieve mv from 0
                mv = 0
                return lambda_v2, mv
        elif src == 1:
            if rank == 0:
                lambda_v1 = next_share
                lambda_v2 = same_share
                return lambda_v1, lambda_v2
            elif rank == 1:
                lambda_v1 = current_share
                mv = V + lambda_v1 + same_share
                #send mv to 2
                return lambda_v1, mv
            elif rank == 2:
                lambda_v2 = same_share
                mv = 0
                #recieve mv from 1
                return lambda_v2, mv
        elif src == 2:
            if rank == 0:
                lambda_v1 = same_share
                lambda_v2 = current_share
                return lambda_v1, lambda_v2
            elif rank == 1:
                lambda_v1 = same_share
                mv = 0
                #recive mv from 2
                return lambda_v1, mv
            elif rank == 2:
                lambda_v2 = next_share
                mv = V + lambda_v2 + same_share
                #send mv to 1
                return lambda_v2, mv
            



################################################################################################
def __init__(
        self,
        tensor=None,
        size=None,
        broadcast_size=False,
        precision=None,
        src=0,
        device=None,
    ):
        """
        Creates the shared tensor from the input `tensor` provided by party `src`.

        The other parties can specify a `tensor` or `size` to determine the size
        of the shared tensor object to create. In this case, all parties must
        specify the same (tensor) size to prevent the party's shares from varying
        in size, which leads to undefined behavior.

        Alternatively, the parties can set `broadcast_size` to `True` to have the
        `src` party broadcast the correct size. The parties who do not know the
        tensor size beforehand can provide an empty tensor as input. This is
        guaranteed to produce correct behavior but requires an additional
        communication round.

        The parties can also set the `precision` and `device` for their share of
        the tensor. If `device` is unspecified, it is set to `tensor.device`.
        """

        # do nothing if source is sentinel:
        if src == SENTINEL:
            return

        # assertions on inputs:
        assert (
            isinstance(src, int) and src >= 0 and src < comm.get().get_world_size()
        ), "specified source party does not exist"
        if self.rank == src:
            assert tensor is not None, "source must provide a data tensor"
            if hasattr(tensor, "src"):
                assert (
                    tensor.src == src
                ), "source of data tensor must match source of encryption"
        if not broadcast_size:
            assert (
                tensor is not None or size is not None
            ), "must specify tensor or size, or set broadcast_size"

        # if device is unspecified, try and get it from tensor:
        if device is None and tensor is not None and hasattr(tensor, "device"):
            device = tensor.device

        # encode the input tensor:
        self.encoder = FixedPointEncoder(precision_bits=precision)
        if tensor is not None:
            if is_int_tensor(tensor) and precision != 0:
                tensor = tensor.float()
            tensor = self.encoder.encode(tensor)
            tensor = tensor.to(device=device)
            size = tensor.size()

        # if other parties do not know tensor's size, broadcast the size:
        if broadcast_size:
            size = comm.get().broadcast_obj(size, src)

        # generate pseudo-random zero sharing (PRZS) and add source's tensor:
        self.share = AstraSharedTensor.astrashares(size, device=device)
        # print(f"current_share {self.share[0]} process {self.rank}" )
        # print(f"next_share   {self.share[1]} process {self.rank}" )
        # print(f"same_share   {self.share[2]} process {self.rank}" )

        #  =
        # .share
        
        # if self.rank == src:
            # self.share += tensor
        # rank = self.rank
        # if source is zero 
        # if src == 0:
        #     if rank == 0:
        #         lambda_v1 = self.share[1]  #self.share[1]  #next_share
        #         lambda_v2 = self.share[0] #current_share
        #         mv = tensor + lambda_v1 + lambda_v2
        #         # send to 1 and 2 
        #         return lambda_v1, lambda_v2
        #     elif rank == 1:
        #         lambda_v1 = self.share[0] #current_share
        #         mv = 0
        #         #recieve mv from 0 
        #         return lambda_v1, mv
        #     elif rank == 2:
        #         lambda_v2 = self.share[1]  #next_share
        #         #recieve mv from 0
        #         mv = 0
        #         return lambda_v2, mv
        # elif src == 1:
        #     if rank == 0:
        #         lambda_v1 = self.share[1]  #next_share
        #         lambda_v2 = self.share[2]  #same_share
        #         return lambda_v1, lambda_v2
        #     elif rank == 1:
        #         lambda_v1 = self.share[0] #current_share
        #         mv = tensor + lambda_v1 + self.share[2]  #same_share
        #         #send mv to 2
        #         return lambda_v1, mv
        #     elif rank == 2:
        #         lambda_v2 = self.share[2]  #same_share
        #         mv = 0
        #         #recieve mv from 1
        #         return lambda_v2, mv
        # elif src == 2:
        #     if rank == 0:
        #         lambda_v1 = self.share[2]  #same_share
        #         lambda_v2 = self.share[0] #current_share
        #         return lambda_v1, lambda_v2
        #     elif rank == 1:
        #         lambda_v1 = self.share[2]  #same_share
        #         mv = 0
        #         #recive mv from 2
        #         return lambda_v1, mv
        #     elif rank == 2:
        #         lambda_v2 = self.share[1]  #next_share
        #         mv = tensor + lambda_v2 + self.share[2]  #same_share
        #         #send mv to 1
        #         return lambda_v2, mv

################################################################################################


    @staticmethod
    def astrashares(*size, device=None):
        """
        Generate a Pseudo-random Sharing of Zero (using arithmetic shares)

        This function does so by generating `n` numbers across `n` parties with
        each number being held by exactly 2 parties. One of these parties adds
        this number while the other subtracts this number.
        """
        from crypten import generators

        tensor = AstraSharedTensor(src=SENTINEL)
        if device is None:
            device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        g0 = generators["prev"][device]
        g1 = generators["next"][device]
        g2 = generators["global"][device]
        current_share = generate_random_ring_element(*size, generator=g0, device=device) #r1
        self.share[1]  #next_share = generate_random_ring_element(*size, generator=g1, device=device) #r2
        self.share[2]  #same_share = generate_random_ring_element(*size, generator=g2, device=device) # same accross all parties

        return current_share, self.share[1]  #next_share, self.share[2]  #same_share
        
        # tensor.share = current_share - self.share[1]  #next_share
        # return tensor


################################################################################################