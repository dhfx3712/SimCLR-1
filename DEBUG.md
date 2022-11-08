train_loader
utils.py->CIFAR10Pair(self.transform通过参数传入，random的作用pos1、pos2输出不同)




net_input : pos1-torch.Size([2, 3, 32, 32]) ,pos2-torch.Size([2, 3, 32, 32])
net_out : feature_1-torch.Size([2, 2048]),out_1-torch.Size([2, 128]) ,feature_2-torch.Size([2, 2048]) ,out_2-torch.Size([2, 128])
net_out_cat : torch.Size([4, 128])
net sim_matrix : torch.Size([4, 4]), tensor([[7.3891, 0.9955, 6.6494, 0.9894],
        [0.9955, 7.3891, 1.0675, 6.6062],
        [6.6494, 1.0675, 7.3891, 0.8638],
        [0.9894, 6.6062, 0.8638, 7.3891]], grad_fn=<ExpBackward0>)
net mask : torch.Size([4, 4]) , tensor([[False,  True,  True,  True],
        [ True, False,  True,  True],
        [ True,  True, False,  True],
        [ True,  True,  True, False]])
net mask_select : tensor([0.9955, 6.6494, 0.9894, 0.9955, 1.0675, 6.6062, 6.6494, 1.0675, 0.8638,
        0.9894, 6.6062, 0.8638], grad_fn=<MaskedSelectBackward0>)
net sim_matrix_mask : torch.Size([4, 3]) , tensor([[0.9955, 6.6494, 0.9894],
        [0.9955, 1.0675, 6.6062],
        [6.6494, 1.0675, 0.8638],
        [0.9894, 6.6062, 0.8638]], grad_fn=<ViewBackward0>)
net_pos : torch.Size([2]) 
net_pos_sim : tensor([6.6494, 6.6062, 6.6494, 6.6062], grad_fn=<CatBackward0>) , sim_matrix_dim 归一化作用: tensor([8.6344, 8.6692, 8.5807, 8.4595], grad_fn=<SumBackwa , log : tensor([0.2612, 0.2718, 0.2550, 0.2473], grad_fn=<NegBackward0>)




    out1  | out2
   ---｜---｜---｜---|
   b1  b2   b1   b2 
b1
b2
b1
b2

matrix相似方阵

torch.sum(out_1 * out_2, dim=-1) --b1与b1点乘，b2与b2点乘。加合
torch.cat([pos_sim, pos_sim], dim=0)拼接作用是与matrix方阵的行保持一致

loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean() 分子部分都是b1与b1,b2与b2相似值，分母是所有相似的加合






