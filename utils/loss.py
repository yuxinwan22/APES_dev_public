import torch


def consistency_loss(tgt):
    """
    :param tgt: should be a list of tensors in shape BxNxD
    """
    loss = torch.nn.MSELoss()
    loss_list = []
    for i in range(len(tgt)):
        for j in range(len(tgt)):
            if i < j:
                loss_list.append(loss(tgt[i], tgt[j]))
            else:
                continue
    return sum(loss_list) / len(tgt)
