import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Anatomy_constraint_Loss(nn.Module):
    def __init__(self, n_classes):
        super(Anatomy_constraint_Loss, self).__init__()
        self.n_classes = n_classes

    def soft_dilate(self, input_tensor):
        if len(input_tensor.shape) == 4:
            return F.max_pool2d(input_tensor, (3, 3), (1, 1), (1, 1))
        elif len(input_tensor.shape) == 5:
            input_tensor = F.max_pool3d(input_tensor, (3, 3, 3), (1, 1, 1), (1, 1, 1))
            return F.max_pool3d(input_tensor, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def forward(self, inputs, target, cls_weights=None):
        n, c, h, w, d = inputs.size()
        nt, ht, wt, dt = target.size()
        if h != ht and w != wt and d != dt:
            inputs = F.interpolate(inputs, size=(ht, wt, dt), mode="bilinear", align_corners=True)

        inputs_d = torch.argmax(torch.softmax(inputs, dim=1), dim=1).float()
        inputs_d = self.soft_dilate(inputs_d)
        ins = F.relu(target - inputs_d)

        ce_tensor = torch.unsqueeze(nn.CrossEntropyLoss(weight=cls_weights, reduction='none')(inputs, target),dim=1)
        ins = torch.unsqueeze(ins, dim=1)
        ce_tensor[:, 0] = ce_tensor[:, 0] * torch.squeeze(ins, dim=1)
        loss = ce_tensor.view(-1).mean()
        return loss

def CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss

class Dice_loss(nn.Module):
    def __init__(self, n_classes):
        super(Dice_loss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        inputs = inputs.float()
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes