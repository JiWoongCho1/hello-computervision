import torch
import shutil
import os
def save_checkpoint(state, is_best, filename = 'checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar') #메타데이터, 아키텍쳐 저장

def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)


def accuracy(output, target, topk = (1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True) #dim = 1, 내림차순 = True, 인덱스 반환 = True
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []

        for k in topk:
            correct_k = correct[:k].reshape(-1).float(),sum(0, keepdim = True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res