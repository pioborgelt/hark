"""Training-only own clean posterior target with a bounded alignment tolerance."""
import torch

def paired_denoising_loss(student,teacher,lengths,temperature=2.):
    if student.shape!=teacher.shape or student.ndim!=3 or temperature<=0:
        raise ValueError('Paired posterior contract')
    lengths=torch.as_tensor(lengths,device=student.device)
    if lengths.shape!=(student.shape[0],) or torch.any(lengths<=0) or torch.any(lengths>student.shape[1]):
        raise ValueError('Paired posterior lengths')
    mask=torch.arange(student.shape[1],device=student.device)[None,:]<lengths[:,None]
    target=torch.softmax(teacher.detach()/temperature,-1)
    weighted=(target*mask[...,None]).transpose(1,2)
    target=torch.nn.functional.avg_pool1d(weighted,3,stride=1,padding=1).transpose(1,2)
    count=torch.nn.functional.avg_pool1d(mask[:,None].to(student.dtype),3,stride=1,padding=1).transpose(1,2)
    target=target/count.clamp_min(1e-12)
    logs=torch.log_softmax(student/temperature,-1)
    frame=(target*(target.clamp_min(1e-12).log()-logs)).sum(-1)*temperature**2
    # Include silence while emphasizing phonetic evidence from the own teacher.
    weight=(.1+1-target[...,0])*mask
    return ((frame*weight).sum(-1)/weight.sum(-1)).clamp_min(0.)
