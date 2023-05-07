from torchvision import transforms
from gaussian_blur import GaussianBlur

def get_byol_data_transforms(input_shape, s = 1):
    color_jitter = transforms.ColorJitter(0.8* s, 0.8*s, 0.8*s, 0.2*s)
    data_transforms = transforms.Compose([
        transforms.RandomresizedCrop(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p = 0.8),
        transforms.RandomGrayscale(p = 0.2),
        GaussianBlur(kernel_size = int(0.1*eval(input_shape[0]))),
        transforms.ToTensor()
    ])

