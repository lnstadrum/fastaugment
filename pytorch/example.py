import matplotlib.pyplot as plt
import random
import torch
import timeit

from torchvision.datasets import Flowers102
from torchvision.transforms import PILToTensor
from torch.utils.data import DataLoader
from fast_augment_torch import CenterCrop, FastAugment


# benchmark on a synthetic input
augment = FastAugment()
batch = (torch.rand(128, 224, 224, 3) * 255).type(torch.uint8).cuda()
labels = torch.zeros(batch.shape[0], 1000)
number = 1000
time = timeit.timeit(lambda: augment(batch, labels), number=number)
print('Sampling %d batches of %s size in %0.3f s' % (number, batch.shape, time))

# make a test batch
dataset = Flowers102('flowers',
                     download=True,
                     transform=PILToTensor())

crop = CenterCrop((500, 500))
x = []
y = []
for _ in range(20):
    image, label = dataset[random.randint(0, len(dataset))]
    x.append(crop(image.cuda().permute(1, 2, 0).contiguous()))
    y.append(torch.nn.functional.one_hot(torch.LongTensor([label - 1]), num_classes=102))

x = torch.stack(x, dim=0)
y = torch.cat(y).to(torch.float32)

# run data augmentation
augment = FastAugment(mixup=0.5)
x, y = augment(x, y)

# plot images
plt.figure(1)
for i in range(len(x)):
    # show image
    plt.subplot(4, 5, i + 1)
    plt.axis(False)
    plt.imshow(x[i].cpu())

    # get labels and probabilities
    proba = y[i]
    idx = torch.where(proba > 0)[0]
    proba = 100 * proba[idx]
    label = [str(i.item()) for i in idx]

    # add a title
    if len(proba) == 1:
        plt.gca().set_title('[%s]' % label[0])
    else:
        plt.gca().set_title('[%s] %d%% / [%s] %d%%' % (label[0], proba[0], label[1], proba[1]))

plt.show()

