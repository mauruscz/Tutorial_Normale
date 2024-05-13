import os
import numpy as np

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12, 9) # (w, h)

from tqdm import tqdm

from sklearn.model_selection import train_test_split

import pickle

n_epochs = 4000
batch_size = 146


lr = 0.0002
b1 = 0.5
b2 = 0.999
n_cpu = 8
latent_dim = 100
channels = 1

torch.manual_seed(3110)



#######------TRAINING DATA------#########
#######------#######------########
#######------#######------########


city = "NYC"
transp = "Bike"

dir = "./adj/" + transp + city
arrays = []
for filename in os.listdir(dir):
    if filename.endswith('.npy'):
        arrays.append(np.load(dir + "/"+filename))

v_train, v_test = train_test_split(arrays, test_size=0.2, random_state=42)
print("Training set length:", len(v_train))
print("Test set length:", len(v_test))

img_size = v_train[0].shape[1]


cuda = True if torch.cuda.is_available() else False
print(cuda)



class VectorialDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, transform=None):
        super(VectorialDataset, self).__init__()
        self.input_data = torch.tensor(np.expand_dims(input_data, axis = 1)).float() 
        #for Conv2d, we need a 4D tensor, with the first dimension being the batch size, the second the number of channels, and the third and fourth the height and width of the image.
        self.transform = transform

    def __len__(self):
        return self.input_data.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.input_data[idx, :]

        if self.transform:
          sample = self.transform(sample)

        return sample 


training_set = VectorialDataset(input_data=v_train)
dataloader = torch.utils.data.DataLoader(training_set, 
                                           batch_size=batch_size, 
                                           shuffle=True)
for i, imgs in enumerate(dataloader):
  print(i, imgs.shape)



#######------BLOCKS------#########
#######------#######------########
#######------#######------########

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2)) 
        #fully connected layer, from latent_dim to 128 * init_size ** 2. Essentially, it is a reshaping of the input, one-dimensional tensor for each item in the batch.

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1), #input-channels (feature maps dim), output-channels (feature-maps dim), kernel size, stride, padding
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.ReLU()
            #nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        #from the reshaped input, we get a 4D tensor, 
        #with the first dimension being the batch size, the second the number of channels (i.e. number of feature maps), and the third and fourth the init_size

        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)] 
            #pay attention to the stride: it is 2, so the image is downsampled by a factor of 2
            #in_filters and out_filters are the number of channels entering and exiting the convolutional layer. 3 is the kernel size, 2 is the stride, 1 is the padding.
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4 # 4 is the resulting dimension of the image after the 4 convolutions with stride 2
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid()) #128 channels of feature maps, each of size ds_size x ds_size

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity




#######------INITIALIZATION------#########
#######------#######------########
#######------#######------########


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2),weight_decay=1e-4)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr= lr, betas=( b1,  b2), weight_decay=1e-4)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


#######------TRAINING------#########
#######------#######------########
#######------#######------########

G_losses = []
D_losses = []

real_scores = np.zeros(n_epochs)
fake_scores = np.zeros(n_epochs)


for epoch in tqdm(range(n_epochs)) :
    for i, imgs in enumerate(dataloader):

        # Adversarial ground truths vectors. The length is the length of the batch.
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0],  latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        # i.e. it wants the discriminator to think that the gen_imgs are real (valid)
        # So it compares the output of the discriminator with the valid vector
        # The lower the loss, the lower the cross-entropy between the output of the discriminator and the valid vector
        # The more similar are the labels of gen_imgs and valid        
        
        g_loss = adversarial_loss(discriminator(gen_imgs), valid) 

        g_loss.backward() #computes the gradients of g_loss with respect to the generator's parameters
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples

        #How much is the discriminator able to say that a real image is real?
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        #How much is the discriminator able to say that a fake image is fake?
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

        #Let's take the average of the two losses
        d_loss = (real_loss + fake_loss) / 2

        #Accuracies
        outputs = discriminator(real_imgs)
        real_score = outputs
        outputs = discriminator(gen_imgs.detach())
        fake_score = outputs

        d_loss.backward()
        optimizer_D.step()

        
        if epoch%10 == 0 and i == len(dataloader)-1:
            print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
                    
        # Save Losses for plotting later and accuracies
        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())

        real_scores[epoch] = real_scores[epoch]*(i/(i+1.)) + real_score.mean().data*(1./(i+1.))
        fake_scores[epoch] = fake_scores[epoch]*(i/(i+1.)) + fake_score.mean().data*(1./(i+1.))

print("end")






















#######------PLOTS------#########
#######------#######------########
#######------#######------########

# Plot Losses
plt.figure(figsize=(12, 9))
plt.title("MoGAN-Training", size=23, fontweight="bold")
plt.plot(G_losses, label="Generator")
plt.plot(D_losses, label="Discriminator")
plt.xlabel("iterations", size=23)
plt.ylabel("loss", size=23)
plt.tick_params(labelsize=20)
plt.legend(prop={'size': 21})
plt.savefig("lossMoGAN.pdf")
plt.show(block=False)

# Plot Scores
plt.figure(figsize=(12, 9))
plt.title("MoGAN: Scores", size=23, fontweight="bold")
plt.plot(fake_scores, label='synthetic score')
plt.plot(real_scores, label='real score')
plt.xlabel("epochs", size=23)
plt.ylabel("score", size=23)
plt.tick_params(labelsize=20)
plt.legend(prop={'size': 18})
plt.savefig("scoresMoGAN.pdf")
plt.show(block=False)

#######------Fake set and dump------#########
#######------#######------########
#######------#######------########


fake_set = []
t = Tensor(np.random.normal(0, 1, (len(v_test),  latent_dim))) #instead of batch_size should be len(v_test), but depends on your GPU power.
t = generator(t).cpu().detach().numpy()
print(t.shape)

for i in range(0, t.shape[0]):
  fake_set.append(np.rint(t[i][0]).astype(int))

print("len of fake set", len(fake_set))


with open("./" + transp + city + "/fake_set.txt", "wb") as fp:   #Pickling
   pickle.dump(fake_set, fp) 

with open("./" + transp + city +"/v_test.txt", "wb") as fp:   #Pickling
   pickle.dump(v_test, fp) 

with open("./" + transp + city +"/v_train.txt", "wb") as fp:   #Pickling
   pickle.dump(v_train, fp) 