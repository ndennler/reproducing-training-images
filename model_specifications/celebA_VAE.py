import torch
import torch.nn as nn

# ------------------------- ENCODER ARCHITECTURE -----------------------------

class Encoder(nn.Module):
  def __init__(self, nz):
    super().__init__()

    self.net = nn.Sequential(
        nn.Conv2d(3, 32, 7, stride=4),
        nn.LeakyReLU(),

        nn.Conv2d(32, 64, 7, stride=4),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),

        nn.Conv2d(64,128,5, stride=2),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),

        nn.Conv2d(128, 256, 5, stride=3),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(),

        nn.Conv2d(256,512,4, stride=2),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(),

        nn.Flatten(),
        nn.Linear(512, nz)
    )
  
  def forward(self, x):
    return self.net(x)

# ------------------------- DECODER ARCHITECTURE -----------------------------

class Decoder(nn.Module):
  def __init__(self, nz):
    super().__init__()
    self.map = nn.Linear(nz, 512)   # for initial Linear layer

    self.net = nn.Sequential(
        nn.BatchNorm2d(512),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(512, 256, 4, stride=2, padding=0, output_padding=0),

        nn.BatchNorm2d(256),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(256, 128, 5, stride=3, padding=0, output_padding=0),

        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(128,64,5, stride=3, padding=1, output_padding=0),

        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(64,32,7, stride=4, padding=0, output_padding=0),

        nn.LeakyReLU(),
        nn.ConvTranspose2d(32,32,7,stride=3, padding=1, output_padding=0),

        nn.Conv2d(32, out_channels= 3, kernel_size= 6, padding= 1),
        nn.Tanh()
    )
  
  def forward(self, x):
    return self.net(self.map(x).reshape(-1, 512, 1, 1))

  
# -------------------- PUTTING ENCODER AND DECODER TOGETHER --------------------
class AutoEncoder(nn.Module):
  def __init__(self, nz):
    super().__init__()
    self.encoder = Encoder(nz)
    self.decoder = Decoder(nz)

  def forward(self, x):
    return {'rec': self.decoder(self.encoder(x))}

  def reconstruct(self, x):
    return self.decoder(self.encoder(x))

  def loss(self, x, outputs):
    # compute reconstruction loss
    rec_loss = nn.MSELoss()(x, outputs['rec'])
    return rec_loss, {'rec_loss': rec_loss, 'kl_loss': None} 



if __name__ == '__main__':
    sample = torch.randn((32,3,512,512))

    encoder = Encoder(100)
    output = encoder(sample)
    print(f"output of encoder is {output.shape}")

    decoder = Decoder(100)
    output = decoder(output)
    print(f"output of decoder is {output.shape}")