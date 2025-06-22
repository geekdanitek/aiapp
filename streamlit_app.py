# ---------------------------------------------------------------
# streamlit_app.py  –  Conditional VAE digit generator
# ---------------------------------------------------------------
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

LATENT_DIM = 20
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

# ───────────────────────── Model definition (same as training)
class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1  = nn.Linear(28*28 + 10, 400)
        self.fc21 = nn.Linear(400, LATENT_DIM)
        self.fc22 = nn.Linear(400, LATENT_DIM)
        self.fc3  = nn.Linear(LATENT_DIM + 10, 400)
        self.fc4  = nn.Linear(400, 28*28)

    def decode(self, z, y_onehot):
        h = F.relu(self.fc3(torch.cat([z, y_onehot], dim=1)))
        return torch.sigmoid(self.fc4(h))

# ───────────────────────── Load trained weights
model = CVAE().to(DEVICE)
model.load_state_dict(torch.load('cvae_mnist.pth', map_location=DEVICE))
model.eval()

# ───────────────────────── Streamlit UI
st.title('Handwritten Digit Image Generator')
st.caption('Pick a digit and generate five MNIST-like samples using a'
           ' Conditional VAE trained from scratch.')

digit = st.selectbox('Choose a digit (0-9):', list(range(10)), index=0)

if st.button('Generate Images'):
    with torch.no_grad():
        # Create five latent vectors conditioned on the chosen digit
        y     = torch.tensor([digit] * 5, device=DEVICE)
        y_1h  = F.one_hot(y, 10).float()
        z     = torch.randn(5, LATENT_DIM, device=DEVICE)
        imgs  = model.decode(z, y_1h).cpu().view(-1, 1, 28, 28)  # (5, 1, 28, 28)

    # Build a 1×5 grid and convert to uint8
    grid = vutils.make_grid(imgs, nrow=5, pad_value=1)          # (C, H, W)
    grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype('uint8')

    st.image(grid_np, caption=f'Five samples of digit {digit}',
             width=500, clamp=True)
