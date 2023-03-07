import sys
import os
import torch
import torchvision
import matplotlib.pyplot as plt

# Récupération de l'argument entré par l'utilisateur
if len(sys.argv) != 2:
    print("Usage: python3 main.py <number>")
    sys.exit(1)

try:
    number = int(sys.argv[1])
    if number < 0 or number > 9:
        raise ValueError()
except ValueError:
    print("Le nombre doit être compris entre 0 et 9")
    sys.exit(1)

# Génération de l'image
out = torch.randn((3, 64, 64))
grid_img = torchvision.utils.make_grid(out).cpu()

# Enregistrement de l'image au format PDF
filename = "generation_number_{}.pdf".format(number)
filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
plt.figure()
plt.imshow(grid_img.permute(1, 2, 0))
plt.axis('off')
plt.savefig(filepath, bbox_inches='tight', pad_inches=0)

print("L'image a été enregistrée sous le nom", filename)
