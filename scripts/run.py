"""Example script showcasing the usage of ezmm."""

from ezmm import MultimodalSequence, Image

img1 = Image("in/roses.jpg")
img2 = Image("in/garden.jpg")

seq = MultimodalSequence("The image", img1, "shows two beautiful roses while",
                         img2, "shows a nice garden with many flowers.")

print(seq)
print(seq.__repr__())
