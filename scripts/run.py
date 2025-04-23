"""Example script showcasing the usage of ezmm."""

from ezmm import MultimodalSequence, Image

img = Image("in/roses.jpg")

seq = MultimodalSequence("The image",
                         img,
                         "shows two beautiful roses.")

print(seq)
print(seq.__repr__())
