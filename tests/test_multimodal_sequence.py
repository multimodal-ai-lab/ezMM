from ezmm import MultimodalSequence, Image


def test_multimodal_sequence():
    seq = MultimodalSequence("This is just some text.")
    print(seq)

    img = Image("in/roses.jpg")
    seq = MultimodalSequence("The image", img, "shows two beautiful roses.")
    print(seq)


def test_seq_equality():
    img = Image("in/roses.jpg")
    seq1 = MultimodalSequence("The image", img, "shows two beautiful roses.")
    seq2 = MultimodalSequence(["The image", img, "shows two beautiful roses."])
    assert seq1 == seq2
    assert seq1 is not seq2


def test_seq_images():
    img1 = Image("in/roses.jpg")
    img2 = Image("in/garden.jpg")
    seq = MultimodalSequence("The images", img1, img2, "show two beautiful roses and a garden.")
    images = seq.images
    print(images)
