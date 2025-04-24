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
    seq3 = MultimodalSequence(f"The image {img.reference} shows two beautiful roses.")
    assert seq1 == seq2
    assert seq1 == seq3
    assert seq1 is not seq2
    assert seq1 is not seq3
    assert seq2 is not seq3


def test_seq_inequality():
    img1 = Image("in/roses.jpg")
    img2 = Image("in/garden.jpg")
    seq1 = MultimodalSequence("The image", img1, "is nice.")
    seq2 = MultimodalSequence("The image", img2, "is nice.")
    assert seq1 != seq2


def test_list_comprehension():
    img = Image("in/roses.jpg")
    seq = MultimodalSequence("The image", img, "shows two beautiful roses.")
    assert seq[0] == "The image"
    assert seq[1] == img
    assert seq[2] == "shows two beautiful roses."
