from ezmm import MultimodalSequence, Image, Video


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


def test_empty():
    seq0 = MultimodalSequence()
    seq1 = MultimodalSequence("")
    seq2 = MultimodalSequence(" ")
    seq3 = MultimodalSequence([])
    seq4 = MultimodalSequence(None)
    seq5 = MultimodalSequence([None])
    assert seq0 == seq1 == seq2 == seq3 == seq4 == seq5


def test_bool_false():
    seq0 = MultimodalSequence()
    seq1 = MultimodalSequence(None)
    seq2 = MultimodalSequence([])
    seq3 = MultimodalSequence("")
    seq4 = MultimodalSequence(None, None)
    assert not seq0
    assert not seq1
    assert not seq2
    assert not seq3
    assert not seq4


def test_bool_true():
    seq0 = MultimodalSequence("This is just some text.")
    seq1 = MultimodalSequence("The image", Image("in/roses.jpg"), "shows two beautiful roses.")
    assert seq0
    assert seq1


def test_sequence_resolve():
    img1 = Image("in/roses.jpg")
    img2 = Image("in/garden.jpg")
    string = f"The image {img1.reference} shows two beautiful roses. The image {img2.reference} shows a nice garden."
    seq = MultimodalSequence(string)
    assert img1 in seq
    assert img2 in seq
    assert seq.images == [img1, img2]


# def test_render():
#     seq = MultimodalSequence(
#         "The image",
#         Image("in/roses.jpg"),
#         "shows two beautiful roses while the video",
#         Video("in/mountains.mp4"),
#         "shows a nice mountain view."
#     )
#     seq.render()
