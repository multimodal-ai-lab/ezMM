from ezmm import Image, MultimodalSequence, Video


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


def test_nested_sequence():
    img = Image("in/roses.jpg")
    vid = Video("in/mountains.mp4")
    string = "The previous media show things from nature."
    seq = MultimodalSequence(img, vid, string)
    seq_nested = MultimodalSequence("The sequence is nested.", seq)
    print(seq_nested)
    assert img in seq_nested
    assert vid in seq_nested
    assert seq_nested.images == [img]
    assert seq_nested.videos == [vid]
    assert len(seq_nested) == 4
    assert seq_nested[1] == img
    assert seq_nested[2] == vid
    assert seq_nested[3] == string


def test_wildly_nested_sequence():
    img = Image("in/roses.jpg")
    vid = Video("in/mountains.mp4")
    seq1 = MultimodalSequence("The image", img, "shows two beautiful roses.")
    seq2 = MultimodalSequence("The video", vid, "shows a nice mountain view.")
    seq3 = MultimodalSequence([seq1, seq2, "Nature is beautiful.", ["Double nested", img]], "Here is another string.")
    print(seq3)
    assert len(seq3) == 10
    assert seq3.images == [img, img]
    assert seq3.videos == [vid]
    assert seq3[1] == img
    assert seq3[4] == vid


def _assert_flat(seq: MultimodalSequence):
    assert all(isinstance(el, (str, Image, Video)) for el in seq.data)


def test_text():
    img = Image("in/roses.jpg")
    seq = MultimodalSequence("The image", img, "shows two beautiful roses.")
    assert seq.text == "The image shows two beautiful roses."
    assert MultimodalSequence(img).text == ""
    assert MultimodalSequence(f"Look: {img.reference} nice").text == "Look: nice"


def test_append():
    img = Image("in/roses.jpg")
    seq = MultimodalSequence("Hello")
    seq.append(img, "world")
    seq.append(MultimodalSequence("nested", Video("in/mountains.mp4")))
    seq.append(None)
    assert len(seq) == 5
    assert seq[1] is img
    _assert_flat(seq)


def test_append_resolves_references():
    img = Image("in/roses.jpg")
    seq = MultimodalSequence()
    seq.append(f"The image {img.reference} is nice.")
    assert seq.images == [img]


def test_extend():
    img = Image("in/roses.jpg")
    seq = MultimodalSequence("A")
    seq.extend(["B", img, ["C", MultimodalSequence("D")]])
    assert seq.data == ["A", "B", img, "C", "D"]


def test_insert():
    img = Image("in/roses.jpg")
    seq = MultimodalSequence("A", "C")
    seq.insert(1, MultimodalSequence("B", img))
    assert seq.data == ["A", "B", img, "C"]
    seq.insert(-1, "X")
    assert seq.data == ["A", "B", img, "X", "C"]
    seq.insert(0, "Start")
    assert seq[0] == "Start"
    _assert_flat(seq)


def test_setitem():
    img = Image("in/roses.jpg")
    seq = MultimodalSequence("A", "B", "C")
    seq[1] = img
    assert seq.data == ["A", img, "C"]
    seq[-1] = MultimodalSequence("X", "Y")  # Gets flattened into the sequence
    assert seq.data == ["A", img, "X", "Y"]
    seq[0:2] = "Z"
    assert seq.data == ["Z", "X", "Y"]
    _assert_flat(seq)


def test_delete_remove_pop():
    img = Image("in/roses.jpg")
    seq = MultimodalSequence("A", img, "B", "C")
    del seq[0]
    assert seq.data == [img, "B", "C"]
    seq.remove(img)
    assert seq.data == ["B", "C"]
    assert seq.pop() == "C"
    seq.clear()
    assert not seq


def test_add():
    img = Image("in/roses.jpg")
    seq1 = MultimodalSequence("A", img)
    seq2 = MultimodalSequence("B")
    combined = seq1 + seq2
    assert combined.data == ["A", img, "B"]
    assert seq1.data == ["A", img]  # Unchanged
    assert ("Start" + seq2).data == ["Start", "B"]
    assert (seq2 + img).data == ["B", img]


def test_iadd():
    img = Image("in/roses.jpg")
    seq = MultimodalSequence("A")
    original = seq
    seq += MultimodalSequence(img, "B")
    seq += "C"
    assert seq is original
    assert seq.data == ["A", img, "B", "C"]
    _assert_flat(seq)


def test_self_append():
    seq = MultimodalSequence("A", "B")
    seq += seq
    assert seq.data == ["A", "B", "A", "B"]


def test_copy():
    seq = MultimodalSequence("A")
    copied = seq.copy()
    copied.append("B")
    assert seq.data == ["A"]


# def test_render():
#     seq = MultimodalSequence(
#         "The image",
#         Image("in/roses.jpg"),
#         "shows two beautiful roses while the video",
#         Video("in/mountains.mp4"),
#         "shows a nice mountain view."
#     )
#     seq.render()
