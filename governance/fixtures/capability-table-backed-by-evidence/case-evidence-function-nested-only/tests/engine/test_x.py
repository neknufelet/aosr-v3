"""樣本考卷：同名函式藏在別的函式裡。"""


def test_other() -> None:
    def test_x_validated() -> None:
        assert True

    test_x_validated()
