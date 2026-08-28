from othello_rl.utils.progress import _NullBar, make_progress


def test_make_progress_disabled_returns_nullbar():
    bar = make_progress(100, enabled=False)
    assert isinstance(bar, _NullBar)
    bar.update(5)
    bar.set_description("x")
    bar.set_postfix({"a": 1})
    bar.close()
    assert bar.n == 5


def test_make_progress_auto_off_when_not_tty(capsys):
    # pytest captures stdout, so isatty() is False -> NullBar
    bar = make_progress(10, enabled="auto")
    assert isinstance(bar, _NullBar)


def test_nullbar_is_context_manager():
    with make_progress(3, enabled=False) as bar:
        bar.update(1)
    assert bar.n == 1


def test_make_progress_enabled_returns_working_bar():
    bar = make_progress(5, enabled=True)
    # tqdm is installed in this environment; just make sure the surface works
    bar.update(1)
    bar.set_postfix({"loss": 0.1})
    bar.close()
