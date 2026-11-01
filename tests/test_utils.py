"""`--data_args` values must keep the type the user wrote.

`handle_arg_string` used `str.isnumeric()` to decide whether a value was an
integer. That predicate disagrees with `int()` in both directions: "-5" is not
numeric but is an integer, and "½" is numeric but `int()` rejects it. So a
negative integer silently arrived as a float, and a Unicode numeral raised an
unhandled ValueError on the way to `load_dataset`.
"""

import pytest

from sparsify.utils import handle_arg_string, simple_parse_args_string


@pytest.mark.parametrize(
    "text,expected",
    [
        ("5", 5),
        ("0", 0),
        ("-5", -5),
        ("+5", 5),
        ("-0", 0),
    ],
)
def test_integers_stay_integers(text, expected):
    """Negative and signed integers used to come back as floats."""
    result = handle_arg_string(text)
    assert result == expected
    assert isinstance(result, int)


@pytest.mark.parametrize("text", ["3.5", "-3.5", "1e3"])
def test_floats_are_still_floats(text):
    result = handle_arg_string(text)
    assert isinstance(result, float)
    assert result == float(text)


@pytest.mark.parametrize("text", ["½", "²", "Ⅷ"])
def test_unicode_numerals_do_not_raise(text):
    """str.isnumeric() accepts these; int() does not, so this used to crash."""
    assert handle_arg_string(text) == text


@pytest.mark.parametrize(
    "text,expected",
    [("true", True), ("True", True), ("false", False), ("FALSE", False)],
)
def test_booleans(text, expected):
    assert handle_arg_string(text) is expected


def test_plain_strings_pass_through():
    assert handle_arg_string("train") == "train"


def test_parse_args_string_keeps_integer_types():
    """The end-to-end path a `--data_args` value takes."""
    parsed = simple_parse_args_string("num_proc=-1,name=wikitext,ratio=0.5,flag=true")

    assert parsed == {
        "num_proc": -1,
        "name": "wikitext",
        "ratio": 0.5,
        "flag": True,
    }
    assert isinstance(parsed["num_proc"], int)


def test_parse_args_string_empty():
    assert simple_parse_args_string("") == {}
    assert simple_parse_args_string("   ") == {}
