"""Common functions used for several modules."""


def unescape(text, space_slashes=False):
    """Function copy from G&C16.

    :param text: text to be unescaped
    :type text: str
    :param space_slashes: if add space when convert single-slash
    :type space_slashes: bool
    :return: the unescaped text
    """
    # Reverse various substitutions made on output
    text = text.replace("@semicolon@", ";")
    text = text.replace("@comma@", ",")
    if space_slashes:
        text = text.replace("@slash@", " / ")
        text = text.replace("@slashes@", " // ")
    else:
        text = text.replace("@slash@", "/")
    return text
