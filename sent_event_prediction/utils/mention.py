"""Mention class for coreferenced entities."""
import re

from multichain.utils.common import unescape


mention_re = re.compile(r"\((?P<char_start>\d+),(?P<char_end>\d+)\);"
                        r"(?P<text>.+);"
                        r"(?P<np_sentence_position>\d+);"
                        r"(?P<np_doc_position>\d+);"
                        r"(?P<nps_in_sentence>\d+);"
                        r"(?P<sentence_num>\d+);"
                        r"\((?P<head_start>\d+),(?P<head_end>\d+)\)")


class Mention(dict):
    """Mention class."""

    def __init__(self, **kwargs):
        """Entity mention."""
        super(Mention, self).__init__(**kwargs)

    def __getattr__(self, item):
        """Get attribute."""
        return self[item]

    def get_head_word(self):
        """Return head word of this mention in lower case."""
        char_start = self["char_span"][0]
        head_start, head_end = self["head_span"]
        return self["text"][head_start-char_start:head_end-char_start].lower()

    @classmethod
    def from_text(cls, text):
        """Construct Mention object from text.

        Refer to G&C16

        :param text: text to be parsed.
        """
        text = unescape(text, space_slashes=True)
        result = mention_re.match(text)
        # Match succeeded.
        groups = result.groupdict()
        # Though only char_span, text, head_span are used,
        # we save all information.
        char_span = (int(groups["char_start"]), int(groups["char_end"]))
        text = groups["text"]
        np_sentence_position = int(groups["np_sentence_position"])
        np_doc_position = int(groups["np_doc_position"])
        nps_in_sentence = int(groups["nps_in_sentence"])
        sentence_num = int(groups["sentence_num"])
        head_span = (int(groups["head_start"]), int(groups["head_end"]))
        return cls(char_span=char_span,
                   text=text,
                   np_sentence_position=np_sentence_position,
                   np_doc_position=np_doc_position,
                   nps_in_sentence=nps_in_sentence,
                   sentence_num=sentence_num,
                   head_span=head_span)


__all__ = ["Mention"]
