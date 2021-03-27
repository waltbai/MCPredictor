"""Entity class for gigaword processed document."""
import os
import re
import string

from sent_event_prediction.utils.mention import Mention


# Punctuation regex
punct_re = re.compile('([%s])+' % re.escape(string.punctuation))
# Pronouns
PRONOUNS = [
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "myself", "yourself", "himself", "herself", "itself", "ourself", "ourselves", "themselves",
    "my", "your", "his", "its", "it's", "our", "their",
    "mine", "yours", "ours", "theirs",
    "this", "that", "those", "these"]
# Load stop word list
with open(os.path.join("data", "english_stopwords.txt"), "r") as f:
    STOPWORDS = f.read().splitlines()


def field_value(field, text):
    """Find value for certain field.

    :param field:
    :type field: str
    :param text:
    :type text: str
    :return:
    """
    parts = text.split("=")
    assert field == parts[0]
    return parts[1]


def filter_words_in_mention(words):
    """Filter stop words and pronouns in mention."""
    return [w for w in words if w not in PRONOUNS and w not in STOPWORDS]


def get_head_word(mentions):
    """Get head word of mentions.

    Copy from G&C16
    """
    entity_head_words = set()
    for mention in mentions:
        # Get head word from mention
        mention_head = mention.get_head_word()
        # Remove punctuation
        mention_head = punct_re.sub(" ", mention_head)
        # Split words
        head_words = mention_head.split()
        # Get rid of words that won't help us: stopwords and pronouns
        head_words = filter_words_in_mention(head_words)
        # Don't use any 1-letter words
        head_words = [w for w in head_words if len(w) > 1]
        # If there are no words left, we can't get a headword from this mention
        # If there are multiple (a minority of cases), use the rightmost,
        # which usually is the headword
        if head_words:
            entity_head_words.add(head_words[-1])
    if len(entity_head_words):
        return list(sorted(entity_head_words))[0]
    else:
        return "None"


class Entity(dict):
    """Entity class."""
    def __init__(self, **kwargs):
        # Convert dict mention to Mention object
        mentions = []
        for mention in kwargs["mentions"]:
            if isinstance(mention, Mention):
                mentions.append(mention)
            else:
                mentions.append(Mention(**mention))
        kwargs["mentions"] = mentions
        # Get head word
        kwargs.setdefault("head", get_head_word(mentions))
        super(Entity, self).__init__(**kwargs)

    def __getattr__(self, item):
        return self[item]

    def __repr__(self):
        return "<{entity_id:}:{head:}>".format(**self)

    def find_mention_by_pos(self, verb_position):
        """Find mention by verb_position."""
        sent_id = verb_position[0]
        mentions = [m["text"] for m in self["mentions"] if m["sentence_num"] == sent_id]
        if len(mentions) == 0:
            return "None"
        else:
            return max(mentions, key=lambda x: len(x))

    def find_longest_mention(self):
        """Find longest mention."""
        mentions = [m["text"] for m in self["mentions"]]
        return max(mentions, key=lambda x: len(x))

    def clear_mentions(self):
        """Clear mentions field in order to save space during storing."""
        self["mentions"] = []

    def get_head_word(self):
        """Return head word."""
        return self["head"]

    @classmethod
    def from_text(cls, text):
        """Construct Entity object from text.

        :param text: text to be parsed.
        """
        # Though parsing can be done by regex,
        # it is not necessary.
        entity_id, text = text.split(":", 1)
        # Find entity_id
        entity_id = "entity-{}".format(entity_id)
        # Find other attributes
        parts = [p.strip() for p in text.split(" // ")]
        category = field_value("category", parts[0])
        gender = field_value("gender", parts[1])
        gender_prob = float(field_value("genderProb", parts[2]))
        number = field_value("number", parts[3])
        number_prob = float(field_value("numberProb", parts[4]))
        # Parse mentions
        mentions = field_value("mentions", parts[5]).split(" / ")
        mentions = [Mention.from_text(m) for m in mentions]
        mentions = [m for m in mentions if m.text]
        # Parse type
        if len(parts) > 6:
            type_ = field_value("type", parts[6])
        else:
            type_ = "misc"
        return cls(entity_id=entity_id,
                   category=category,
                   gender=gender,
                   gender_prob=gender_prob,
                   number=number,
                   number_prob=number_prob,
                   mentions=mentions,
                   type=type_)


def transform_entity(entity, verb_position=None):
    """Transform entity/str into json object."""
    item = {}
    if isinstance(entity, Entity):
        item["head"] = entity["head"]
        item["entity"] = int(entity["entity_id"][7:])
        if verb_position is not None:
            item["mention"] = entity.find_mention_by_pos(verb_position)
        else:
            item["mention"] = entity.find_longest_mention()
    else:
        item["head"] = entity
        item["mention"] = entity
        item["entity"] = -1
    return item


__all__ = ["Entity", "filter_words_in_mention", "transform_entity"]
