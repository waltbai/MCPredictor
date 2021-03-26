"""Event class for gigaword processed document."""
import re

from multichain.utils.common import unescape
from multichain.utils.entity import Entity

event_re = re.compile(r'(?P<verb>[^/]*) / (?P<verb_lemma>[^/]*) / '
                      r'verb_pos=\((?P<sentence_num>\d+),(?P<word_index>\d+)\) / '
                      r'type=(?P<type>[^/]*) / subj=(?P<subj>[^/]*) / obj=(?P<obj>[^/]*) / '
                      r'iobj=(?P<iobj>[^/]*)')


HEAD_MODE = "h"
MENTION_MODE = "m"
GLOBAL_MODE = "g"


def find_entity_by_id(s, entity_list):
    """Return entity according to eid.
    This function helps to reduce memory cost,
    since each event saves pointer of entity instead of a string.

    :param s: could be a word or entity id
    :type s: str
    :param entity_list: all mentioned entities in document
    :type entity_list: list[Entity]
    :return: str or Entity
    """
    if s.startswith("entity-"):
        return entity_list[int(s[7:])]
    else:
        return unescape(s)


class Event(dict):
    """Event class."""
    def __init__(self, **kwargs):
        for key in ["subject", "object", "iobject"]:
            if isinstance(kwargs[key], dict) and not isinstance(kwargs[key], Entity):
                kwargs[key] = Entity(**kwargs[key])
        super(Event, self).__init__(**kwargs)

    def __getattr__(self, item):
        return self[item]

    def __repr__(self):
        return "[{verb_lemma:}:" \
               "{subject:}," \
               "{object:}," \
               "{iobject:}]".format(**self)

    def contain(self, argument):
        """Check if the event contains the argument."""
        if argument == "None":
            return False
        else:
            return self["subject"] == argument or \
                   self["object"] == argument or \
                   self["iobject"] == argument

    def find_role(self, argument):
        """Find the role of the argument."""
        if argument == self["subject"]:
            return "subj"
        elif argument == self["object"]:
            return "obj"
        elif argument == self["iobject"] and self["iobject"] != "None":
            return "prep_{}".format(self["iobject_prep"])
        else:
            return "None"

    def predicate_gr(self, argument):
        """Convert event representation to predicate grammar role like."""
        return "{}:{}".format(self["verb_lemma"], self.find_role(argument))

    def tuple(self, protagonist=None, mode=HEAD_MODE, last_verb_pos=None):
        """Convert event to tuple.

        If protagonist is null, return quadruple (verb, subj, obj, iobj),
        else return quintuple (verb, subj, obj, iobj, role).

        If last verb position is given, we will find all mentions before the last_verb_pos
        """
        verb_position = self["verb_position"]
        t = (self["verb_lemma"], )
        for role in ["subject", "object", "iobject"]:
            if isinstance(self[role], Entity):
                if mode == HEAD_MODE:
                    t = t + (self[role].head, )
                elif mode == MENTION_MODE:
                    t = t + (self[role].find_mention_by_pos(verb_position).replace(" ", "_"), )
                else:   # GLOBAL_MODE
                    if last_verb_pos is not None:
                        t = t + ("##".join(self[role].find_mentions_by_pos(last_verb_pos)).replace(" ", "_"), )
                    else:
                        t = t + ("##".join(self[role].find_mentions_by_pos(verb_position)).replace(" ", "_"), )
            else:
                t = t + (self[role], )
        if protagonist is not None:
            t = t + (self.find_role(protagonist), )
        return t

    def get_entities(self):
        """Get entities."""
        entities = []
        for key in ["subject", "object", "iobject"]:
            if isinstance(self[key], Entity):
                entities.append(self[key])
        return entities

    def replace_argument(self, __old, __new):
        """Replace an argument with a new one."""
        for key in ["subject", "object", "iobject"]:
            if self[key] == __old:
                self[key] = __new

    @classmethod
    def from_text(cls, text, entities):
        """Construct Event object from text.

        :param text: text to be parsed.
        :type text: str
        :param entities: entity list from document
        :type entities: list[Entity]
        """
        result = event_re.match(text)
        groups = result.groupdict()
        # Get verb infos
        verb = groups["verb"]
        verb_lemma = groups["verb_lemma"]
        verb_position = (int(groups["sentence_num"]), int(groups["word_index"]))
        type = groups["type"]
        # Get subject
        subject = find_entity_by_id(groups["subj"], entities)
        # Get object
        object = find_entity_by_id(groups["obj"], entities)
        # Get indirect object
        if groups["iobj"] == "None":
            iobject_prep = "None"
            iobject = "None"
        else:
            parts = groups["iobj"].split(",")
            iobject_prep = parts[0]
            iobject = find_entity_by_id(parts[1], entities)
        return cls(verb=verb,
                   verb_lemma=verb_lemma,
                   verb_position=verb_position,
                   type=type,
                   subject=subject,
                   object=object,
                   iobject_prep=iobject_prep,
                   iobject=iobject)


__all__ = ["Event"]
