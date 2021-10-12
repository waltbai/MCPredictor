"""Event class for gigaword processed document."""
import re

from mcpredictor.utils.common import unescape
from mcpredictor.utils.entity import Entity, transform_entity

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

    @property
    def filter(self):
        """In filtered format, an event is represented as follows:

        event: {
            sent: str,
            verb_lemma: str,
            verb_position: [int, int],
            subject: {head: str, mention: str, entity: bool}
            object: {head: str, mention: str, entity: bool}
            iobject: {head: str, mention: str, entity: bool}
            iobject_prep: str
        }
        """
        item = {
            "verb_lemma": self["verb_lemma"],
            "verb_position": self["verb_position"],
            "subject": transform_entity(self["subject"], self["verb_position"]),
            "object": transform_entity(self["object"], self["verb_position"]),
            "iobject": transform_entity(self["iobject"], self["verb_position"]),
            "iobject_prep": self["iobject_prep"]
        }
        # For negative events, they have no corresponding sentences.
        if "sent" in self:
            item["sent"] = self["sent"]
        return item

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

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

    def find_role(self, argument, stoplist=None):
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

    def get_words(self):
        """Get words."""
        ret_val = [self["verb"]]
        for role in ["subject", "object", "iobject"]:
            if isinstance(self[role], Entity):
                ret_val.append(self[role]["head"])
            elif self[role] != "None":
                ret_val.append(self[role])
        return ret_val

    def get_entities(self):
        """Get entities."""
        entities = []
        for key in ["subject", "object", "iobject"]:
            if isinstance(self[key], Entity):
                entities.append(self[key])
        return entities

    def replace_mention(self, __old: str, __new: str):
        """Replace mention in sentence."""
        # After replacement, verb position will changed
        token_index = self["verb_position"][1]
        sent = self["sent"].split()
        before_verb = " ".join(sent[:token_index])
        verb = sent[token_index]
        after_verb = " ".join(sent[token_index+1:])
        before_verb = before_verb.replace(__old, __new).split()
        after_verb = after_verb.replace(__old, __new).split()
        new_sent = before_verb + [verb] + after_verb
        token_index = len(before_verb)
        self["sent"] = " ".join(new_sent)
        self["verb_position"] = [self["verb_position"][0], token_index]

    def tagged_sent(self, role, mask_list=None):
        """Tag verb role of the sentence."""
        sent = self["sent"].lower().split()
        pos = self["pos"]
        if role not in ["subj", "obj"]:
            role = "iobj"
        sent_id = self["verb_position"][0]
        verb_index = self["verb_position"][1]
        token_list = []
        pos_list = []
        # Use "O" to represent control tokens
        for index, token in enumerate(sent):
            if index == verb_index:
                token_list.extend(["[{}]".format(role), sent[verb_index], "[{}]".format(role)])
                pos_list.extend(["O", pos[verb_index], "O"])
            elif mask_list is not None and token in mask_list:
                token_list.append("[UNK]")
                pos_list.append("O")
            else:
                token_list.append(token)
                pos_list.append(pos[index])
        # Extract sent
        return token_list, pos_list

    def replace_argument(self, __old, __new):
        """Replace an argument with a new one."""
        for key in ["subject", "object", "iobject"]:
            if self[key] == __old:
                self[key] = __new

    @classmethod
    def from_text(cls, text, entities, doc_text=None, doc_pos=None):
        """Construct Event object from text.

        :param text: text to be parsed.
        :type text: str
        :param entities: entity list from document
        :type entities: list[Entity]
        :param doc_text: document text
        :type doc_text: list[str]
        :param doc_pos: document pos
        :type doc_pos: list[list[str]]
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
        # Get sentence
        if doc_text is not None:
            sent = doc_text[verb_position[0]]
        else:
            sent = None
        if doc_pos is not None:
            pos = doc_pos[verb_position[0]]
        else:
            pos = None
        return cls(verb=verb,
                   verb_lemma=verb_lemma,
                   verb_position=verb_position,
                   # type=type,
                   subject=subject,
                   object=object,
                   iobject_prep=iobject_prep,
                   iobject=iobject,
                   sent=sent,
                   pos=pos
                   )


__all__ = ["Event", "transform_entity"]
