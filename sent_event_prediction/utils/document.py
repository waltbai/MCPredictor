"""Document class for gigaword processed corpus."""
import os
import random
import tarfile

from sent_event_prediction.utils.entity import Entity
from sent_event_prediction.utils.event import Event


def _parse_document(text, tokenized_dir=None):
    """Parse document.

    Refer to G&C16

    :param text: document content.
    :param tokenized_dir: raw text directory
    :return: doc_id, entities, events
    """
    lines = [_.strip() for _ in text.splitlines()]
    # Get doc_id
    doc_id = lines[0]
    # get entity position and event position
    entity_pos = lines.index("Entities:")
    event_pos = lines.index("Events:")
    # Add entities
    entities = []
    for line in lines[entity_pos + 1:event_pos]:
        if line:
            entities.append(Entity.from_text(line))
    # Read raw text if tokenized_dir is given
    if tokenized_dir is not None:
        raw_path = os.path.join(tokenized_dir, doc_id[:14].lower(), doc_id + ".txt")
        with open(raw_path, "r") as f:
            content = f.read().splitlines()
    else:
        content = None
    # Add events
    events = []
    for line in lines[event_pos + 1:]:
        if line:
            cur_event = Event.from_text(line, entities, doc_text=content)
            # Check if current event is duplicate.
            # Since events are sorted by verb_pos,
            # we only need to look back one event.
            # if len(events) == 0 or events[-1]["verb_position"] != cur_event["verb_position"]:
            #     events.append(cur_event)
            # TODO: In old code, duplicate check is invalid,
            #  however the code works well.
            #  Thus we do not check duplicate event temporally.
            events.append(cur_event)
    return doc_id, entities, events


class Document:
    """Document class."""

    def __init__(self, doc_id, entities=None, events=None):
        self.doc_id = doc_id
        self.entities = entities or []
        self.events = events or []

    @classmethod
    def from_text(cls, text, tokenized_dir=None):
        """Initialize Document from text.

        :param text: document content
        :type text: str
        :param tokenized_dir: raw text (tokenized) directory
        :type tokenized_dir: str
        """
        doc_id, entities, events = _parse_document(text, tokenized_dir)
        return cls(doc_id, entities, events)

    def get_chain_for_entity(self, entity, end_pos=None, duplicate=False, stoplist=None):
        """Get chain for specified entity.

        :param entity: protagonist
        :type entity: Entity
        :param end_pos: get events until stop position
        :type end_pos: tuple[int, int] or None
        :param duplicate: whether to obtain duplicate verb
        :param stoplist: stop word list
        :return:
        """
        # Get chain
        result = [event for event in self.events if event.contain(entity)]
        if not duplicate:
            result = [event for idx, event in enumerate(result)
                      if idx == 0 or event["verb_position"] != result[idx-1]["verb_position"]]
        if end_pos is not None:
            result = [event for event in result if event.verb_position <= end_pos]
        if stoplist is not None:
            result = [event for event in result if event.predicate_gr(entity) not in stoplist]
        return result

    def get_chains(self, stoplist=None):
        """Get all (protagonist, chain) pairs.

        :param stoplist: stop verb list
        :type stoplist: None or list[str]
        """
        # Get entities
        result = [(entity, self.get_chain_for_entity(entity))
                  for entity in self.entities]
        if stoplist is not None:
            result = [(entity, [event for event in chain
                                if event.predicate_gr(entity) not in stoplist
                                and event.verb_lemma not in stoplist])
                      for (entity, chain) in result]
        # Filter null chains
        result = [(entity, chain) for (entity, chain) in result
                  if len(chain) > 0]
        return result

    def non_protagonist_entities(self, entity):
        """Return list of non protagonist entities

        :param entity: protagonist
        :type entity: Entity
        :return:
        """
        result = [e for e in self.entities if e is not entity]
        return result


def _parse_question(text, entities, doc_id, tokenized_dir=None):
    """Parse question.

    :param text: question text
    :param entities: entity list of the document
    :param doc_id: document id
    :param tokenized_dir: raw text directory
    :return: entity, context, choices, target
    """
    lines = text.splitlines()
    entity_pos = lines.index("Entity:")
    context_pos = lines.index("Context:")
    choices_pos = lines.index("Choices:")
    target_pos = lines.index("Target:")
    entity = entities[int(lines[entity_pos + 1])]
    # Read raw text if tokenized_dir is given
    if tokenized_dir is not None:
        raw_path = os.path.join(tokenized_dir, doc_id[:14].lower(), doc_id + ".txt")
        with open(raw_path, "r") as f:
            content = f.read().splitlines()
    context = [Event.from_text(e, entities, doc_text=content)
               for e in lines[context_pos + 1:choices_pos - 1] if e]
    choices = [Event.from_text(e, entities)
               for e in lines[choices_pos + 1:target_pos - 1] if e]
    target = int(lines[target_pos + 1])
    # Assign sent to answer
    answer = choices[target]
    answer["sent"] = content[answer["verb_position"][0]]
    return entity, context, choices, target


class TestDocument(Document):
    """TestDocument class."""

    def __init__(self, doc_id, entities=None, events=None,
                 entity=None, context=None, choices=None, target=None):
        super(TestDocument, self).__init__(doc_id, entities, events)
        self.entity = entity
        self.context = context or []
        self.choices = choices or []
        self.target = target

    @classmethod
    def from_text(cls, text, tokenized_dir=None):
        """"Content should first be split into question part and document part.

        :param text: text to be processed.
        :type text: str
        :param tokenized_dir: raw text directory
        :type tokenized_dir: str
        """
        # Split lines
        lines = text.splitlines()
        # Get positions
        document_pos = lines.index("Document:")
        # Parse document part
        document_part = "\n".join(lines[document_pos+1:])
        doc_id, entities, events = _parse_document(document_part, tokenized_dir)
        # Parse question part
        question_part = "\n".join(lines[:document_pos])
        entity, context, choices, target = _parse_question(question_part, entities, doc_id, tokenized_dir)
        return cls(doc_id, entities, events, entity, context, choices, target)

    def get_question(self):
        """Transform entity into head word;
        transform context and choices events into predicate grammar multichain_role like ones.
        """
        entity = self.entity.get_head_word()
        context = [event.predicate_gr(self.entity) for event in self.context]
        choices = [event.predicate_gr(self.entity) for event in self.choices]
        target = self.target
        return entity, context, choices, target


def document_iterator(corp_dir,
                      tokenized_dir=None,
                      file_type="tar",
                      doc_type="train",
                      shuffle=False):
    """Iterator of documents."""
    # Check file_type
    assert file_type in ["tar", "txt"], "Only accept tar/txt as file_type!"
    # Check doc_type
    assert doc_type in ["train", "eval"], "Only accept train/eval as doc_type!"
    # Read file_list
    fn_list = os.listdir(corp_dir)
    if shuffle:
        random.shuffle(fn_list)
    fn_list = [fn for fn in fn_list if fn.endswith(file_type)]
    if file_type == "txt":
        for fn in fn_list:
            fpath = os.path.join(corp_dir, fn)
            with open(fpath, "r") as f:
                content = f.read()
            if doc_type == "train":
                yield Document.from_text(content, tokenized_dir)
            else:   # doc_type == "eval"
                yield TestDocument.from_text(content, tokenized_dir)
    else:   # file_type == "tar"
        for fn in fn_list:
            fpath = os.path.join(corp_dir, fn)
            with tarfile.open(fpath, "r") as f:
                members = f.getmembers()
                if shuffle:
                    random.shuffle(members)
                for member in members:
                    content = f.extractfile(member).read().decode("utf-8")
                    if doc_type == "train":
                        yield Document.from_text(content, tokenized_dir)
                    else:
                        yield TestDocument.from_text(content, tokenized_dir)


__all__ = ["Document", "TestDocument", "document_iterator"]
