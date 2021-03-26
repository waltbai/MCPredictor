"""Document class for gigaword processed corpus."""
from multichain.utils.entity import Entity
from multichain.utils.event import Event


def _parse_document(text):
    """Parse document.

    Refer to G&C16

    :param text: document content.
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
    # Add events
    events = []
    for line in lines[event_pos + 1:]:
        if line:
            cur_event = Event.from_text(line, entities)
            # Check if current event is duplicate.
            # Since events are sorted by verb_pos,
            # we only need to look back one event.
            # if len(events) == 0 or not events[-1] == cur_event:
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
    def from_text(cls, text):
        """Initialize Document from text.

        :param text: document content
        :type text: str
        """
        doc_id, entities, events = _parse_document(text)
        return cls(doc_id, entities, events)

    def get_chain_for_entity(self, entity, end_pos=None):
        """Get chain for specified entity.

        :param entity: protagonist
        :type entity: Entity
        :param end_pos: get events until stop position
        :type end_pos: tuple[int, int] or None
        :return:
        """
        # Get chain
        result = [event for event in self.events if event.contain(entity)]
        if end_pos is not None:
            result = [event for event in result if event.verb_position <= end_pos]
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


def _parse_question(text, entities):
    """Parse question.

    :param text: question text
    :param entities: entity list of the document
    :return: entity, context, choices, target
    """
    lines = text.splitlines()
    entity_pos = lines.index("Entity:")
    context_pos = lines.index("Context:")
    choices_pos = lines.index("Choices:")
    target_pos = lines.index("Target:")
    entity = entities[int(lines[entity_pos + 1])]
    context = [Event.from_text(e, entities)
               for e in lines[context_pos + 1:choices_pos - 1] if e]
    choices = [Event.from_text(e, entities)
               for e in lines[choices_pos + 1:target_pos - 1] if e]
    target = int(lines[target_pos + 1])
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
    def from_text(cls, text):
        """"Content should first be split into question part and document part.

        :param text: text to be processed.
        :type text: str
        """
        # Split lines
        lines = text.splitlines()
        # Get positions
        document_pos = lines.index("Document:")
        # Parse document part
        document_part = "\n".join(lines[document_pos+1:])
        doc_id, entities, events = _parse_document(document_part)
        # Parse question part
        question_part = "\n".join(lines[:document_pos])
        entity, context, choices, target = _parse_question(question_part, entities)
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
