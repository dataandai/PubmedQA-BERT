import spacy
import scispacy

nlp = spacy.load("en_ner_bc5cdr_md")

def recognize_entities(text):

    doc = nlp(text)

    return doc

