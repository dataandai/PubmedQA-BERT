from pymed import PubMed
import backend.squad as squad


def parse_entity(doc):

    pubmed = PubMed(tool="BERT_QA_FOR_PUBMED", email="feldmann.adam@pte.hu")
    texts = []

# ha a címben szerepel az azonosított entitás, akkor max 50 absztraktot beolvas és összefűz, majd visszatér vele. 
    for ent in doc.ents:
        try:
            results = pubmed.query(ent.text + "[Title]", max_results=50)

            for article in results:
                if article.abstract != None:
                    texts.append(article.abstract)

        except IndexError:
            return texts

    return texts
