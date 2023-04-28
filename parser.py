import argparse
import nltk
from nltk.corpus import stopwords
from pdfminer.high_level import extract_text

stop = stopwords.words('english')


def parse(input: str) -> str:
    """
    Extracts text from pdf
    :param input:
    :return:
    """
    output = extract_text(input)
    return output


def ie_preprocess(document):
    document = ' '.join([i for i in document.split() if i not in stop])
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences


def extract_names(text: str):
    """
    Parses input and returns the name
    :param text:
    :return:
    """

    names = []
    sentences = ie_preprocess(text)
    for tagged_sentence in sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                if chunk.label() == 'PERSON':
                    names.append(' '.join([c[0] for c in chunk]))
    return names


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="One input file."
    )
    args = parser.parse_args()

    text = parse(args.inputfile)
    name = extract_names(text)
    print(text)
    print(name)