import argparse
import spacy
import re
from pdfminer.high_level import extract_text

NER = spacy.load("en_core_web_trf")
NLP = spacy.load('en_core_web_sm')


def parse(input: str) -> str:
    """
    Extracts text from pdf
    :param input:
    :return:
    """
    output = extract_text(input)
    return output


def extract_names(text: str):
    """
    Reads text and returns the name of applicant.
    :param text:
    :return:
    """

    text1 = NER(text)
    for word in text1.ents:
        # First name is most likely to be the applicant's name, since people usually put their names early on in the
        # resume
        if word.label_ == "PERSON":
            return word


def extract_phone_number(text: str):
    """
    Reads text and returns the applicant's phone number.
    :param text:
    :return:
    """
    PHONE_REG = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')

    phone = re.findall(PHONE_REG, text)

    return [re.sub(r'\D', '', num) for num in phone]


def extract_email(text: str):
    """
    Reads text and returns the appplciant's email.
    :param text:
    :return:
    """
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return r.findall(text)


def extract_skills(text):
    nlp_text = NLP(text)

    # removing stop words and implementing word tokenization
    tokens = [token.text for token in nlp_text if not token.is_stop]

    skills = ["machine learning",
              "deep learning",
              "nlp",
              "natural language processing",
              "mysql",
              "sql",
              "django",
              "computer vision",
              "tensorflow",
              "opencv",
              "mongodb",
              "sql",
              "postgres",
              "postgresql"
              "artificial intelligence",
              "ai",
              "flask",
              "robotics",
              "data structures",
              "python",
              "c",
              "c#"
              "c++",
              "matlab",
              "css",
              "html",
              "github",
              "php",
              "java"
              "react"]

    skillset = []

    # check for one-grams (example: python)
    for token in tokens:
        if token.lower() in skills:
            skillset.append(token)

    # check for bi-grams and tri-grams (example: machine learning)
    for token in nlp_text.noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)

    return [i.capitalize() for i in set([i.lower() for i in skillset])]


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
    number = extract_phone_number(text)
    email = extract_email(text)
    skills = extract_skills(text)
    print(name, number[0], email[0], skills)
