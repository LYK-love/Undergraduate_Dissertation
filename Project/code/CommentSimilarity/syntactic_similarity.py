import nltk
from nltk.parse import CoreNLPParser
from nltk.tree import Tree
from difflib import SequenceMatcher


def detect_syntactic_clones(text1, text2, threshold=0.9):
    # Set up Stanford CoreNLP server and parser
    if text1 == '' or text2 == '':
        return False

    parser = CoreNLPParser(url='http://localhost:9000')
    parser.parser_annotator = 'parse'

    # Tokenize and parse each text
    trees1 = list(parser.raw_parse(text1))
    trees2 = list(parser.raw_parse(text2))

    # Compute syntactic similarity score for each pair of trees
    scores = []
    for tree1 in trees1:
        for tree2 in trees2:
            score = SequenceMatcher(None, str(tree1), str(tree2)).ratio()
            scores.append(score)

    # Compute average syntactic similarity score
    avg_score = sum(scores) / len(scores)

    # Compare average score to threshold and return result
    if avg_score > threshold:
        return True
    else:
        return False


# # Example usage
# text1 = "The cat sat on the mat."
# text2 = "The dog slept on the rug."
# threshold = 0.8
#
# if detect_syntactic_clones(text1, text2, threshold):
#     print("The texts are syntactically similar")
# else:
#     print("The texts are not syntactically similar")
