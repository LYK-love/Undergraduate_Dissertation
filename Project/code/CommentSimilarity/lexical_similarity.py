import fuzzywuzzy.fuzz as fuzz

def detect_lexical_clones(doc1, doc2, threshold=98):
    """
    Check if two Javadoc comments are lexical clones.

    Parameters:
    - doc1 (str): The first Javadoc comment.
    - doc2 (str): The second Javadoc comment.
    - threshold (int): The minimum similarity score to consider the comments a clone.

    Returns:
    - True if the comments are lexical clones, False otherwise.
    """
    # Remove any trailing or leading whitespace from the comments
    doc1 = doc1.strip()
    doc2 = doc2.strip()

    # Compute the Levenshtein distance between the comments
    distance = fuzz.ratio(doc1, doc2)

    # Check if the distance is below the threshold
    if distance >= threshold:
        return True
    else:
        return False
