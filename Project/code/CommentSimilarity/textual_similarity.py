import difflib


# Define a function to detect textual clones
def detect_textual_clones(text1, text2):
    # Remove white space
    # The split() method splits a string into a list.
    #
    # You can specify the separator, default separator is any whitespace.
    # whitespace: 包括空格、换行(\n)、制表符(\t)
    stripped_text1 = ' '.join(text1.split())
    stripped_text2 = ' '.join(text2.split())

    # Calculate the similarity ratio using the SequenceMatcher class from the difflib module
    ratio = difflib.SequenceMatcher(None, stripped_text1, stripped_text2).ratio()

    # Return True if the similarity ratio is greater than or equal to the threshold
    return ratio == 1


if __name__ == "__main__":
    text1 = 'Returns the name of the person\n hh.'
    text2 = 'Returns the age of the person.'
    detect_textual_clones(text1,text2)
