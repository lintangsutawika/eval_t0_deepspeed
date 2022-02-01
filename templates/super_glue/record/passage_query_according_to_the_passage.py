import re

def map_fn(ex):

    ex['passage'] =  re.sub('@highlight', "", ex['passage'])
    ex['query'] =  re.sub('@placeholder', 'BLANK', ex['query'])

    return [
        (
        "Passage: {passage} \n"
        "Query: {query} \n"
        "According to the passage, what does the BLANK in the query refer to?"
        ).format(**ex),
        "</>".join(ex["answers"]),
    ]