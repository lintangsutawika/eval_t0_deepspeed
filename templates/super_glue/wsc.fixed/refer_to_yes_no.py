import re

def map_fn(ex):

    answer_choices = ["No", "Yes"]

    return [
        (
        "{text}"
        " Does the pronoun \"{span2_text}\" refers to \"{span1_text}\"."
        " Yes or no?"
        ).format(**ex),
        answer_choices[ex["label"]]
    ]