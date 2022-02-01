import re
import string

def map_fn(ex):

    ex['passage'] =  re.sub('@highlight', "", ex['passage'])
    ex['query'] =  re.sub('@placeholder', 'BLANK', ex['query'])

    choices = ex['entities']
    alphabet = string.ascii_uppercase

    choice_list = list(zip(alphabet[:len(choices)], choices))
    answer_choices = []
    for alphabet, choice in choice_list:
        if choice in ex['answers']:
            answer_choices.append(alphabet)

    multiple_choice = ",\n ".join([". ".join(i) for i in choice_list])
    ex['multiple_choice'] = multiple_choice

    return [
        (
        "Passage: {passage} \n"
        "Query: {query} \n"
        "According to the passage, Which of these choices does the BLANK in the query refer to?\n "
        "{multiple_choice}"
        ).format(**ex),
        "---".join(answer_choices),
    ]