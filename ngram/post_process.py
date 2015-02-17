import re

TWO_PIECERS = 'ఘపఫషసహ'
aksh_pattern = re.compile(r"([ఁ-ఔృౄ])|( )|(([క-హ]్)*[క-హ][ా-ూె-ౌ])|"
                          r"(([క-హ]్)*[క-హ](?![ా-ూె-్]))|(([క-హ]్)+(?=\s))")


def post_process(content):
    rules = [
        # ౦) ర వత్తులు కుదుర్చు
        (r'్ర౧', r'్ర'),

        # ౧) Attach ఘపఫషసహ with ✓ ి  ీ  ె  ే  ్
        (r'([ిీెే్])([ఘపఫషసహ])', r'\2\1'),
        (r'([✓])([ఘపఫషసహ])', r'\2'),

        # ౨) కె, ై  ఉంటే వాటిని కై గా మార్చు
        (r'([క-హ])ెై', r'\1ై'),

        # ౩) ఘొ ఘో లు
        (r'ెఘా', r'ఘో'),
        (r'ెఘు', r'ఘొ'),

        # ౪) ఏఎ ని ఏ గా మార్చు
        (r'ఏఎ', r'ఏ'),

        # ౫) సున్నాలు సరిజూఁడు
        (r'([ఁ-ౌ])[0౦]', r'\1ం'),

        # ౬) సంయుక్తాలను అమర్చు
        (r'([ా-ూె-్])(్[క-హ‍])', r'\2\1'),
        (r'([ా-ూె-్])(్[క-హ‍])', r'\2\1'),
    ]

    for find, replace in rules:
        content = re.sub(find, replace, content)

    return content


def impossible(chars):
    """
    Is a given sequence of labels impossible.
    eg:- ఏ followed by anything but an ఎ
    :param chars:
    :return: boolean True if seq is impossible
    """

    if len(chars) < 2:
        return False

    should_follow = [
        ('ఏ', 'ఎ'),
        ('✓ిీెే్', 'ఘపఫషసహ'),
    ]

    for x, y in should_follow:
        if chars[0] in x and not chars[1] in y:
            return True

    return False