import re

TWO_PIECERS = 'ఘపఫషసహ'
aksh_pattern = re.compile(r"([ఁ-ఔృౄ])|( )|(([క-హ]్)*[క-హ][ా-ూె-ౌ])|"
                          r"(([క-హ]్)*[క-హ](?![ా-ూె-్]))|(([క-హ]్)+(?=\s))")

def post_process(content):
    """
    TODO:
        ౦) ర వత్తులు కుదుర్చు
        ౧) Attach ఘపఫషసహ with ✓ ి  ీ  ె  ే  ్
        ౨) కె, ై  ఉంటే వాటిని కై గా మార్చు
        ౩) ఘొ ఘో లు
        ౪) ఏఎ ని ఏ గా మార్చు
        ౫) సున్నాలు సరిజూఁడు
        ౬) సంయుక్తాలను అమర్చు
    """
    content = re.sub(r'్ర౧', r'్ర', content)
    content = re.sub(r'([ిీెే])([ఘపఫషసహ])', r'\2\1', content)
    content = re.sub(r'([✓])([ఘపఫషసహ])', r'\2', content)
    content = re.sub(r'([క-హ])ె ై', r'\1ై', content)
    content = re.sub(r'ఏఎ', r'ఏ', content)
    content = re.sub(r'([ఁ-ౌ])[0౦]', r'\1ం', content)
    for _ in 0, 0:
        content = re.sub(r'([ా-ూె-్])(్[క-హ‍])', r'\2\1', content)

    return content
