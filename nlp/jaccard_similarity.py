def jaccard(text1, text2):
    token1= set(text1.lower().split())
    token2 = set(text2.lower().split())
    intersection = token1 & token2
    union = token1 | token2
    return len(intersection)/len(union)