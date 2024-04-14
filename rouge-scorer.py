from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

generated = """
A 38-year-old woman in the US, who was apprehended twice for allegedly trying to jump the White House fence last week, has been arrested for scaling a fence at the Treasury Building. She was arrested after an alarm sounded at about 2:15 am yesterday when she scaled a fence at the building. She was charged with unlawful entry and contempt of court.
"""

reference = """
A woman, who was arrested twice last week for trying to jump the White House fence, has been arrested for the third time for scaling a fence at the Treasury Building, next to the White House. The woman, who told officers that she wanted to speak to President Donald Trump, has been charged with unlawful entry and contempt of court.
"""

scores = scorer.score(reference, generated)

print(scores)