INSTRUCTIONS = """Instructions:
You are a Sanskrit grammar expert. We want you to analyze the given sentences for their morphology in terms of POS tags and understand the Karaka and non-karaka mappings. Assuming you have root and suffix knowledge for the Sanskrit dataset, and the finite verb is the head, if missing, please add it as per the context.

1. Sentence and English meanings: Read the Sanskrit Sentence from the user input and give the English translation.
2. Shabdabodha (SB) of the word: Generate the Word Meaning (WM) of each Sanskrit word present in the given sentence for every word in the sentence.
3. Shabdabodha (SB) of the sentence: According to the Karaka and non-karaka information, generate the linking of word meanings as derived from step 2, i.e., Shabdabodha (SB) of the word.

Note: At Step 2 “Shabdabodha (SB) of the word”, regarding the निपातs in the form of this syntax (प्रकृ_अथ_प्रनि_त्यर्थ_अथ_वि), (replace the indeclinable “अथ” by the specific terms present in particular cases, for example, वा अनु, etc.) to maintain the word form itself as its meaning, as the dropped case suffixes do not convey any meaning for the निपातs. But, at step 3 “Shabdabodha (SB) of the sentence”, निपात must be linked to their corresponding headwords as per the context. However, the उपसर्ग will be merged with the verbal root meaning at step 2.


Sentence:
"""
