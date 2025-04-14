## odd narratives
idx= 171, 107, 175, 221

# Situation 1: conjunction
MANIP=True
idx= 293
narrative = """The model predicted with a 66.00% probability that the student would pass the year.
The student's limited free time after school, rated as 1, was the most influential factor, contributing negatively to the likelihood of passing, possibly indicating a lack of balance between study and relaxation.
The absence of past class failures, with a value of 0, contributed positively, as having no failures typically indicates a strong academic record.
The mother's job not being categorized as "other," with a value of 0, contributed positively, possibly indicating that certain job categories might provide more stability or resources beneficial for academic success.
However, the student's low frequency of going out with friends, rated as 2, positively impacted the likelihood of passing, suggesting that limited social interactions might allow for more focus on studies.
Overall, while the prediction leaned towards passing, the positive influences of the mother's job category, absence of past failures, and limited social outings were slightly countered by the negative aspect of limited free time.
"""
# when i use the system_message_coherence to use gpt directly (not from api), it can successfully identify the following isses, however, through my agent system, it says no issue.
# the "however used here is not coherent; 
# and "The mother's job not being categorized as "other," with a value of 0, contributed..." can be better like "The mother's job being in a category other than 'other' (value = 0) contributed positively…""


# Situation 2: redundant information from original manipulated narraive & Adverbs of Degree："primary"
MANIP=True
idx= 309
narrative = """The model predicted a 64.00% probability that the student would pass, indicating a moderate level of confidence in this outcome.
The student's age of 19, slightly above the average, might indicate maturity and experience, which can be advantageous in academic settings, but in this case, it negatively influenced the prediction.
The student's history of past class failures, with only one recorded failure, also contributed negatively, suggesting that even a single failure can impact the likelihood of passing.
Additionally, the student received extra educational support, indicated by a value of 1, which is generally associated with improved academic performance, but in this instance, it had a negative impact, possibly due to the student's reliance on support rather than independent study.
Lastly, the primary factor contributing to this prediction was the father's education level, which was at the primary education level.
This likely did not provide sufficient academic support, negatively influencing the prediction.
In summary, the combination of age, past failures, additional support, and parental education collectively contributed to the model's prediction that the student would pass.
"""
# when 
# when MANIP=False the narrative should be like this:
"""The model predicted with 64% certainty that the student would pass the final year.
However, several factors contributed negatively to this prediction.
The student's age of 19, which is higher than the average, likely suggests a delay in their academic progress, possibly due to past failures or late school entry, thus reducing the likelihood of passing.
Additionally, the student had one past class failure, which further decreased the probability of passing, as past academic struggles can indicate ongoing challenges.
Receiving extra educational support, indicated by a value of 1, while generally beneficial, in this case, might imply that the student is struggling more than their peers, contributing negatively to the prediction.
Lastly, the father's education level being at primary education suggests limited academic support at home, which can be a disadvantage in the student's educational journey.
In summary, while the model predicted a pass, the student's age, past failures, need for extra support, and father's education level all contributed negatively to the prediction.
"""

