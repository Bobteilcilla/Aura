# package_aura/multiple_mapping.py

'''This module provides functions to map numerical values to categorical labels
for different discomfort levels. We can always fine-tune the mapping logic here.'''

def discomfort_to_label(d: float) -> str:
    if d <= 0.2:
        return "very_comfortable"
    elif d <= 0.4:
        return "comfortable"
    elif d <= 0.6:
        return "neutral"
    elif d <= 0.8:
        return "uncomfortable"
    else:
        return "stressed"
