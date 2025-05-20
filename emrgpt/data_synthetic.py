"""
TODO

- Randomly insert medication that should not be given: expect [error] token
    - Potassium (may have to check that this isn't actually indicated e.g. only give iso hyperK)
    - Insulin (any dose iso of hypoglycemia)
    - Beta-blocker iso bradycardia
    - Pressor iso hypertension
    - Anticoagulants iso hemorrhage / low Hct
    - Extubation low O2 / high vent settings
    - NOTE: will need specialized loss handling for this case:
        Don't want the model to learn [normal sequence] [bad med]
        While it's learning [normal sequence] [bad med] [error token]
        Will need to mask loss except last token

- Randomly 10x increase dose of medication that is given: expect [error] token


- Start with 90% real, 10% synthetic data
"""
