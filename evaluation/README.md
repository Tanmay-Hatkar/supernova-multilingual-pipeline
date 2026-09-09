# evaluation/

Judge selection and validation, per language, against human labeled data.

The pattern that scales: each language's judge is selected and
recorded in its own `configs/judges/<code>.yaml`, never as a single
global choice. See the language registry in `gepa/language_registry.py`
for how this is loaded.
