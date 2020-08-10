


class Stats:

    def __init__(self, text):
        self._text = text

    @property
    def text(self):
        return self._text

    @property
    def characters(self):
        text = self._text.replace(" ", "")
        return [*text]

    @property
    def letters(self):
        text = remove_punctuation(self._text)
        return [*text]

    @property
    def words(self):
        return []

    @property
    def syllables(self):
        return []

    @property
    def difficult_words(self):
        return []

    def avg(self, attribute, per):
        attribute_count = float(len(getattr(self, attribute)))
        per_count = float(len(getattr(self, per)))

        try:
            return attribute_count / per_count
        except ZeroDivisionError:
            return 0.0


class Text(Stats):

    @property
    def sentences(self):
        return []


class Sentence(Stats):
    ...
