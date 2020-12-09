import logging
import os
import unicodedata

# zemberek-python -> https://github.com/Loodos/zemberek-python
from zemberek import TurkishSentenceNormalizer, TurkishSentenceExtractor, TurkishMorphology

SPIECE_UNDERLINE = u"▁".encode("utf-8")


class TextNormalization:
    """Text normalization task
    """

    def __init__(self):
        """Constructor method
        """
        self.zemberek_morpholgy = TurkishMorphology.create_with_defaults()
        self.zemberek_normalizer = TurkishSentenceNormalizer(self.zemberek_morpholgy)
        self.zemberek_extractor = TurkishSentenceExtractor()

    def normalize(self,
                  text: str,
                  remove_space: bool = True,
                  do_lower_case: bool = True,
                  normalize_function: str = "NFKC",
                  is_turkish: bool = True,
                  use_zemberek: bool = True):
        """Preprocess text by removing extra space and normalizing via python-unicodedata library.

        :param str text: Text for normalization
        :param bool remove_space: Whether remove empty spaces or not (defaults to True)
        :param bool do_lower_case: Whether do lower case or not (defaults to True)
        :param str normalize_function: Unicodedata normalize function.
            Either "NFC", "NFKC", "NFD" or "NFKD". (defaults to "NFKC")
        :param bool is_turkish: Whether text is in Turkish or not (defaults to True)
        :param bool use_zemberek: Whether to use Zemberek-Python's normalizer. Always do lowercase inside (defaults to True)
        :return: Normalized text
        """
        outputs: str = text

        if remove_space:
            outputs = " ".join(outputs.strip().split())

        outputs = unicodedata.normalize(normalize_function, outputs)
        outputs = "".join([c for c in outputs if not unicodedata.combining(c)])

        if use_zemberek:
            sentences = self.zemberek_extractor.from_paragraph(outputs)
            normalized_sentences = []
            for sentence in sentences:
                normalized_sentences.append(self.zemberek_normalizer.normalize(sentence))
            outputs = "".join(normalized_sentences)

        if do_lower_case:
            if is_turkish:
                outputs = outputs.replace('\u0049', '\u0131')  # I -> ı
                outputs = outputs.replace('\u0130', '\u0069')  # İ -> i

            outputs = outputs.casefold()

        return outputs
