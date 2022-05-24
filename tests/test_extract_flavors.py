from unittest import TestCase
import spacy

import logic.analyse_descriptions as desc
import logic.constants as co


class TestFoodPairings(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        output_dir = co.data_dir / co.custom_nlp_model_name
        cls.nlp = spacy.load(output_dir)

    def test_extract_food_pairings_1(self):
        test_sent = 'Aromatic, dense and toasty, it deftly blends aromas and flavors of toast, cigar box, blackberry,' \
                    ' black cherry, coffee and graphite.'
        sent_nlp = self.nlp(test_sent)
        expected = {'FOOD_OTHER': ['cigar box', 'coffee', 'graphite'],
                    'FOOD_BERRIES': ['blackberry'],
                    'FOOD_FRUITS': ['black cherry']}
        result = desc.extract_flavors(sent_nlp)
        self.assertEqual(expected, result)

    def test_extract_food_pairings_2(self):
        test_sent = 'Chocolate is a key flavor, while baked berry and cassis flavors are hardly wallflowers.'
        sent_nlp = self.nlp(test_sent)
        expected = {'FOOD_SWEET': ['Chocolate'],
                    'FOOD_BERRIES': ['baked berry and cassis flavors']}
        result = desc.extract_flavors(sent_nlp)
        self.assertEqual(expected, result)

    def test_extract_food_pairings_3(self):
        test_sent = 'That delicately lush flavor has considerable length.'
        sent_nlp = self.nlp(test_sent)
        expected = {}
        result = desc.extract_flavors(sent_nlp)
        self.assertEqual(expected, result)
