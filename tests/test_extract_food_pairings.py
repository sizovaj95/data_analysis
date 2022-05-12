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
        test_sent = 'Though mineral-driven, it offers generous white peach and grapefruit flavors that will pair' \
                    ' well with fuller-bodied fish.'
        sent_nlp = self.nlp(test_sent)
        pair_start = test_sent.index('pair')
        expected = {'FOOD_SEAFOOD': ['fuller-bodied fish']}
        result = desc.extract_food_pairings(sent_nlp, pair_start)
        self.assertEqual(expected, result)

    def test_extract_food_pairings_2(self):
        test_sent = 'From one of the few sources of some of these grapes in the Russian River Valley,' \
                    ' this Rhône-inspired wine is ready to pair with lobster or a roast turkey dinner.'
        sent_nlp = self.nlp(test_sent)
        pair_start = test_sent.index('pair')
        expected = {'FOOD_SEAFOOD': ['lobster'],
                    'FOOD_MEAT_AND_POULTRY': ['a roast turkey dinner']}
        result = desc.extract_food_pairings(sent_nlp, pair_start)
        self.assertEqual(expected, result)

    def test_extract_food_pairings_3(self):
        test_sent = 'A great weight of solid tannins are paired with luscious red fruits.'
        sent_nlp = self.nlp(test_sent)
        pair_start = test_sent.index('paired')
        expected = {}
        result = desc.extract_food_pairings(sent_nlp, pair_start)
        self.assertEqual(expected, result)

    def test_extract_food_pairings_4(self):
        test_sent = "It's a fine wine for drinking now with Pinot-friendly fare, such as roast lamb or a wild" \
                    " mushroom risotto with grated cheese."
        sent_nlp = self.nlp(test_sent)
        pair_start = test_sent.index('drinking')
        expected = {'FOOD_MEAT_AND_POULTRY': ['roast lamb'],
                    'FOOD_OTHER': ['a wild mushroom'],
                    'FOOD_GENERAL_DISHES': ['risotto with grated cheese']}
        result = desc.extract_food_pairings(sent_nlp, pair_start)
        self.assertEqual(expected, result)

    def test_extract_food_pairings_5(self):
        test_sent = "Herbaceous in flavor and firm in texture, it brings a freshness and tang with plenty of acidity" \
                    " to perk up fish, shellfish or chicken like a squeeze of fresh lemon."
        sent_nlp = self.nlp(test_sent)
        pair_start = test_sent.index('perk up')
        expected = {'FOOD_SEAFOOD': ['fish', 'shellfish'],
                    'FOOD_MEAT_AND_POULTRY': ['chicken'],
                    'FOOD_FRUITS': ['fresh lemon']}
        result = desc.extract_food_pairings(sent_nlp, pair_start)
        self.assertEqual(expected, result)

    def test_extract_food_pairings_6(self):
        test_sent = "Pair this with fried calamari or pasta topped with clams and mussels."
        sent_nlp = self.nlp(test_sent)
        pair_start = test_sent.index('Pair')
        expected = {'FOOD_SEAFOOD': ['fried calamari', 'clams', 'mussels'],
                    'FOOD_GENERAL_DISHES': ['pasta']}
        result = desc.extract_food_pairings(sent_nlp, pair_start)
        self.assertEqual(expected, result)

    def test_extract_food_pairings_7(self):
        test_sent = 'It is meant for drinking young with its ripe, black currant and plum skin flavors,' \
                    ' stalky acidity and great depth of flavor.'
        sent_nlp = self.nlp(test_sent)
        pair_start = test_sent.index('drinking')
        expected = {}
        result = desc.extract_food_pairings(sent_nlp, pair_start)
        self.assertEqual(expected, result)

    def test_extract_food_pairings_8(self):
        test_sent = "This is an easygoing wine that's neither too oaky nor too crisp, with a decent blend" \
                    " of ripe pear and baking spice flavors paired with soft, smooth texture."
        sent_nlp = self.nlp(test_sent)
        pair_start = test_sent.index('paired')
        expected = {}
        result = desc.extract_food_pairings(sent_nlp, pair_start)
        self.assertEqual(expected, result)

    def test_extract_food_pairings_9(self):
        test_sent = "Enjoy with shrimp stir fried in olive oil for a pairing nirvana."
        sent_nlp = self.nlp(test_sent)
        pair_start = test_sent.index('Enjoy')
        expected = {'FOOD_SEAFOOD': ['shrimp stir fried in olive oil']}
        result = desc.extract_food_pairings(sent_nlp, pair_start)
        self.assertEqual(expected, result)
