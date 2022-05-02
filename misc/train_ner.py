import spacy
import json
import random
from spacy.training import Example
import numpy as np

import logic.constants as co

nlp = spacy.load("en_core_web_sm")

LABELS = ["FOOD_MEAT AND POULTRY", "FOOD_SEAFOOD", "FOOD_GENERAL DISHES", "FOOD_COUNTRY CUISINE", "FOOD_OTHER"]


def main():
    ner = nlp.get_pipe('ner')
    with open(co.data_dir / "ner_training_data.json", 'r') as f:
        data = json.load(f)
        train_data = data[:-50]
        test_data = data[-50:]
    # for label in LABELS:
    #     ner.add_label(label)
    # optimizer = nlp.resume_training()
    # move_names = list(ner.move_names)
    # pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    # other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # with nlp.disable_pipes(*other_pipes):
    #     sizes = spacy.util.compounding(4, 32, 1.001)
    #     loss_delta = np.inf
    #     i = 0
    #     loss_per_prev_iter = 0
    #     stop_cond = 0
    #     while loss_delta > 0.01:
    #     # for i in range(30):
    #
    #         print(f"ITERATION: {i}")
    #         previous_loss = 0
    #         random.shuffle(train_data)
    #         batches = spacy.util.minibatch(train_data, size=sizes)
    #         losses = {}
    #         all_losses = []
    #         for batch in batches:
    #             examples = []
    #             for text, annotation in batch:
    #                 doc = nlp.make_doc(text)
    #                 example = Example.from_dict(doc, annotation)
    #                 examples.append(example)
    #             nlp.update(examples, sgd=optimizer, drop=0.35, losses=losses)
    #             current_loss = losses['ner'] - previous_loss
    #             all_losses.append(current_loss)
    #             previous_loss = losses['ner']
    #             # print(f"Loss: {current_loss}")
    #             # print("Losses", losses)
    #         loss_per_current_iter = np.mean(all_losses)
    #         print(f"Average loss per iteration {i}: {loss_per_current_iter}")
    #         loss_delta = abs(loss_per_current_iter - loss_per_prev_iter)
    #         if loss_delta <= 0.01:
    #             stop_cond += 1
    #         else:
    #             stop_cond = 0
    #         loss_per_prev_iter = loss_per_current_iter
    #         i += 1
    #
    # nlp.meta['name'] = 'food_ner'
    #
    output_dir = co.data_dir / 'ner_model'
    # if not output_dir.exists():
    #     output_dir.mkdir()
    #
    # print(f"Saved model to {output_dir}")
    # nlp.to_disk(output_dir)

    print(f"Loading model from {output_dir}")
    food_nlp = spacy.load(output_dir)
    test_texts = [d[0] for d in test_data]
    entities = [d[1] for d in test_data]
    for test_text, entities in test_data:
        doc = food_nlp(test_text)
        print('\n')
        print("Entities in '%s'" % test_text)
        print("Predicted entities:")
        for ent in doc.ents:
            print(ent.text, ent.label_)
        print('\n')
        print("Regex entities:")
        entities = entities['entities']
        for ent in entities:
            print(f"{test_text[ent[0]:ent[1]]}, {ent[2]}")



if __name__ == "__main__":
    main()
