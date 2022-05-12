import spacy
import json
import random
from spacy.training import Example
import numpy as np
import datetime as dt
import time
from typing import List, Dict, Tuple, Optional, Iterable
from pathlib import Path

import logic.constants as co


SPACY_TRAIN_DATA_FORMAT = List[Tuple[str, Dict[str, List[Iterable]]]]

time_now = dt.datetime.now()
time_suffix = f"{time_now.year}_{time_now.month}_{time_now.day}"

LABELS = co.food_labels
PIPE_EXCEPTIONS = ["ner", "trf_wordpiecer", "trf_tok2vec"]
MODEL_NAME = 'food_ner'
OUTPUT_FOLDER_NAME = 'custom_ner_{}_{}'
STOP_NUMBER = 2
TOLERANCE = 0.1
DROPOUT_RATE = 0.35


def load_and_split_data() -> Tuple[SPACY_TRAIN_DATA_FORMAT, SPACY_TRAIN_DATA_FORMAT]:
    with open(co.data_dir / "ner_training_data.json", 'r') as f:
        data = json.load(f)
        train_data = data[:-50]
        test_data = data[-50:]
    return train_data, test_data


def update_model(train_data: SPACY_TRAIN_DATA_FORMAT):
    nlp = spacy.load("en_core_web_sm")
    ner = nlp.get_pipe('ner')

    for label in LABELS:
        ner.add_label(label)
    optimizer = nlp.resume_training()

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in PIPE_EXCEPTIONS]
    with nlp.disable_pipes(*other_pipes):
        sizes = spacy.util.compounding(4, 64, 1.001)
        i = 0
        loss_per_prev_iter = 0
        stop_cond = 0
        while stop_cond <= STOP_NUMBER and i < 10:
            start_time = time.time()
            print(f"ITERATION: {i}")
            previous_loss = 0
            random.shuffle(train_data)
            batches = spacy.util.minibatch(train_data, size=sizes)
            losses = {}
            all_losses = []
            for batch in batches:
                examples = []
                for text, annotation in batch:
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotation)
                    examples.append(example)
                nlp.update(examples, sgd=optimizer, drop=DROPOUT_RATE, losses=losses)
                current_loss = losses['ner'] - previous_loss
                all_losses.append(current_loss)
                previous_loss = losses['ner']
            loss_per_current_iter = np.mean(all_losses)
            print(f"Average loss per iteration {i}: {loss_per_current_iter}")
            loss_delta = abs(loss_per_current_iter - loss_per_prev_iter)
            if loss_delta <= TOLERANCE:
                stop_cond += 1
            else:
                stop_cond = 0
            loss_per_prev_iter = loss_per_current_iter
            i += 1
            duration = time.time() - start_time
            unit = 'sec'
            if duration >= 60:
                unit = 'min'
                duration /= 60
            print("Iteration {} took {:.2f} {}".format(i, duration, unit))
    return nlp


def test_model(model_dir: Path, test_data: SPACY_TRAIN_DATA_FORMAT):
    print(f"Loading model from {model_dir}")
    food_nlp = spacy.load(model_dir)

    correctly_predicted_entities = 0
    total_regex_entities = 0
    for test_text, entities in test_data:
        doc = food_nlp(test_text)

        predicted_entities = set([(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents])

        regex_entities = entities['entities']
        regex_entities = set([tuple(ent) for ent in regex_entities])
        total_regex_entities += len(regex_entities)
        intersection = predicted_entities.intersection(regex_entities)
        correctly_predicted_entities += len(intersection)

        in_predicted_not_in_actual = predicted_entities.difference(regex_entities)
        in_actual_not_in_predicted = regex_entities.difference(predicted_entities)
        if in_predicted_not_in_actual:
            print('\n')
            print(test_text)
            print("Found in predicted labels but not in actual labels:")
            for start, end, label in in_predicted_not_in_actual:
                print(f"{test_text[start:end]} {label}")

        if in_actual_not_in_predicted:
            print('\n')
            print(test_text)
            print("Found in actual labels but not in predicted labels:")
            for start, end, label in in_actual_not_in_predicted:
                print(f"{test_text[start:end]} {label}")
    print('\n')
    print(f"Correctly predicted: {correctly_predicted_entities} entities ({total_regex_entities} total actual entities)")


def main():

    train_data, test_data = load_and_split_data()
    # nlp = update_model(train_data)
    #
    # nlp.meta['name'] = MODEL_NAME
    #
    # output_dir = co.data_dir / OUTPUT_FOLDER_NAME.format(time_suffix, len(train_data))
    # if not output_dir.exists():
    #     output_dir.mkdir()
    #
    # print(f"Saved model to {output_dir}")
    # nlp.to_disk(output_dir)

    model_dir = co.data_dir / co.custom_nlp_model_name
    test_model(model_dir, test_data)


if __name__ == "__main__":
    main()
