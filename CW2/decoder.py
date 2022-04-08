"""
COMP5623M Coursework on Image Caption Generation


python decoder.py
"""

import torch
import random
import nltk
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from datasets import Flickr8k_Images, Flickr8k_Features
from models import DecoderRNN, EncoderCNN
from utils import *
from config import *

# if false, train model; otherwise try loading model from checkpoint and evaluate
EVAL = False


# reconstruct the captions and vocab, just as in extract_features.ipynb
lines = read_lines(TOKEN_FILE_TRAIN)
image_ids, cleaned_captions = parse_lines(lines)
vocab = build_vocab(cleaned_captions)


# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if not EVAL:
    # load the features saved from extract_features.ipynb
    print(len(lines))
    features = torch.load('features.pt', map_location=device)
    print("Loaded features", features.shape)

    features = features.repeat_interleave(5, 0)
    print("Duplicated features", features.shape)

    dataset_train = Flickr8k_Features(
        image_ids=image_ids,
        captions=cleaned_captions,
        vocab=vocab,
        features=features,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64,  # change as needed
        shuffle=True,
        num_workers=0,  # may need to set to 0
        collate_fn=caption_collate_fn,  # explicitly overwrite the collate_fn
    )

    # initialize the models and set the learning parameters
    decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab), NUM_LAYERS).to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)

    print(len(image_ids))
    print(len(cleaned_captions))
    print(features.shape)


#########################################################################
#
#        QUESTION 1.3 Training DecoderRNN
#
#########################################################################

    # Write training loop on decoder here
    for i in range(5):
        running_loss = 0
        n = 0

        for features, captions, lengths in train_loader:
            features, captions = features.to(device), captions.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward, backward, and update parameters
            outputs = decoder(features, captions, lengths)

            # for each batch, prepare the targets using this torch.nn.utils.rnn function
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # accumulate loss
            running_loss += loss.item()
            n += 1

        train_loss = running_loss / n
        print(f"{i + 1}  train_loss: {train_loss:.3f}")

    # save model after training
    torch.save(decoder, "decoder.ckpt")


# if we already trained, and EVAL == True, reload saved model
else:
    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),   # using ImageNet norms
                             (0.229, 0.224, 0.225))])

    test_lines = read_lines(TOKEN_FILE_TEST)
    test_image_ids, test_cleaned_captions = parse_lines(test_lines)
    test_image_ids = test_image_ids[::5]

    # load models
    encoder = EncoderCNN().to(device)
    encoder.eval()
    decoder = torch.load("decoder.ckpt").to(device)
    decoder.eval()  # generate caption, eval mode to not influence batchnorm


#########################################################################
#
#        QUESTION 2.1 Generating predictions on test data
#
#########################################################################

    dataset_test = Flickr8k_Images(
        image_ids=test_image_ids,
        transform=data_transform
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    test_features = torch.tensor([], device=device)
    for images in test_loader:
        images = images.to(device)
        out = encoder(images).squeeze()
        test_features = torch.cat((test_features, out), dim=0)

    outputs = decoder.sample(test_features)
    # torch.save(outputs, 'test_outputs.pt')
    # outputs = torch.load("test_outputs.pt").to(device)

    index = random.randint(0, outputs.shape[0])

    # Define decode_caption() function in utils.py
    predicted_caption = decode_caption(outputs[index].tolist(), vocab)

    print(f"Image ID: {test_image_ids[index]}")
    print('Reference Captions:')
    for i in range(index*5, index*5 + 5):
        print(test_cleaned_captions[i])
    print('Generated Caption:')
    print(predicted_caption)


#########################################################################
#
#        QUESTION 2.2-3 Caption evaluation via text similarity
#
#########################################################################

    bleu_scores = []
    for index in range(outputs.shape[0]):
        reference = []
        for i in range(index*5, index*5 + 5):
            reference.append(test_cleaned_captions[i].split())
        candidate = decode_caption(outputs[index].tolist(), vocab).split()

        bleu_scores.append(sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))

    print('BLEU Statistics:')
    print(stats.describe(bleu_scores))

    print('\n----------Worst Caption (BLEU)----------')
    print_captions(test_image_ids, test_cleaned_captions, outputs, np.argmin(bleu_scores), vocab)
    print('----------Best Caption (BLEU)----------')
    print_captions(test_image_ids, test_cleaned_captions, outputs, np.argmax(bleu_scores), vocab)


    cosine_sim_scores = []
    for index in range(outputs.shape[0]):
        candidate_embedding = embed_caption(decode_caption(outputs[index].tolist(), vocab), vocab, decoder, device)

        similarity_sum = 0
        for i in range(index * 5, index * 5 + 5):
            reference_embedding = embed_caption(test_cleaned_captions[i], vocab, decoder, device)

            similarity_sum += cosine_similarity(candidate_embedding, reference_embedding)[0][0]

        cosine_sim_scores.append(similarity_sum / 5)

    cosine_sim_scores = MinMaxScaler().fit_transform(np.array(cosine_sim_scores).reshape(-1, 1)).squeeze().tolist()

    print('Cosine Similarity Statistics:')
    print(stats.describe(cosine_sim_scores))

    print('\n----------Worst Caption (Cosine Similarity)----------')
    print_captions(test_image_ids, test_cleaned_captions, outputs, np.argmin(cosine_sim_scores), vocab)
    print('----------Best Caption (Cosine Similarity)----------')
    print_captions(test_image_ids, test_cleaned_captions, outputs, np.argmax(cosine_sim_scores), vocab)

    score_sims = np.abs(np.subtract(bleu_scores, cosine_sim_scores))
    most_similar_score = np.argmin(score_sims)
    least_similar_score = np.argmax(score_sims)

    print('\n----------Most similar score----------')
    print(f"BLEU score: {bleu_scores[most_similar_score]:.3f} Cosine Similarity score: {cosine_sim_scores[most_similar_score]:.3f}")
    print_captions(test_image_ids, test_cleaned_captions, outputs, most_similar_score, vocab)
    print('----------Least similar score----------')
    print(f"BLEU score: {bleu_scores[least_similar_score]:.3f} Cosine Similarity score: {cosine_sim_scores[least_similar_score]:.3f}")
    print_captions(test_image_ids, test_cleaned_captions, outputs, least_similar_score, vocab)
