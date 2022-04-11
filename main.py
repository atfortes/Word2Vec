import nltk
import torch
from utils import *
from constants import *
from train import train
from skipgram import SkipGram
from text8_dataset import Text8Dataset
from torch.utils.data import DataLoader


EPOCHS = 5
WINDOW_SIZE = 5
BATCH_SIZE = 1024
SUB_SAMP_MIN = 5
SUB_SAMP_T = 1e-6
TRAIN_SIZE = -1
LEARNING_RATE = 0.01
NEG_SAMPLING_SIZE = 5
NEG_SAMPLING_POOL_SIZE = 1e4
EMBEDDING_DIMS = (100, 200, 300)

SAMPLES = [
    "uncle","grandfather","brother","stepfather","dad",
    "lettuce","broccoli","carrot","spinach","asparagus",
    'soccer','baseball','basketball','tennis','boxing'
]

if __name__ == '__name__':

    nltk.download('stopwords')

    dataset = Text8Dataset(
        CORPUS_FILENAME,
        TRAIN_SIZE,
        SUB_SAMP_MIN,
        SUB_SAMP_T,
        WINDOW_SIZE,
        NEG_SAMPLING_POOL_SIZE,
        NEG_SAMPLING_SIZE
    )

    models = (
        SkipGram(dataset.__len__(), EMBEDDING_DIMS[0]).to(DEVICE),
        SkipGram(dataset.__len__(), EMBEDDING_DIMS[1]).to(DEVICE),
        SkipGram(dataset.__len__(), EMBEDDING_DIMS[2]).to(DEVICE)
    )

    for model in models:
        dataloader = DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=BATCH_SIZE
        )
        optimizer = torch.optim.SparseAdam(model.parameters())
        train(model, dataset, dataloader, optimizer, EPOCHS, DEVICE)
        plot_embeddings(model, dataset, SAMPLES)
