"""Modules to extract stimulus features."""
import numpy as np
import torch

from typing import List
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .DataSequence import DataSequence
from .hard_coded_things import (
    bad_words,
    bad_words_with_sentence_boundaries,
    sentence_end_word,
    sentence_start_words,
)

if torch.cuda.is_available():
    print("Using gpu")
    device = "cuda"
else:
    print("Using cpu")
    device = "cpu"


def mapdict(d, fun):
    return dict(list(zip(list(d.keys()), list(map(fun, list(d.values()))))))


def clean_split_inds(removed_indices: List[int], split_inds: np.ndarray):
    for removed_index in removed_indices[::-1]:
        split_inds[split_inds > removed_index] = (
            split_inds[split_inds > removed_index] - 1
        )
    return split_inds


def get_clean_text(ds, words_to_remove=sentence_start_words):
    text = np.array(ds.data)
    indices_to_remove = []
    indices_to_remove += list(np.where(np.isin(text, words_to_remove))[0])
    indices_to_remove = list(set(indices_to_remove))
    indices_to_remove.sort()
    text = np.delete(text, indices_to_remove)
    sentence_ends = np.where(text == sentence_end_word)[0]
    sentence_starts = np.array([-1] + list(sentence_ends[:-1])) + 1
    text[sentence_ends] = "."
    text = np.array(text)
    data_times = np.copy(ds.data_times)
    data_times = np.delete(data_times, indices_to_remove)
    split_inds = np.copy(ds.split_inds)
    split_inds = clean_split_inds(
        removed_indices=indices_to_remove, split_inds=split_inds
    )
    return sentence_starts, sentence_ends, split_inds, data_times, text


def make_word_ds(
    grids, trfiles, bad_words=bad_words_with_sentence_boundaries, extra_bad_words=[]
):
    """Creates DataSequence objects containing the words from each grid, with any words appearing
    in the [bad_words] set removed.
    """
    ds = dict()
    bad_words = frozenset(list(bad_words) + extra_bad_words)
    stories = grids.keys()
    for st in stories:
        grtranscript = grids[st].tiers[1].make_simple_transcript()
        # Filter out bad words
        goodtranscript = [
            x for x in grtranscript if x[2].lower().strip("{}").strip() not in bad_words
        ]
        goodtranscript = [(x[0], x[1], x[2].lower().strip()) for x in goodtranscript]
        d = DataSequence.from_grid(goodtranscript, trfiles[st][0])
        ds[st] = d
    return ds


class Features(object):
    def __init__(
        self,
        grids,
        trfiles,
        bad_words=bad_words,
        bad_words_with_sentence_boundaries=bad_words_with_sentence_boundaries,
        **kwargs,
    ):
        """Initializes a Features object that can be used to create feature-space
        representations of the stimulus with the given [grids] and [trfiles].

        [kwargs] are passed to the interpolation function.
        """
        self.grids = grids
        self.trfiles = trfiles

        self.interpargs = kwargs
        self.wordseqs_with_sentence_boundaries = make_word_ds(
            grids, trfiles, bad_words=bad_words
        )
        self.wordseqs = make_word_ds(
            self.grids, self.trfiles, bad_words=bad_words_with_sentence_boundaries
        )

    def downsample(self, dsdict, interp, **kwargs):
        """Downsamples each DataSequence in [dsdict] using the settings specified in the
        initializer.
        """
        return mapdict(dsdict, lambda h: h.chunksums(interp, **kwargs))

    def numwords(self):
        """Simple model: the number of words per TR."""
        return mapdict(
            self.wordseqs,
            lambda s: np.atleast_2d(list(map(len, s.chunks()))).T.astype(float),
        )

    def contextual_lm(
        self,
        layer_num,
        model_name,
        downsample=True,
        interp="lanczos",
        split_type: str = "sentence",
        bert_trial_num=None,
        bert_step_num=None,
        max_seq_length: int = 512,
    ):
        clm_stimulus = dict()
        for stimulus_name, ds in list(self.wordseqs_with_sentence_boundaries.items()):
            clm_stimulus[stimulus_name] = get_contextual_embeddings(
                ds_with_sentence_boundaries=ds,
                model_name=model_name,
                layer_num=layer_num,
                split_type=split_type,
                #bert_trial_num=bert_trial_num,
                #bert_step_num=bert_step_num,
                max_seq_length=max_seq_length,
            )
        if downsample:
            return self.downsample(clm_stimulus, interp=interp)
        else:
            return mapdict(clm_stimulus, lambda h: h.data)


def get_contextual_embeddings(
    ds_with_sentence_boundaries: DataSequence,
    layer_num: int,
    use_special_tokens: bool = True,
    model_name: str = "bert-base-uncased",
    split_type: str = "sentence",
    max_seq_length: int = 512,
):
    """Extracts contextual embeddings from a pretrained language model.

    Parameters:
    ----------
    ds_with_sentence_boundaries: A DataSequence containing stimuli for which to retrieve embeddings.
    layer_num: The layer from which to extract embeddings or `-1` if using all the layers.
    use_special_tokens: If True, includes the special tokens (e.g., [CLS] and [SEP] for BERT) in the input sequence.
    model_name: The pretrained model to load.
    split_type: If `sentence`, feed one sentence at a time. If `causal_all`, feed in all preceding words, up to max_seq_length words.
    max_seq_length: The maximum number of words to feed into the model at a time. Only used if `split_type` is `causal_all`.
    """

    torch.manual_seed(0)

    def token_to_word_embeddings(
        layer_embedding: np.ndarray,
        input_sequence: List[str],
        tokenizer,
        embedding_type: str = "mean",
    ):
        """Converts an array of token embeddings to an array of word embeddings.

        Uses the last token of a word as its embedding.
        If remove_period == True, does not include the last embedding (which is the . in the sentence).
        """
        word_embeddings = []
        embedding_index = 0
        if use_special_tokens:
            layer_embedding_cleaned = layer_embedding[1:-1]
        else:
            layer_embedding_cleaned = layer_embedding
        for word in input_sequence:
            num_word_tokens = len(tokenizer.tokenize(word))
            if embedding_type == "mean":
                word_embeddings.append(
                    np.expand_dims(
                        layer_embedding_cleaned[
                            embedding_index : embedding_index + num_word_tokens
                        ].mean(0),
                        axis=0,
                    )
                )
            elif embedding_type == "max":
                word_embeddings.append(
                    np.expand_dims(
                        np.max(
                            layer_embedding_cleaned[
                                embedding_index : embedding_index + num_word_tokens
                            ].numpy(),
                            axis=0,
                        ),
                        axis=0,
                    )
                )
            elif embedding_type == "last":
                word_embeddings.append(
                    np.expand_dims(layer_embedding_cleaned[embedding_index - 1], axis=0)
                )
            embedding_index += num_word_tokens
        assert (embedding_index) == len(layer_embedding_cleaned)
        return word_embeddings

    # Create model and tokenizer.
    config = AutoConfig.from_pretrained(model_name)
    config.output_hidden_states = True
    config.output_attentions = False
    model = AutoModel.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)

    new_data = []
    if split_type == "sentence":
        sentence_starts, sentence_ends, split_inds, data_times, text = get_clean_text(
            ds_with_sentence_boundaries, 
            # remove_repeated_words=False,
        )
    elif split_type == "causal_all":
        _, _, split_inds, data_times, text = get_clean_text(
            ds_with_sentence_boundaries, 
            # remove_repeated_words=False
        )
        sentence_starts = [
            np.clip(0, a_min=word_index - max_seq_length, a_max=None)
            for word_index in range(len(text))
        ]
        sentence_ends = [word_index for word_index in range(len(text))]
    print(
        f"Extracting embeddings from {model_name} using {split_type} split type. {len(sentence_starts)} input sequences."
    )
    for sentence_start, sentence_end in zip(sentence_starts, sentence_ends):
        sentence = text[sentence_start : sentence_end + 1]
        if sentence[-1] == ".":
            sentence = sentence[:-1]
        else:
            continue
        tokenized_input_sequence = np.concatenate(
            [tokenizer.tokenize(word) for word in sentence]
        )
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_input_sequence)
        if use_special_tokens:
            indexed_tokens = tokenizer.build_inputs_with_special_tokens(indexed_tokens)
        tokens_tensor = torch.tensor([indexed_tokens]).to(device)

        with torch.no_grad():
            outputs = model(tokens_tensor)
            if layer_num == -1:
                layer_embedding = torch.cat(outputs[-1], dim=-1)[0].to(
                    "cpu"
                )  # [num_tokens x (hidden_size * num_layers)]
            else:
                layer_embedding = outputs[-1][layer_num][0].to(
                    "cpu"
                )  # [num_tokens x hidden_size]
            if split_type == "sentence":
                layer_embedding = token_to_word_embeddings(
                    layer_embedding=layer_embedding,
                    input_sequence=sentence,
                    tokenizer=tokenizer,
                )
            elif split_type == "causal_all":
                layer_embedding = [np.expand_dims(layer_embedding[-1], axis=0)]
            new_data += layer_embedding
    new_data = np.squeeze(np.array(new_data))
    embedding_ds = DataSequence(
        new_data, split_inds, data_times, ds_with_sentence_boundaries.tr_times
    )
    return embedding_ds
