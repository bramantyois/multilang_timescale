import os
import numpy as np
from collections import defaultdict
import pdb

from .textgrid import TextGrid
from .features import Features

from ..configurations import (
    feature_sets_path,
    grids_path,
    trs_path,
)

from .hard_coded_things import featuresets_dict

def load_grid(story, grid_dir=grids_path):
    """Loads the TextGrid for the given [story] from the directory [grid_dir].
    The first file that starts with [story] will be loaded, so if there are
    multiple versions of a grid for a story, beward.
    """
    gridfile = [
        os.path.join(grid_dir, gf)
        for gf in os.listdir(grid_dir)
        if gf.startswith(story)
    ][0]
    return TextGrid(open(gridfile).read())


def load_grids_for_stories(stories, grid_dir=grids_path):
    """Loads grids for the given [stories], puts them in a dictionary."""
    return dict([(st, load_grid(st, grid_dir)) for st in stories])


def load_5tier_grids_for_stories(stories, rootdir):
    grids = dict()
    for story in stories:
        storydir = os.path.join(
            rootdir, [sd for sd in os.listdir(rootdir) if sd.startswith(story)][0]
        )
        storyfile = os.path.join(
            storydir, [sf for sf in os.listdir(storydir) if sf.endswith("TextGrid")][0]
        )
        grids[story] = TextGrid(open(storyfile).read())
    return grids



class TRFile(object):
    def __init__(self, trfilename, expectedtr=2.0045):
        """Loads data from [trfilename], should be output from stimulus presentation code."""
        self.trtimes = []
        self.soundstarttime = -1
        self.soundstoptime = -1
        self.otherlabels = []
        self.expectedtr = expectedtr

        if trfilename is not None:
            self.load_from_file(trfilename)

    def load_from_file(self, trfilename):
        """Loads TR data from report with given [trfilename]."""
        ## Read the report file and populate the datastructure
        for ll in open(trfilename):
            timestr = ll.split()[0]
            label = " ".join(ll.split()[1:])
            time = float(timestr)

            if label in ("init-trigger", "trigger"):
                self.trtimes.append(time)

            elif label == "sound-start":
                self.soundstarttime = time

            elif label == "sound-stop":
                self.soundstoptime = time

            else:
                self.otherlabels.append((time, label))

        ## Fix weird TR times
        itrtimes = np.diff(self.trtimes)
        badtrtimes = np.nonzero(itrtimes > (itrtimes.mean() * 1.5))[0]
        newtrs = []
        for btr in badtrtimes:
            ## Insert new TR where it was missing..
            newtrtime = self.trtimes[btr] + self.expectedtr
            newtrs.append((newtrtime, btr))

        for ntr, btr in newtrs:
            self.trtimes.insert(btr + 1, ntr)

    def simulate(self, ntrs):
        """Simulates [ntrs] TRs that occur at the expected TR."""
        self.trtimes = list(np.arange(ntrs) * self.expectedtr)

    def get_reltriggertimes(self):
        """Returns the times of all trigger events relative to the sound."""
        return np.array(self.trtimes) - self.soundstarttime

    @property
    def avgtr(self):
        """Returns the average TR for this run."""
        return np.diff(self.trtimes).mean()


def load_generic_trfiles(stories, root=trs_path):
    """Loads a dictionary of generic TRFiles (i.e. not specifically from the session
    in which the data was collected.. this should be fine) for the given stories.
    """
    trdict = dict()

    for story in stories:
        try:
            trf = TRFile(os.path.join(root, "%s.report" % story))
            trdict[story] = [trf]
        except (Exception, e):
            print(e)

    return trdict


def get_feature(featureset, features_object):
    model_function, model_kwargs = featureset[0]
    modeldata = getattr(features_object, model_function)(**model_kwargs)
    return modeldata


def load_story_info(story_name: str, grids_path: str, trs_path: str, featureset_name: str = None):
    """Load stimulus info about story."""
    grids = load_grids_for_stories([story_name], grids_path)
    trfiles = load_generic_trfiles([story_name], trs_path)
    features_object = Features(grids, trfiles)

    if not featureset_name:
        model_data = None
    else:
        featureset = featuresets_dict[featureset_name]
        featureset[0][1]["downsample"] = False
        model_data = get_feature(featureset, features_object)

    # Get words and word times.
    word_presentation_times = features_object.wordseqs[story_name].data_times
    tr_times = features_object.wordseqs[story_name].tr_times

    if featureset_name:
        story_data = model_data[story_name]
        print("story_data", story_data.shape)
        print("word_presentation_times", word_presentation_times.shape)
        assert len(story_data) == len(word_presentation_times)
    else:
        story_data = None

    # Get num_words.
    featureset = featuresets_dict["numwords"]
    num_words_feature = get_feature(featureset, features_object)[story_name]
    return story_data, word_presentation_times, tr_times, num_words_feature


def get_mirrored_matrix(original_matrix: np.ndarray, mirror_length: int):
    """Concatenates mirrored versions of the matrix to the beginning and end.
    Used to avoid edge effects when filtering.

    Parameters:
    ----------
    original_matrix : np.ndarray
        num_samples x num_features matrix of original matrix values.
    mirror_length : int
        Length of mirrored segment. If longer than num_samples, num_samples is used as mirror_length.

    Returns:
    -------
    mirrored_matrix : np.ndarray
        (num_samples + 2 * mirror_length) x num_features mirrored matrix.
    """
    mirrored_matrix = np.concatenate(
        [
            original_matrix[:mirror_length][::-1],
            original_matrix,
            original_matrix[-mirror_length:][::-1],
        ],
        axis=0,
    )
    return mirrored_matrix


def get_unmirrored_matrix(
    mirrored_matrix: np.ndarray, mirror_length: int, original_num_samples: int
):
    """Retrieves portion of mirrored matrix corresponding to original matrix.
    Parameters:
    ----------
    mirrored_matrix : np.ndarray
        (original_num_samples + 2 * mirror_length) x num_features mirrored matrix.
    mirror_length : int
        Length of mirrored segment. If longer than num_samples, num_samples is used as mirror_length.
    original_num_sample : int
        Number of samples in original (pre-mirroring) matrix.

    Returns:
    -------
    unmirrored_matrix : np.ndarray
        original_num_samples x num_features unmirrored matrix.
    """
    return mirrored_matrix[mirror_length : mirror_length + original_num_samples]
