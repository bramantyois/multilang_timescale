featuresets_dict = {
    "numwords": [["numwords", {}]],
    "BERT_all": [
        ["contextual_lm", {"model_name": "bert-base-uncased", "layer_num": -1}]
    ],
    "BERT_10": [
        [
            "contextual_lm",
            {
                "model_name": "bert-base-uncased",
                "layer_num": -1,
                "max_seq_length": 10,
                "split_type": "causal_all",
            },
        ]
    ],
    "BERT_100": [
        [
            "contextual_lm",
            {
                "model_name": "bert-base-uncased",
                "layer_num": -1,
                "max_seq_length": 100,
                "split_type": "causal_all",
            },
        ]
    ],
    "mBERT_all": [
        [
            "contextual_lm",
            {
                "model_name": "bert-base-multilingual-uncased",
                "layer_num": -1,
            },
        ]
    ],
    "mBERT_10": [
        [
            "contextual_lm",
            {
                "model_name": "bert-base-multilingual-uncased",
                "layer_num": -1,
                "max_seq_length": 10,
                "split_type": "causal_all",
            },
        ]
    ],
    "mBERT_100": [
        [
            "contextual_lm",
            {
                "model_name": "bert-base-multilingual-uncased",
                "layer_num": -1,
                "max_seq_length": 100,
                "split_type": "causal_all",
            },
        ]
    ],
}

features_sets = [
    "BERT_all",
    "mBERT_all",
]

silence_length = 5
noise_trim_length = 5
train_stories = [
    "alternateithicatom",
    "avatar",
    "howtodraw",
    "legacy",
    "life",
    "myfirstdaywiththeyankees",
    "naked",
    "odetostepfather",
    "souls",
    "undertheinfluence",
]
test_stories = ["wheretheressmoke"]

periods = [
    "2_4_words",
    "4_8_words",
    "8_16_words",
    "16_32_words",
    "32_64_words",
    "64_128_words",
    "128_256_words",
    "256+ words",
]

frequency_to_period_name_dict = {
    0.375: "2_4_words",
    0.1875: "4_8_words",
    0.09375: "8_16_words",
    0.046875: "16_32_words",
    0.0234375: "32_64_words",
    0.01171875: "64_128_words",
    0.005859375: "128_256_words",
    0.00390625: "256+ words",
}

frequency_str_to_period_name_dict = {
    "0.375": "2_4_words",
    "0.1875": "4_8_words",
    "0.09375": "8_16_words",
    "0.046875": "16_32_words",
    "0.0234375": "32_64_words",
    "0.01171875": "64_128_words",
    "0.005859375": "128_256_words",
    "0.00390625": "256+ words",
}

bad_words = [
    "br",
    "lg",
    "ls",
    "ns",
    "sp",
    "ig",
    "cg",
    "sl",
    "",
    "ls)",
    "(br",
    "ns_ap",
    "ig",
]
bad_words_with_sentence_boundaries = bad_words + [
    "sentence_start",
    "sentence_end",
    "sentence start",
]
sentence_start_words = ["sentence_start", "sentence start"]
sentence_end_word = "sentence_end"
