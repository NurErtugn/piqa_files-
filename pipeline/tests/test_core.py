from fineweb_gems.core import classify, clean_document, hamming, parse_bool, piqa_overlap, segment, simhash

CFG = {"min_useful_tokens": 5, "gem_max_noise": .2, "hidden_gem_min_noise": .3, "hidden_gem_max_noise": .7}

def test_clean_preserves_physical_and_tracks_removed():
    r = clean_document("Accept all cookies\nTighten the bolt to 20 Nm before use.\nTighten the bolt to 20 Nm before use.")
    assert "20 Nm" in r["text_clean"]
    assert "cookies" in r["removed_text"]
    assert r["removed_text"].count("Tighten") == 1

def test_bool_string_false_is_false():
    assert parse_bool("false") is False
    assert parse_bool("true") is True

def test_classification_rules():
    base = {"precision_judge": {"task_relevance": 3, "transferable_knowledge": 3, "risk_flags": {}}, "useful_tokens": 20}
    assert classify({**base, "semantic_noise_fraction": .1}, CFG) == "gem"
    assert classify({**base, "semantic_noise_fraction": .5}, CFG) == "hidden_gem"
    risky = {**base, "semantic_noise_fraction": .1, "precision_judge": {**base["precision_judge"], "risk_flags": {"weapon_instruction": "true"}}}
    assert classify(risky, CFG) == "bad"

def test_simhash_and_segmentation():
    a = simhash("tighten the loose bolt with a wrench until secure")
    b = simhash("tighten the loose bolt with a wrench until secure")
    assert hamming(a, b) == 0
    assert segment("First tighten it. Then test it.")

def test_piqa_overlap():
    phrase = "use a cotton swab to apply the powder without a brush"
    from fineweb_gems.core import ngrams
    assert piqa_overlap(phrase, ngrams(phrase, 4), 4) > .5

