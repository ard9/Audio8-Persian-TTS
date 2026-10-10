import fa_tools as fa


def test_persian_characters_and_digits():
    assert fa.normalize_fa("كتاب ي") == "کتاب ی"
    assert fa.normalize_fa("سال ۱۴۰۲") == "سال هزار و چهارصد و دو"
    assert fa.normalize_fa("۲۵٪ تخفیف") == "بیست و پنج درصد تخفیف"
    assert fa.normalize_fa("قیمت 3.5 دلار") == "قیمت سه ممیز پنج دلار"
    assert fa.normalize_fa("۲۰٬۰۰۰ نفر") == "بیست هزار نفر"


def test_invisible_characters_and_spacing():
    # ZWNJ is removed explicitly because the upstream clean_text() strips it anyway.
    assert fa.normalize_fa("می‌خواهم") == "میخواهم"
    assert fa.normalize_fa("سلام   ، دنیا !") == "سلام، دنیا!"
    assert fa.normalize_fa("کتابَ") == "کتاب"


def test_split_sentences_respects_limit():
    text = "این یک جمله بلند است. " * 20
    chunks = fa.split_sentences(text, max_chars=150)
    assert len(chunks) > 1
    assert all(len(c) <= 150 for c in chunks)
    assert fa.split_sentences("کوتاه") == ["کوتاه"]
    assert fa.split_sentences("   ") == []
