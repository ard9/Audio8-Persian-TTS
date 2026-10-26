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


def test_central_kurdish_numbers_and_letters():
    assert fa.normalize_ckb("ساڵی ١٤٠٢") == "ساڵی هەزار و چوار سەد و دوو"
    assert fa.normalize_ckb("٢٥٪ داشکاندن") == "لەسەدا بیست و پێنج داشکاندن"
    assert fa.normalize_ckb("نرخی 3.5 دۆلار") == "نرخی سێ پۆینت پێنج دۆلار"
    assert fa.normalize_ckb("٢٠٬٠٠٠ کەس") == "بیست هەزار کەس"
    assert fa.normalize_ckb("كتێبي ھەر") == "کتێبی هەر"  # Arabic yeh/kaf, heh doachashmee
    assert fa.normalize_ckb("ده‌نووسم") == "دەنووسم"  # heh + ZWNJ written for ە


def test_central_kurdish_keeps_sorani_letters():
    sentence = "هاتینە کتێبی دووهەمی نووسین بە ئاوی خۆڵەمێش و بەشی بیستەم."
    assert fa.normalize_ckb(sentence) == sentence
    # The Persian rules would not keep these letters intact, so the two must stay separate.
    assert fa.normalize_text("ڕۆژ ٣", "ckb") == "ڕۆژ سێ"
    assert fa.normalize_text("روز ۳", "fa") == "روز سه"


def test_central_kurdish_number_words():
    assert fa.ckb_number_words(0) == "سفر"
    assert fa.ckb_number_words(17) == "حەڤدە"
    assert fa.ckb_number_words(121) == "سەد و بیست و یەک"
    assert fa.ckb_number_words(2300) == "دوو هەزار و سێ سەد"
    assert fa.ckb_number_words(1_000_000) == "یەک ملیۆن"


def test_unknown_language_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        fa.normalize_text("x", "xx")
