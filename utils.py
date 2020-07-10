# coding: utf-8

from pyltp import SentenceSplitter


def split_doc(doc):
    doc = doc.strip().replace(u'\u3000', u'').replace(u'\\n', u'。').replace(' ', '。').replace(u'(。)+', u'。')
    return [sent for sent in SentenceSplitter.split(doc) if len(sent) > 1]