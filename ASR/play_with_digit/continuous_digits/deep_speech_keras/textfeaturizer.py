import codecs


class TextFeaturizer(object):
    def __init__(self, vocabulary):
        self.token_to_index = {}
        self.index_to_token = {}
        self.tokens = []
        lines = []
        with codecs.open(vocabulary, "r", "utf-8") as f:
            lines.extend(f.readlines())
        index = 0
        for line in lines:
            if line.startswith("#"):
                continue
            line = line.strip()  # 去掉\n
            self.token_to_index[line] = index
            self.index_to_token[index] = line
            self.tokens.append(line)
            index += 1
