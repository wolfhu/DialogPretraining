# encoding: utf-8

import re


def regexp(sentence):
    """
    去除url、email、手机号
    :param sentence: input sentence
    :return: output sentence with url, email and phone number removed.
    """
    # url
    regexp_url = u"(https?|ftp|file|ttp)://[-A-Za-z0-9+&@#</%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]|www.(.*?).+(com|cn|org|htm|html)"
    sentence = re.sub(regexp_url, u'', sentence)
    # email
    regexp_email = u'[a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+)+|[0-9]+qqcom|([0-9]+qq com)|([0-9]+ qq com)'
    sentence = re.sub(regexp_email, u'', sentence)
    # phone num
    regexp_phone = u"1[3|4|5|7|8][0-9]{9}|0\d{2}-\d{8}|0\d{3}-\d{7}0\d{2}\d{8}|0\d{3}\d{7}"
    sentence = re.sub(regexp_phone, u'', sentence)
    return sentence


def contain_url(string):
    """
    判断是否包含url
    """
    pattern = '[a-zA-Z0-9.\/:]{1,70}[-]{0,70}[a-zA-Z0-9.\/:]{1,70}(\.com|\.net|\.cn|\.org|\.gov|\.xyz)' \
              '[a-zA-Z0-9\/]{0,256}(\.htm|\.html|\.jsp|\.php|\.asp|\.jpg|\.jpeg|\.png){0,1}'
    if re.search(pattern, string) is None:
        return False
    else:
        return True


def contain_email(string):
    pattern = '[a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+)+|[0-9]+qqcom|([0-9]+qq com)|([0-9]+ qq com)'
    if re.search(pattern, string) is None:
        return False
    else:
        return True


def contain_phone(string):
    pattern = '1[3|4|5|7|8][0-9]{9}|0\d{2}-\d{8}|0\d{3}-\d{7}0\d{2}\d{8}|0\d{3}\d{7}'
    if re.search(pattern, string) is None:
        return False
    else:
        return True


def contain_qq(string):
    pattern = 'Q{1,2}[0-9]{1,11}'
    if re.search(pattern, string) is None:
        return False
    else:
        return True


def contain_long_repeat_num(string):
    """
    判断是否包含长距离（7个以上）重复的数字
    """
    pattern = '[0-9]{7,}'
    if re.search(pattern, string) is None:
        return False
    else:
        return True


def remove_html_tag(sentence):
    """
    去除所有被 "<", ">" 包含的文本（HTML标签）
    :param sentence:
    :return: 去除html 标签的句子
    """
    pattern = u"(<.*?>){1,}"
    sentence = re.sub(pattern, u" ", sentence)
    return sentence


def remove_str_in_brackets(sentence):
    """
    去除中英文圆括号中的内容
    """
    pattern = u"\(.*?\)"
    sentence = re.sub(pattern, u"", sentence)
    pattern = u"（.*?）"
    sentence = re.sub(pattern, u"", sentence)
    return sentence


def remove_str_in_square_brackets(sentence):
    """
    去除中英文方括号中的内容
    """
    pattern = u"\[.*?\]"
    sentence = re.sub(pattern, u"", sentence)
    pattern = u"【.*?】"
    sentence = re.sub(pattern, u"", sentence)
    return sentence


def remove_book_title_mark(sentence):
    """
    去除中文书名好中的内容
    """
    pattern = u"《.*?》"
    sentence = re.sub(pattern, u"", sentence)
    return sentence


def remove_escape_character(sentence):
    """
    去除转义字符
    :param sentence: input sentence
    :return: sentence removed escape character
    """
    pattern = u"&[a-zA-Z]{1,10};"
    sentence = re.sub(pattern, u"", sentence)
    pattern = u"&#[0-9]{1,10};"
    sentence = re.sub(pattern, u"", sentence)
    return sentence


def remove_time_stamp(sentence):
    """
    去除时间戳
    仅限 yyyy-mm-dd HH:MM:SS 格式
    """
    pattern = u"[0-9]{2,4}-[0-9]{1,2}-[0-9]{1,2}"
    sentence = re.sub(pattern, u"", sentence)
    pattern = u"[0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2}"
    sentence = re.sub(pattern, u"", sentence)
    return sentence


def strQ2B(ustring):
    """把字符串全角转半角"""
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)


def strB2Q(ustring):
    """把字符串全角转半角"""
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 32:  # 全角空格直接转换
                inside_code = 12288
            elif (inside_code >= 33 and inside_code <= 126):  # 全角字符（除空格）根据关系转化
                inside_code += 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)


def split_sentence(para):
    para = re.sub('([，。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([，。！？\?][”’])([^。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")


def remove_blank(s):
    res = ""
    for ch in s:
        if ch != " " and ch != ' ' and ch != '　':
            res += ch
    return res


def remove_qq(string):
    """
    去除QQ号
    """
    return re.sub('Q{1,2}：{0,}[0-9]{1,11}', '', string)


def remove_repeat_ch(string):
    """
    字符连续重复三次以上，则删除多余的重复字符
    """
    if len(string) == 0:
        return string
    res = string[0]
    repeat_count = 1
    prev = string[0]
    for ch in string[1:]:
        if ch == prev:
            repeat_count += 1
        else:
            repeat_count = 1
            prev = ch
        if repeat_count <= 3:
            res += ch
    return res


def remove_repeat_punc(string):
    """
    puncutation连续重复1次以上，则删除多余的重复字符
    """
    punctuation_list = {",",".","!","?","。","，","！","？"}
    if len(string) == 0:
        return string
    res = string[0]
    repeat_count = 1
    prev = string[0]
    for ch in string[1:]:
        if ch in ch == prev:
            repeat_count += 1
        else:
            repeat_count = 1
            prev = ch
        if repeat_count <= 1:
            res += ch
    return res

def str_extreme_clean(string):
    """
    终极字符串清洗
    """
    string = regexp(string)
    string = remove_html_tag(string)
    # string = remove_str_in_square_brackets(string)
    string = remove_escape_character(string)
    string = remove_qq(string)
    string = remove_repeat_ch(string)
    string = remove_repeat_punc(string)
    string = remove_time_stamp(string)
    string = string.strip()
    return string


def contain_blocked_str(string):
    """
    是否包含URL、email、电话号码、QQ号"""
    if contain_url(string) or contain_email(string) or contain_phone(string) \
            or contain_qq(string) or contain_long_repeat_num(string):
        return True
    else:
        return False


def edit_distance(word1, word2):
    """
    编辑距离
    :type word1: str
    :type word2: str
    :rtype: int
    """
    n = len(word1)
    m = len(word2)

    # if one of the strings is empty
    if n * m == 0:
        return n + m

    # array to store the convertion history
    d = [[0] * (m + 1) for _ in range(n + 1)]

    # init boundaries
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j

    # DP compute
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            left = d[i - 1][j] + 1
            down = d[i][j - 1] + 1
            left_down = d[i - 1][j - 1]
            if word1[i - 1] != word2[j - 1]:
                left_down += 1
            d[i][j] = min(left, down, left_down)

    return d[n][m]


def n_gram_overlap(string1, string2, n=2):
    if n == 1:
        gram_set1 = set(string1)
        gram_set2 = set(string2)
    elif n in (2, 3):
        gram_set1 = set()
        gram_set2 = set()
        for idx in range(len(string1) - n):
            gram_set1.add(string1[idx:idx+n])
        for idx in range(len(string2) - n):
            gram_set2.add(string2[idx:idx+n])
    else:
        raise Exception('n must be 1, 2 or 3.')
    # print('set1: ' + str(list(gram_set1)))
    # print('set2: ' + str(list(gram_set2)))
    overlap_count = 0.0
    for gram in gram_set1:
        if gram in gram_set2:
            overlap_count += 1
    return overlap_count / min(len(gram_set1), len(gram_set2))


punctuation_list = [',', '.', '?', '!', '\'', '\"', ':', ';', '(', ')', '[', ']', '{',  '}', '&',
                    '，', '。', '？', '！', '《', '》', '‘', '’', '“', '”', '：', '：', '…',
                    '（',  '）',  '【',  '】', ]


def is_punctuation(uchar):
    """
    判断给定的单个字符是否是中英文标点
    """
    if uchar in punctuation_list:
        return True
    else:
        return False


def is_chinese(uchar):
    """
    判断一个unicode字符是否是汉字
    """
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False


def is_alphabet(uchar):
    """
    判断一个unicode字符是否是汉字
    """
    if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
        return True
    else:
        return False


def chinese_rate(string):
    chinese_count = 0.0
    for ch in string:
        if is_chinese(ch):
            chinese_count += 1
    return chinese_count / len(string)


def alpha_rate(string):
    alpha_count = 0.0
    for ch in string:
        if is_alphabet(ch):
            alpha_count += 1
    return alpha_count / len(string)


def digit_rate(string):
    digit_count = 0.0
    for ch in string:
        if ch.isdigit():
            digit_count += 1
    return digit_count / len(string)


def to_sentences(paragraph):
    """由段落切分成句子，段落分句"""

    def __merge_symmetry(sentences, symmetry=('“', '”')):
        '''合并对称符号，如双引号'''
        effective_ = []
        merged = True
        for index in range(len(sentences)):
            if symmetry[0] in sentences[index] and symmetry[1] not in sentences[index]:
                merged = False
                effective_.append(sentences[index])
            elif symmetry[1] in sentences[index] and not merged:
                merged = True
                effective_[-1] += sentences[index]
            elif symmetry[0] not in sentences[index] and symmetry[1] not in sentences[index] and not merged:
                effective_[-1] += sentences[index]
            else:
                effective_.append(sentences[index])

        return [i.strip() for i in effective_ if len(i.strip()) > 0]

    sentences = re.split(r"(？{1,}|。{1,}|！{1,}|!{1,}|\?{1,}|\…{1,})", paragraph)
    sentences.append("")
    sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
    sentences = [i.strip() for i in sentences if len(i.strip()) > 0]

    for j in range(1, len(sentences)):
        if sentences[j][0] == '”':
            sentences[j - 1] = sentences[j - 1] + '”'
            sentences[j] = sentences[j][1:]

    return __merge_symmetry(sentences)
