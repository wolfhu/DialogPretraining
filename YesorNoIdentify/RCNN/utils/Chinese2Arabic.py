chs_arabic_map = {u'零':0, u'一':1, u'二':2, u'三':3, u'四':4,
        u'五':5, u'六':6, u'七':7, u'八':8, u'九':9,
        u'十':10, u'百':100, u'千':10 ** 3, u'万':10 ** 4,
        u'〇':0, u'壹':1, u'贰':2, u'叁':3, u'肆':4,
        u'伍':5, u'陆':6, u'柒':7, u'捌':8, u'玖':9,
        u'拾':10, u'佰':100, u'仟':10 ** 3, u'萬':10 ** 4}

def convertChineseDigitsToArabic (chinese_digits, encoding="utf-8"):

    result  = 0
    tmp     = 0
    hnd_mln = 0
    for count in range(len(chinese_digits)):
        curr_char  = chinese_digits[count]
        curr_digit = chs_arabic_map.get(curr_char, None)
        # meet 「亿」 or 「億」
        if curr_digit == 10 ** 8:
            result  = result + tmp
            result  = result * curr_digit
            # get result before 「亿」 and store it into hnd_mln
            # reset `result`
            hnd_mln = hnd_mln * 10 ** 8 + result
            result  = 0
            tmp     = 0
        # meet 「万」 or 「萬」
        elif curr_digit == 10 ** 4:
            result = result + tmp
            result = result * curr_digit
            tmp    = 0
        # meet 「十」, 「百」, 「千」 or their traditional version
        elif curr_digit >= 10:
            tmp    = 1 if tmp == 0 else tmp
            result = result + curr_digit * tmp
            tmp    = 0
        # meet single digit
        elif curr_digit is not None:
            tmp = tmp * 10 + curr_digit
        else:
            return result
    result = result + tmp
    result = result + hnd_mln
    return result


if __name__ == '__main__':
    x = convertChineseDigitsToArabic("一三十")
    print(x)